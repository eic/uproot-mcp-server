"""Sandboxed Python kernel execution using RestrictedPython.

Kernels are Python callables of the form::

    def kernel(events):
        ...

where ``events`` is a ``dict[str, array]`` provided by the server.  They run
in a restricted namespace that:

- Blocks all ``import`` statements (no ``__import__`` in builtins)
- Blocks ``exec``, ``eval``, ``open``, ``compile`` and other dangerous builtins
- Blocks explicit dunder (``__``) attribute access in kernel code
- Prevents writes to the injected ``np`` / ``ak`` modules
- Enforces a configurable wall-clock timeout via a daemon thread

Only ``np`` (numpy) and ``ak`` (awkward-array) are available as named packages.
"""

from __future__ import annotations

import math
import threading
import types
from typing import Any

import awkward as ak
import numpy as np
from RestrictedPython import compile_restricted, safe_builtins


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KernelError(ValueError):
    """Raised when a kernel fails to compile or execute."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_CODE_BYTES: int = 65_536  # 64 KB

# Builtins that are safe to expose inside kernels
_ALLOWED_BUILTIN_NAMES: tuple[str, ...] = (
    "len", "range", "list", "dict", "tuple", "set", "frozenset",
    "int", "float", "bool", "str", "bytes", "complex",
    "abs", "min", "max", "sum", "round", "pow",
    "zip", "enumerate", "map", "filter", "sorted", "reversed",
    "isinstance", "issubclass", "type",
    "print",
    "True", "False", "None",
    "Exception", "ValueError", "TypeError", "RuntimeError",
    "IndexError", "KeyError", "AttributeError", "StopIteration",
    "NotImplementedError", "ArithmeticError", "OverflowError",
    "ZeroDivisionError",
)


# ---------------------------------------------------------------------------
# Restricted execution environment
# ---------------------------------------------------------------------------


def _make_safe_globals() -> dict[str, Any]:
    """Build the restricted global namespace for kernel execution.

    Includes:

    - ``np``: numpy
    - ``ak``: awkward-array
    - Whitelisted subset of Python builtins
    - RestrictedPython guard callables (``_getattr_``, ``_getitem_``, etc.)
    """
    restricted_builtins: dict[str, Any] = {
        name: safe_builtins[name]
        for name in _ALLOWED_BUILTIN_NAMES
        if name in safe_builtins
    }

    def _getattr_(obj: Any, name: str) -> Any:
        """Block dunder / private attribute access in kernel code."""
        if name.startswith("_"):
            raise AttributeError(
                f"Access to private/dunder attribute '{name}' is not allowed in kernels"
            )
        return getattr(obj, name)

    def _getitem_(obj: Any, key: Any) -> Any:
        return obj[key]

    def _getiter_(obj: Any) -> Any:
        return iter(obj)

    def _write_(obj: Any) -> Any:
        """Prevent writes to the injected ``np`` and ``ak`` module objects."""
        if obj is np or obj is ak:
            raise AttributeError(
                "Cannot modify the numpy or awkward modules inside a kernel"
            )
        return obj

    def _inplacevar_(op: str, x: Any, y: Any) -> Any:
        """Handle augmented assignment operators (``+=``, ``-=``, etc.)."""
        _ops: dict[str, Any] = {
            "+=":  lambda a, b: a + b,
            "-=":  lambda a, b: a - b,
            "*=":  lambda a, b: a * b,
            "/=":  lambda a, b: a / b,
            "//=": lambda a, b: a // b,
            "%=":  lambda a, b: a % b,
            "**=": lambda a, b: a ** b,
            "&=":  lambda a, b: a & b,
            "|=":  lambda a, b: a | b,
            "^=":  lambda a, b: a ^ b,
        }
        if op not in _ops:
            raise TypeError(f"Unsupported in-place operator: {op}")
        return _ops[op](x, y)

    return {
        "__builtins__": restricted_builtins,
        "__name__": "__kernel__",
        "__doc__": None,
        "_getattr_": _getattr_,
        "_getitem_": _getitem_,
        "_getiter_": _getiter_,
        "_write_": _write_,
        "_inplacevar_": _inplacevar_,
        "np": np,
        "ak": ak,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile_kernel(code: str) -> types.CodeType:
    """Validate and compile *code* under RestrictedPython.

    The source must define a callable named ``kernel`` at module level, e.g.::

        def kernel(events):
            px = events["ReconstructedParticles.momentum.x"]
            py = events["ReconstructedParticles.momentum.y"]
            pz = events["ReconstructedParticles.momentum.z"]
            return np.sqrt(px**2 + py**2 + pz**2)

    Parameters
    ----------
    code:
        Python source string containing ``def kernel(events): ...``.

    Returns
    -------
    types.CodeType
        Compiled restricted code object ready for :func:`execute_kernel`.

    Raises
    ------
    KernelError
        If the code exceeds the size limit, contains a syntax error, or
        RestrictedPython rejects it.
    """
    if len(code.encode()) > _MAX_CODE_BYTES:
        raise KernelError(
            f"Kernel code exceeds the maximum allowed size of "
            f"{_MAX_CODE_BYTES // 1024} KB"
        )

    try:
        code_obj = compile_restricted(code, filename="<kernel>", mode="exec")
    except SyntaxError as exc:
        raise KernelError(f"Kernel syntax error: {exc}") from exc
    except Exception as exc:
        raise KernelError(f"Kernel compilation failed: {exc}") from exc

    if code_obj is None:
        raise KernelError(
            "Kernel code contains restricted constructs and could not be compiled. "
            "Ensure no imports, exec, eval, or other forbidden operations are present."
        )

    return code_obj


def execute_kernel(
    code_obj: types.CodeType,
    branches_data: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> Any:
    """Execute a compiled kernel in a restricted sandbox.

    Parameters
    ----------
    code_obj:
        Code object from :func:`compile_kernel`.
    branches_data:
        Mapping of branch name → array data.  Passed as ``events`` to the
        kernel function.
    timeout:
        Wall-clock execution limit in seconds (default: 30).

    Returns
    -------
    Any
        Return value of ``kernel(events)``.

    Raises
    ------
    KernelError
        If the kernel definition fails, no callable named ``kernel`` is found,
        execution raises an exception, or the timeout is exceeded.
    """
    globs = _make_safe_globals()

    try:
        exec(code_obj, globs)  # noqa: S102 — intentional restricted exec
    except Exception as exc:
        raise KernelError(f"Kernel definition failed: {exc}") from exc

    if "kernel" not in globs or not callable(globs["kernel"]):
        raise KernelError("Kernel code must define a callable named 'kernel'")

    kernel_fn = globs["kernel"]
    result_holder: list[Any] = []
    exc_holder: list[BaseException] = []

    def _run() -> None:
        try:
            result_holder.append(kernel_fn(branches_data))
        except Exception as exc:  # noqa: BLE001
            exc_holder.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise KernelError(f"Kernel execution timed out after {timeout:.1f} s")
    if exc_holder:
        raise KernelError(
            f"Kernel raised an exception: {exc_holder[0]}"
        ) from exc_holder[0]
    if not result_holder:
        raise KernelError("Kernel did not return a result")

    return result_holder[0]
