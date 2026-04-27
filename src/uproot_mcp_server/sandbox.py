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
- Enforces a configurable wall-clock timeout by running the kernel in a
  subprocess that is forcefully terminated (SIGTERM then SIGKILL) on expiry

Only ``np`` (numpy) and ``ak`` (awkward-array) are available as named packages.
The subprocess boundary also prevents any side-effects (file I/O, etc.) from
escaping to the parent server process.
"""

from __future__ import annotations

import marshal
import multiprocessing
import types
from typing import Any

import awkward as ak
import numpy as np
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.Guards import guarded_unpack_sequence


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KernelError(ValueError):
    """Raised when a kernel fails to compile or execute."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Use the "spawn" start method explicitly. The default on Linux is "fork",
# which clones the parent's memory including held locks; forking from a
# process running an asyncio event loop (FastMCP stdio) deadlocks the child
# on inherited locks, so every kernel call would hang until the timeout.
# "spawn" launches a fresh interpreter via exec — no inherited state.
_MP_CTX = multiprocessing.get_context("spawn")

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
        "_unpack_sequence_": guarded_unpack_sequence,
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


def _reduce_worker(
    code_bytes: bytes,
    a: Any,
    b: Any,
    conn: Any,
) -> None:
    """Worker function executed in a subprocess to run the reduce function.

    Compiles and runs the reduce code, then sends the result (or an error
    description) through *conn*.  Expects a callable named ``reduce`` that
    accepts two arguments.
    """
    code_obj = marshal.loads(code_bytes)
    globs = _make_safe_globals()

    try:
        exec(code_obj, globs)  # noqa: S102 — intentional restricted exec
    except Exception as exc:
        conn.send(("def_error", str(exc)))
        conn.close()
        return

    if "reduce" not in globs or not callable(globs["reduce"]):
        conn.send(("missing", None))
        conn.close()
        return

    try:
        result = globs["reduce"](a, b)
        conn.send(("ok", result))
    except Exception as exc:  # noqa: BLE001
        conn.send(("run_error", str(exc)))
    conn.close()


def _kernel_worker(
    code_bytes: bytes,
    branches_data: dict[str, Any],
    conn: Any,
) -> None:
    """Worker function executed in a subprocess to run the kernel.

    Compiles and runs the kernel code, then sends the result (or an error
    description) through *conn* (a :class:`multiprocessing.Connection`).
    The calling process reads back ``("ok", result)``,
    ``("def_error", message)``, ``("missing", None)``, or
    ``("run_error", message)``.
    """
    code_obj = marshal.loads(code_bytes)
    globs = _make_safe_globals()

    try:
        exec(code_obj, globs)  # noqa: S102 — intentional restricted exec
    except Exception as exc:
        conn.send(("def_error", str(exc)))
        conn.close()
        return

    if "kernel" not in globs or not callable(globs["kernel"]):
        conn.send(("missing", None))
        conn.close()
        return

    try:
        result = globs["kernel"](branches_data)
        conn.send(("ok", result))
    except Exception as exc:  # noqa: BLE001
        conn.send(("run_error", str(exc)))
    conn.close()


def execute_kernel(
    code_obj: types.CodeType,
    branches_data: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> Any:
    """Execute a compiled kernel in a restricted subprocess.

    The kernel runs in a separate :class:`multiprocessing.Process`.  If it
    does not complete within *timeout* seconds the process is first sent
    ``SIGTERM`` and then ``SIGKILL``, guaranteeing that runaway kernels
    cannot consume CPU indefinitely.

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
    code_bytes = marshal.dumps(code_obj)
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)

    proc = _MP_CTX.Process(
        target=_kernel_worker,
        args=(code_bytes, branches_data, child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()  # Parent only reads; close the write end here.

    if not parent_conn.poll(timeout):
        # Timeout: forcefully terminate the subprocess.
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(1.0)
        raise KernelError(f"Kernel execution timed out after {timeout:.1f} s")

    proc.join()

    try:
        status, payload = parent_conn.recv()
    except EOFError as exc:
        raise KernelError(
            f"Kernel process exited unexpectedly (exit code: {proc.exitcode})"
        ) from exc

    if status == "ok":
        return payload
    if status == "def_error":
        raise KernelError(f"Kernel definition failed: {payload}")
    if status == "missing":
        raise KernelError("Kernel code must define a callable named 'kernel'")
    # status == "run_error"
    raise KernelError(f"Kernel raised an exception: {payload}")


def execute_reduce(
    code_obj: types.CodeType,
    a: Any,
    b: Any,
    *,
    timeout: float = 30.0,
) -> Any:
    """Execute a compiled ``reduce(a, b)`` function in a restricted subprocess.

    The reduce function runs in a separate :class:`multiprocessing.Process`
    with the same RestrictedPython sandbox as :func:`execute_kernel`.

    Parameters
    ----------
    code_obj:
        Code object from :func:`compile_kernel` — must define ``def reduce(a, b): ...``.
    a:
        Left operand (accumulated result so far).
    b:
        Right operand (next partial result).
    timeout:
        Wall-clock execution limit in seconds (default: 30).

    Returns
    -------
    Any
        Return value of ``reduce(a, b)``.

    Raises
    ------
    KernelError
        If the reduce definition fails, no callable named ``reduce`` is found,
        execution raises an exception, or the timeout is exceeded.
    """
    code_bytes = marshal.dumps(code_obj)
    parent_conn, child_conn = _MP_CTX.Pipe(duplex=False)

    proc = _MP_CTX.Process(
        target=_reduce_worker,
        args=(code_bytes, a, b, child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()

    if not parent_conn.poll(timeout):
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(1.0)
        raise KernelError(f"Reduce execution timed out after {timeout:.1f} s")

    proc.join()

    try:
        status, payload = parent_conn.recv()
    except EOFError as exc:
        raise KernelError(
            f"Reduce process exited unexpectedly (exit code: {proc.exitcode})"
        ) from exc

    if status == "ok":
        return payload
    if status == "def_error":
        raise KernelError(f"Reduce definition failed: {payload}")
    if status == "missing":
        raise KernelError("reduce_code must define a callable named 'reduce'")
    raise KernelError(f"Reduce raised an exception: {payload}")
