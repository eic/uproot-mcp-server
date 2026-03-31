"""Tests for the sandboxed kernel execution module (uproot_mcp_server.sandbox)."""

from __future__ import annotations

import numpy as np
import pytest

from uproot_mcp_server.sandbox import KernelError, compile_kernel, execute_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "x": rng.normal(0.0, 1.0, 200).astype("float32"),
        "y": rng.normal(0.0, 1.0, 200).astype("float32"),
        "charge": rng.integers(-1, 2, 200).astype("int32"),
    }


def _compile_and_run(code: str, events: dict | None = None) -> object:
    """Convenience: compile + execute a kernel, return its result."""
    code_obj = compile_kernel(code)
    return execute_kernel(code_obj, events or _events())


# ---------------------------------------------------------------------------
# compile_kernel — valid inputs
# ---------------------------------------------------------------------------


class TestCompileKernelValid:
    def test_minimal_kernel_compiles(self):
        code = "def kernel(events):\n    return 1\n"
        assert compile_kernel(code) is not None

    def test_numpy_kernel_compiles(self):
        code = (
            "def kernel(events):\n"
            "    return np.sqrt(events['x']**2 + events['y']**2)\n"
        )
        assert compile_kernel(code) is not None


# ---------------------------------------------------------------------------
# compile_kernel — rejected inputs
# ---------------------------------------------------------------------------


class TestCompileKernelRejected:
    def test_code_too_large_raises(self):
        code = "def kernel(events):\n    return 1\n" + "# " + "x" * 65536
        with pytest.raises(KernelError, match="exceeds"):
            compile_kernel(code)

    def test_syntax_error_raises(self):
        with pytest.raises(KernelError, match="[Ss]yntax"):
            compile_kernel("def kernel(events:\n    return 1\n")


# ---------------------------------------------------------------------------
# execute_kernel — valid kernels
# ---------------------------------------------------------------------------


class TestExecuteKernelValid:
    def test_identity_return(self):
        code = "def kernel(events):\n    return events['x']\n"
        events = _events()
        result = _compile_and_run(code, events)
        np.testing.assert_array_equal(result, events["x"])

    def test_numpy_arithmetic(self):
        code = (
            "def kernel(events):\n"
            "    return np.sqrt(events['x']**2 + events['y']**2)\n"
        )
        events = _events()
        result = _compile_and_run(code, events)
        expected = np.sqrt(events["x"] ** 2 + events["y"] ** 2)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_return(self):
        code = "def kernel(events):\n    return float(np.mean(events['x']))\n"
        events = _events()
        result = _compile_and_run(code, events)
        assert isinstance(result, float)
        assert abs(result - float(np.mean(events["x"]))) < 1e-5

    def test_dict_return(self):
        code = (
            "def kernel(events):\n"
            "    return {'mean': float(np.mean(events['x'])), 'n': len(events['x'])}\n"
        )
        result = _compile_and_run(code)
        assert isinstance(result, dict)
        assert "mean" in result and "n" in result

    def test_builtin_len_available(self):
        code = "def kernel(events):\n    return len(events['x'])\n"
        result = _compile_and_run(code)
        assert result == 200

    def test_inplace_add(self):
        code = (
            "def kernel(events):\n"
            "    total = 0.0\n"
            "    total += float(np.sum(events['x']))\n"
            "    return total\n"
        )
        events = _events()
        result = _compile_and_run(code, events)
        assert abs(result - float(np.sum(events["x"]))) < 1e-4

    def test_for_loop_with_range(self):
        code = (
            "def kernel(events):\n"
            "    acc = []\n"
            "    for i in range(3):\n"
            "        acc.append(i)\n"
            "    return acc\n"
        )
        result = _compile_and_run(code)
        assert result == [0, 1, 2]

    def test_awkward_available(self):
        import awkward as ak

        code = (
            "def kernel(events):\n"
            "    arr = ak.Array([1, 2, 3])\n"
            "    return ak.to_numpy(arr)\n"
        )
        result = _compile_and_run(code)
        np.testing.assert_array_equal(result, [1, 2, 3])


# ---------------------------------------------------------------------------
# execute_kernel — blocked operations
# ---------------------------------------------------------------------------


class TestExecuteKernelBlocked:
    def test_import_blocked(self):
        """import statements must not succeed (no __import__ in builtins)."""
        code = "import os\ndef kernel(events):\n    return os.getcwd()\n"
        with pytest.raises(KernelError):
            code_obj = compile_kernel(code)
            execute_kernel(code_obj, {})

    def test_exec_blocked(self):
        code = "def kernel(events):\n    exec('x = 1')\n    return 1\n"
        with pytest.raises(KernelError):
            code_obj = compile_kernel(code)
            execute_kernel(code_obj, {})

    def test_eval_blocked(self):
        code = "def kernel(events):\n    return eval('1+1')\n"
        with pytest.raises(KernelError):
            code_obj = compile_kernel(code)
            execute_kernel(code_obj, {})

    def test_open_blocked(self):
        code = "def kernel(events):\n    return open('/etc/passwd').read()\n"
        with pytest.raises(KernelError):
            code_obj = compile_kernel(code)
            execute_kernel(code_obj, {})

    def test_dunder_attr_access_blocked(self):
        code = "def kernel(events):\n    return events['x'].__class__\n"
        # RestrictedPython rejects dunder attribute access at compile time
        with pytest.raises(KernelError):
            code_obj = compile_kernel(code)
            execute_kernel(code_obj, _events())

    def test_numpy_module_write_blocked(self):
        code = "def kernel(events):\n    np.sqrt = lambda x: 0\n    return 1\n"
        with pytest.raises(KernelError, match="Cannot modify"):
            code_obj = compile_kernel(code)
            execute_kernel(code_obj, _events())

    def test_awkward_module_write_blocked(self):
        code = "def kernel(events):\n    ak.Array = None\n    return 1\n"
        with pytest.raises(KernelError, match="Cannot modify"):
            code_obj = compile_kernel(code)
            execute_kernel(code_obj, _events())


# ---------------------------------------------------------------------------
# execute_kernel — error handling
# ---------------------------------------------------------------------------


class TestExecuteKernelErrors:
    def test_timeout_enforced(self):
        code = "def kernel(events):\n    while True:\n        pass\n"
        code_obj = compile_kernel(code)
        with pytest.raises(KernelError, match="timed out"):
            execute_kernel(code_obj, {}, timeout=0.5)

    def test_kernel_exception_wrapped_as_kernel_error(self):
        code = "def kernel(events):\n    raise ValueError('physics error')\n"
        code_obj = compile_kernel(code)
        with pytest.raises(KernelError, match="physics error"):
            execute_kernel(code_obj, {})

    def test_missing_kernel_function_raises(self):
        code = "x = 42\n"
        code_obj = compile_kernel(code)
        with pytest.raises(KernelError, match="callable named 'kernel'"):
            execute_kernel(code_obj, {})

    def test_non_callable_kernel_raises(self):
        code = "kernel = 42\n"
        code_obj = compile_kernel(code)
        with pytest.raises(KernelError, match="callable named 'kernel'"):
            execute_kernel(code_obj, {})
