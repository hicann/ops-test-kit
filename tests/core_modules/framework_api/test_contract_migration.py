from __future__ import annotations

"""Task 7: atomic contract migration.

After Task 7:
  - device_name() returns the hardware MODEL (via torch.get_device_name), not
    the torch_lib segment ('npu'/'cuda'/'cpu'). CPU has no model, so its
    device_name override stays == alias() ('cpu').
  - soc_version() is removed (merged into device_name).
  - All string-literal role comparisons (== 'npu'/'gpu'/'cpu') in
    framework_api are gone, replaced by is_npu()/alias()/use_device().
  - get_profiler uses is_npu() + profile.get('profiler') instead of
    device_name() string compares.
"""
import subprocess

from ttk.core_modules.framework_api.backends.cpu_backend import CpuTorchBackend
from ttk.core_modules.framework_api.backends.npu_backend import NpuTorchBackend


def test_cpu_device_name_is_cpu_alias():
    """Task 7 后 cpu device_name 走 alias（cpu 无 get_device_name）。"""
    cb = CpuTorchBackend(); cb.torch_lib = "cpu"; cb.profile = {}
    assert cb.use_device() is False
    assert cb.is_npu() is False
    assert cb.alias() == "cpu"
    # cpu has no torch.cpu.get_device_name -> override keeps alias()
    assert cb.device_name() == "cpu"


def test_soc_version_method_removed():
    """soc_version 合并进 device_name，base/backend 不再暴露 soc_version。"""
    cb = CpuTorchBackend(); cb.torch_lib = "cpu"; cb.profile = {}
    assert not hasattr(cb, "soc_version"), \
        "soc_version must be removed in Task 7 (merged into device_name)"


def test_soc_series_default_is_device_name_model():
    """默认 soc_series() == device_name()（型号）；NpuTorchBackend override short。"""
    cb = CpuTorchBackend(); cb.torch_lib = "cpu"; cb.profile = {}
    # base default degrades soc_series to device_name (model)
    assert cb.soc_series() == cb.device_name()


def test_no_string_comparison_on_role():
    """grep 确认无 =='npu'/'gpu'/'cpu' 角色字符串逻辑残留 in framework_api。

    Role comparisons (alias()/device_name()/soc_series() == 'npu'/'gpu'/'cpu')
    are forbidden — routing goes through is_npu()/alias()/use_device().
    torch_lib value matches (e.g. ``torch_lib == "npu"`` for class derivation in
    _build, ``torch_lib == "cpu"`` for the cpu-skip) are ALLOWED: torch_lib is
    the torch module attribute, not a role.

    Implementation note: _build derives the backend class from torch_lib (cuda/
    mlu/musa -> XpuTorchBackend, npu -> NpuTorchBackend, cpu -> CpuTorchBackend);
    the alias is config-driven (_segment_name = yaml segment key), so no
    'xpu'/'gpu' role string is ever compared.
    """
    r = subprocess.run(
        ["grep", "-rnE", "--include=*.py",
         r'''== ?["'](npu|gpu|cpu)["']''',
         "ttk/core_modules/framework_api/"],
        capture_output=True, text=True,
    )
    # filter out allowed torch_lib value matches + docstring example text.
    residue = [
        ln for ln in r.stdout.splitlines()
        if ln and "torch_lib" not in ln
        and "yields alias" not in ln
    ]
    assert not residue, f"string comparison residue:\n" + "\n".join(residue)


def test_no_inequality_comparison_on_role():
    """grep 确认无 !='npu'/'gpu'/'cpu' 角色字符串逻辑残留 in framework_api。"""
    r = subprocess.run(
        ["grep", "-rnE", "--include=*.py",
         r'''!= ?["'](npu|gpu|cpu)["']''',
         "ttk/core_modules/framework_api/"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, f"string inequality residue:\n{r.stdout}"


def test_no_soc_version_residue_in_ttk():
    """全仓 grep 确认 .soc_version( 调用零残留（已合并进 device_name）。"""
    r = subprocess.run(
        ["grep", "-rnE", r'\.soc_version\(', "ttk/"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, f".soc_version() residue:\n{r.stdout}"


def test_get_profiler_uses_is_npu_and_profile_not_device_name():
    """get_profiler 不再读 device_name()；用 is_npu() + profile['profiler']。

    torch.* + non-NPU + profile.profiler != 'builtin' -> TorchProfiler.
    A backend whose device_name() returns a model name (not 'cpu'/'gpu')
    must still resolve correctly -> proves no string compare on device_name.
    """
    from ttk.core_modules.framework_api.profiler import (
        get_profiler, TorchProfiler, WallClockProfiler,
    )

    class _ModelNameBackend:
        # device_name now returns a MODEL, not a segment. Old code compared
        # this to 'gpu'/'npu'/'cpu'; Task 7 must NOT read it for routing.
        torch_lib = "cuda"
        # _build injects torch_lib into profile; mirror that invariant here so
        # TorchProfiler.__init__ (which reads profile["torch_lib"] per §5.3) works.
        profile = {"torch_lib": "cuda",
                   "profiler": {"activities": ["CPU", "CUDA"]}}

        def device_name(self, dev_id=0):
            return "AscendWhatever-Model-Name"  # deliberately non-segment

        def alias(self):
            return "xpu"

        def is_npu(self):
            return False

    prof = get_profiler("torch.add", _ModelNameBackend())
    assert isinstance(prof, TorchProfiler)
    # unknown prefix -> WallClock regardless of backend
    assert isinstance(get_profiler("tf.foo", _ModelNameBackend()), WallClockProfiler)
