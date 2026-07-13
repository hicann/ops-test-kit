"""Tests for ttk.remote.server.config — loader + hardware detection."""
import os
import tempfile

import pytest


class TestLoadServerConfig:
    def test_defaults_when_no_yaml(self, monkeypatch):
        from ttk.remote.server.config import load_server_config
        cfg = load_server_config("/nonexistent/path.yaml")
        assert cfg["bind"] == "127.0.0.1"
        assert cfg["port"] == 9090
        assert cfg["max_concurrent"] == 16
        assert cfg["run_deadline_s"] == 300
        assert cfg["sandbox"] == "none"
        assert cfg["hardware_config"] == {}
        assert cfg["tls_enabled"] is False
        assert cfg["docker_network"] == "none"

    def test_yaml_overrides_defaults(self, tmp_path):
        yaml = tmp_path / "xpu_server.yaml"
        yaml.write_text(
            "server:\n"
            "  port: 7070\n"
            "hardware:\n"
            "  gpu:\n"
            "    dev_prefix: nvidia\n"
            "    torch_lib: cuda\n"
            "    torch_profiler:\n"
            "      activities: [CPU, CUDA]\n"
        )
        from ttk.remote.server.config import load_server_config
        cfg = load_server_config(str(yaml))
        assert cfg["port"] == 7070
        assert "gpu" in cfg["hardware_config"]
        assert cfg["hardware_config"]["gpu"]["torch_lib"] == "cuda"

    def test_load_rejects_non_mapping_hardware(self, tmp_path):
        (tmp_path / "x.yaml").write_text("hardware: foo\n")
        from ttk.remote.server.config import load_server_config
        with pytest.raises(ValueError, match="must be a mapping"):
            load_server_config(str(tmp_path / "x.yaml"))

    def test_load_rejects_uppercase_segment(self, tmp_path):
        (tmp_path / "x.yaml").write_text(
            "hardware:\n  GPU:\n    dev_prefix: nvidia\n    torch_lib: cuda\n    torch_profiler: {activities: [CPU]}\n")
        from ttk.remote.server.config import load_server_config
        with pytest.raises(ValueError, match="not lowercase"):
            load_server_config(str(tmp_path / "x.yaml"))

    def test_load_rejects_non_mapping_segment(self, tmp_path):
        (tmp_path / "x.yaml").write_text("hardware:\n  gpu: cuda\n")
        from ttk.remote.server.config import load_server_config
        with pytest.raises(ValueError, match="must be a mapping"):
            load_server_config(str(tmp_path / "x.yaml"))

    def test_load_rejects_missing_required_field(self, tmp_path):
        (tmp_path / "x.yaml").write_text("hardware:\n  gpu:\n    dev_prefix: nvidia\n    torch_lib: cuda\n")
        from ttk.remote.server.config import load_server_config
        with pytest.raises(ValueError, match="missing required field"):
            load_server_config(str(tmp_path / "x.yaml"))

    def test_providers_from_yaml(self):
        yaml_content = """providers:
  - torch
  - flash_attn
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        try:
            from ttk.remote.server.config import load_server_config
            cfg = load_server_config(yaml_path)
            assert cfg["providers"] == ["torch", "flash_attn"]
        finally:
            os.unlink(yaml_path)


class TestDetectHardware:
    def test_detect_gpu(self, monkeypatch):
        monkeypatch.setattr(os, "listdir", lambda p: ["nvidia0", "nvidia5", "nvidiactl", "tty"])
        from ttk.remote.server.config import detect_hardware
        cfg = {"gpu": {"dev_prefix": "nvidia"}, "mlu": {"dev_prefix": "cambricon"}}
        assert detect_hardware(cfg) == ("gpu", [0, 5])

    def test_detect_mlu(self, monkeypatch):
        monkeypatch.setattr(os, "listdir", lambda p: ["cambricon0", "tty"])
        from ttk.remote.server.config import detect_hardware
        assert detect_hardware({"gpu": {"dev_prefix": "nvidia"}, "mlu": {"dev_prefix": "cambricon"}}) == ("mlu", [0])

    def test_detect_cpu_fallback_empty(self, monkeypatch):
        monkeypatch.setattr(os, "listdir", lambda p: [])
        from ttk.remote.server.config import detect_hardware
        assert detect_hardware({"gpu": {"dev_prefix": "nvidia"}}) == ("cpu", ["cpu"])

    def test_detect_segment_order(self, monkeypatch):
        # gpu+mlu 都命中，段序 gpu 先（dict 保序）
        monkeypatch.setattr(os, "listdir", lambda p: ["nvidia0", "cambricon0"])
        from ttk.remote.server.config import detect_hardware
        assert detect_hardware({"gpu": {"dev_prefix": "nvidia"}, "mlu": {"dev_prefix": "cambricon"}})[0] == "gpu"

    def test_detect_dev_unreadable(self, monkeypatch):
        def boom(p):
            raise OSError("denied")
        monkeypatch.setattr(os, "listdir", boom)
        from ttk.remote.server.config import detect_hardware
        assert detect_hardware({"gpu": {"dev_prefix": "nvidia"}}) == ("cpu", ["cpu"])

    def test_detect_excludes_control_devices(self, monkeypatch):
        # nvidia-uvm/ctl 无数字尾不命中
        monkeypatch.setattr(os, "listdir", lambda p: ["nvidia0", "nvidia-uvm", "nvidiactl"])
        from ttk.remote.server.config import detect_hardware
        assert detect_hardware({"gpu": {"dev_prefix": "nvidia"}}) == ("gpu", [0])

    def test_scan_dev_ids(self):
        from ttk.remote.server.config import _scan_dev_ids
        assert _scan_dev_ids("nvidia", ["nvidia0", "nvidia5", "nvidia-uvm", "nvidiactl", "tty"]) == [0, 5]
        assert _scan_dev_ids("cambricon", ["cambricon0", "cambricon1"]) == [0, 1]
        assert _scan_dev_ids("nvidia", []) == []
