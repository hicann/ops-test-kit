import pickle
from ttk.utilities.classes import SWITCHES


def test_switches_has_config_path_and_provider_filter():
    sw = SWITCHES()
    assert sw.config_path is None
    assert sw.provider_filter is None


def test_switches_config_fields_assignable():
    sw = SWITCHES()
    sw.config_path = "/tmp/x.yaml"
    sw.provider_filter = "torch,tf"
    assert sw.config_path == "/tmp/x.yaml"
    assert sw.provider_filter == "torch,tf"


def test_switches_pickle_roundtrip_with_config_fields():
    """forkserver 经 pickle 传 SWITCHES 到 worker——config 字段必须存活。"""
    sw = SWITCHES()
    sw.config_path = "/tmp/x.yaml"
    sw.provider_filter = "torch"
    revived = pickle.loads(pickle.dumps(sw))
    assert revived.config_path == "/tmp/x.yaml"
    assert revived.provider_filter == "torch"


def test_switches_force_cpu_pickle_roundtrip():
    """force_cpu（替 backend_name，经 SWITCHES pickle 传 worker）必须存活。"""
    sw = SWITCHES()
    sw.force_cpu = True
    revived = pickle.loads(pickle.dumps(sw))
    assert revived.force_cpu is True
    # legacy slot gone for good
    assert not hasattr(revived, "backend_name")
