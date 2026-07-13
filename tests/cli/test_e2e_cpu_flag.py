"""Task 6: --cpu flag replaces --backend (D-scheme hardware-neutral abstraction).

SWITCHES drops the legacy ``backend_name`` slot in favor of a boolean
``force_cpu`` slot, and the e2e CLI surface switches from ``--backend <name>``
to ``--cpu`` (store_true). ``get_backend`` is collapsed to a single
``force_cpu`` kwarg.
"""
from ttk.utilities.classes import SWITCHES


def test_switches_force_cpu_slot():
    assert "force_cpu" in SWITCHES.__slots__
    assert "backend_name" not in SWITCHES.__slots__


def test_switches_force_cpu_default():
    sw = SWITCHES()
    assert sw.force_cpu is False


def test_switches_force_cpu_assignable_and_pickle_roundtrip():
    import pickle
    sw = SWITCHES()
    sw.force_cpu = True
    revived = pickle.loads(pickle.dumps(sw))
    assert revived.force_cpu is True
