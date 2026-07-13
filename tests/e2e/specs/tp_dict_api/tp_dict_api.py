"""third_party style: dict of API strings (multi-vendor)."""
import numpy


class AddDictApiSpec:
    golden = "torch.add"
    third_party = {"torch": "torch.add", "tf": "tf.raw_ops.Add"}


__spec__ = {"add": "AddDictApiSpec"}
