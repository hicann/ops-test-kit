"""third_party style: dict with 'tensorflow' key (归一化→tf)."""
import numpy


class AddTensorflowKeySpec:
    golden = "torch.add"
    third_party = {"tensorflow": "tf.raw_ops.Add", "torch": "torch.add"}


__spec__ = {"add": "AddTensorflowKeySpec"}
