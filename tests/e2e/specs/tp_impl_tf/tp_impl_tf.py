"""third_party style: TF impl class (spec mode)."""
import numpy


class AddTfImplSpec:
    golden = "torch.add"

    class AddTfImpl:
        """tf impl — called on the XPU server (spec mode)."""
        def __call__(self, x, y, **kwargs):
            import tensorflow as tf
            return [tf.raw_ops.Add(x=x, y=y)]

    third_party = {"tf": AddTfImpl}


__spec__ = {"add": "AddTfImplSpec"}
