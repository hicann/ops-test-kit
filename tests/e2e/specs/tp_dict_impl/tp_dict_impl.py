"""third_party style: dict of impl classes (spec mode)."""
import numpy


class AddDictImplSpec:
    golden = "torch.add"

    class AddTorchImpl:
        """torch impl — called on the XPU server (spec mode)."""
        def __call__(self, x, y, **kwargs):
            import torch
            return [torch.add(x, y)]

    third_party = {"torch": AddTorchImpl}


__spec__ = {"add": "AddDictImplSpec"}
