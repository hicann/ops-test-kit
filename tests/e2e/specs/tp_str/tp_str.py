"""third_party style: single API string."""
import numpy


class AddStrSpec:
    golden = "torch.add"
    third_party = "torch.add"


__spec__ = {"add": "AddStrSpec"}
