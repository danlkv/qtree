from qtree.optimizer import Tensor, Var
from qtree.system_defs import NP_ARRAY_TYPE
import numpy as np

def test_empty_tensor():
    shape = (2, 3, 4)
    indices = [Var(i, size=s) for i, s in enumerate(shape)]
    t = Tensor.empty("myT", indices)
    assert t.name == "myT"
    assert t.indices == tuple(indices)
    assert t.shape == shape
    assert t.data.shape == shape
    assert t.data.dtype == NP_ARRAY_TYPE
    print("Success!")


