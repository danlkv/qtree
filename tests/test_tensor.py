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

    t = Tensor.empty("myT", indices, dtype=np.float32)
    assert t.data.dtype == np.float32
    assert t.dtype == np.float32
    print("Success!")

def test_slice_tensor():
    shape = (2, 3, 4)
    indices = [Var(i, size=s) for i, s in enumerate(shape)]
    t = Tensor.empty("myT", indices, dtype=np.uint32)
    S = t[{indices[0]: 1, indices[1]: slice(0, 1)}]
    assert S.data.shape == (1, 4)
    assert indices[2] in S.indices
    assert S.indices[1].size == 4
    assert np.allclose(t.data[1, 0:1], S.data)

    S = t[1]
    assert indices[0] not in S.indices
    assert indices[1] in S.indices
    assert indices[2] in S.indices
    assert np.allclose(t.data[1], S.data)

    S = t[1, 2]
    assert indices[1] not in S.indices
    assert indices[2] in S.indices
    assert np.allclose(t.data[1, 2], S.data)
    S.data[0] = 100
    assert t.data[1, 2, 0] == 100
    print("Success!")


