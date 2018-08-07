from Expression import Expression
from Variable import Variable
from Tensor import Tensor
import numpy as np
import time

np.random.seed(42)

t1 = np.random.randn(2,5)[:,:3]
t2 = np.zeros(8)
t2[2]=1
t3 = np.random.randn(3,5,8)
ey = np.eye(5)

vs = [Variable() for i in range(6)]
free_vars = [vs[0],vs[2]]

e = Expression()
tns = [
    Tensor(t1,variables=[vs[0],vs[1]]),
    Tensor(t2,variables=[vs[3]]),
    Tensor(ey,variables=[vs[1],vs[4]]),
    Tensor(t3,variables=[vs[4],vs[2],vs[3]]),
    ]
print(tns)

e.set_tensors(tns)
tensors = e.evaluate(free_vars)
print(tensors[0]._tensor)
##print(t1.dot(t3))
print()
print()

e = Expression()
t1 = np.random.randn(5,40)
t2 = np.random.randn(40)
t3 = np.random.randn(3,40,8)
e.set_tensors([
    Tensor(t1,variables=[vs[1],vs[2]]),
    Tensor(t2,variables=[vs[2]]),
    Tensor(t3,variables=[vs[5],vs[2],vs[3]]),
])
e.set_order([3,2])
start_time = time.time()
tensors = e.evaluate([vs[1],vs[5],vs[3]])
print("--- %s seconds ---" % (time.time() - start_time))

print(tensors[0]._tensor)
start_time = time.time()
print(np.einsum('ij,j,mjk->imk',t1,t2,t3))
print("--- %s seconds ---" % (time.time() - start_time))

