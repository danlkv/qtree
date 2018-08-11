import numpy as np
import logging
from .Variable import Variable
log = logging.getLogger('qtree')

class Tensor():
    def __init__(self,tensor=None):
        if not isinstance(tensor,np.array):
            raise Exception("expected numpy array got ",tensor)
        self._tensor = tensor
        self.variables = []

    def add_variable(self,*vs):
        log.debug('adding vars'+str(vs))
        self.variables += vs
    def _get_var_ordering(self,var):
        idx = self.variables.index(var)
        vs = self.variables
        # make the var first
        r = list(range(len(vs)))
        r.remove(idx)
        ordering = [idx] + r
        log.debug('ordering'+str(ordering))
        return ordering

    def get_ordered_tensors(self,var):
        log.debug(self._tensor)
        #self._align_to_vars()
        return np.transpose(
            self._tensor,
            self._get_var_ordering(var)
        )
    def multiply(self,tensor,var):
        print(self,tensor)
        t = Tensor()

        xs = self.variables+tensor.variables
        indexes = [i for i,x in enumerate(xs) if x ==var]
        # make target variable last index
        orders =[ np.arange(-1,len(self.variables)-1) + indexes[0],
                 np.arange(-1,len(tensor.variables)-1) + indexes[1]
                ]
        print("indexes of m",indexes,"orders:",orders)
        a = np.transpose(self._tensor,orders[0])
        b = np.transpose(tensor._tensor,orders[1])
        _t = np.einsum('...ji,...ki->...jki')

        t._tensor = _t
        new_variables = [x for i,x in enumerate(xs)
                         if x !=var]
        t.variables=new_variables+[var]
        t.diagonalize_if_dupl()
        #print(t)
        return t
    def diagonalize_if_dupl(self):
        def duplicates(lst,item):
            return [i for i,x in enumerate(lst) if x==item]
        l = self.variables
        i = 0
        while True:
            try:
                v = l[i]
            except IndexError:
                break
            dup = duplicates(l,v)
            if len(dup)>1:
                print("__duplicate of",v,'@',dup)
                print("__vars",l)
                self._tensor = np.diagonal(
                    self._tensor,
                    axis1=dup[0],
                    axis2=dup[1]
                )
                l = [x for i,x in enumerate(l) if x!=v]
                l += [v]
            i+=1
        self.variables = l


    def sum(self,over,axis=-1):
        print('summing',self)
        t = Tensor()
        v = over
        index = self.variables.index(v)
        if len(self.variables)>1:
            print('___',v,'@',index)
            t.variables = self.variables
            t.variables.remove(v)
        else:
            t.variables = []
            print("____ scalar!",self)

        t._tensor = np.sum(self._tensor,axis=index)
        return t
    def __repr__(self):
        if sum(self._tensor.shape)<10:
            return "<tensor \n "+str(self._tensor.round(4))+ '\nvars: '+str(self.variables)+">"
        else:
            s = self._tensor.shape
            r = len(s)
            return f"<tensor with shape {s} and rank {r}\nvars: "+str(self.variables)+f"({len(self.variables)})>"
