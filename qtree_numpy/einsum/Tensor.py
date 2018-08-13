import numpy as np
import logging
from .Variable import Variable

log = logging.getLogger('qtree')

global tensor_count
tensor_count = 0
class Tensor():
    def __init__(self,tensor=None,
                 variables=None,name=None):
        if isinstance(tensor,list):
            tensor = np.array(tensor)
        global tensor_count
        tensor_count+=1
        self.name = name or 'T'+str(tensor_count)
        self._tensor = tensor
        self.variables = []
        try:
            self._order = len(tensor.shape)
            #vs = variables or [Variable() for i in tensor.shape]
            vs = variables or []
        except:
            self._order = 0
            vs = variables or []
        if len(vs)>0:
            self.add_variables(*vs)
    def __del__(self):
        for v in self.variables:
            v.decrement()

    def add_variables(self,*vs):
        log.debug('adding vars'+str(vs))
        self.variables += vs
        [v.increment() for v in vs]
        self.rank = len(self.variables)

    def sum_by(self,v):
        if isinstance(v,Variable):
            axis = self.variables.index(v)
        else:
            raise Exception("expexted Variable got",v)
        self._tensor = np.sum(self._tensor,axis=axis)
        self.variables.remove(v)
        return self

    def merge_with(self,tensors):
        if len(tensors)>0:
            res = self.multiply(tensors[0])
            for t in tensors[1:]:
                # TODO: you can optimize here with diagonal
                res = res.multiply(t)
            return res
        else: return self

    def reorder_by_id(self,id_list):
        ordering = []
        new_vars = []
        for i in id_list:
            for v in self.variables:
                if i ==v._id:
                    ordering.append(
                        self.variables.index(v)
                    )
                    new_vars.append(v)
        log.info('before reordering %s',self)
        log.info('ordering indices %s',ordering)
        self.variables = new_vars
        try:
            self._tensor =np.transpose(
                self._tensor,
                ordering
            )
        except ValueError:
            pass
        return self
    def multiply(self,tensor):
        t = Tensor()
        t.add_variables(
            *(self.variables+tensor.variables))

        t._tensor = np.tensordot(self._tensor,
                                 tensor._tensor,
                                axes=0)
        t.diagonalize_if_dupl()
        t._order = len(t._tensor.shape)
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
                log.debug("  duplicate of %s @ %s",v,dup)
                self._tensor = np.diagonal(
                    self._tensor,
                    axis1=dup[0],
                    axis2=dup[1]
                )
                l = [x for i,x in enumerate(l) if x!=v]
                l += [v]
            i+=1
        self.variables = l

    def __str__(self):
        return self.name+'_'+','.join([str(v._id) for v in self.variables])
    def __repr__(self):
        if sum(self._tensor.shape)<10:
            return "<tensor \n "+str(self._tensor.round(4))+ '\nvars: '+str(self.variables)+">"
        else:
            s = self._tensor.shape
            r = len(s)
            return f"<tensor with shape {s} and rank {r}\nvars: "+str(self.variables)+f"({len(self.variables)})>"
