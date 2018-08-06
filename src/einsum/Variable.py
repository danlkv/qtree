import numpy as np
import logging
log = logging.getLogger('qtree')

global var_count
var_count=0
class Variable():
    def __init__(self,_id=None,space=[0,1]):
        global var_count
        if not _id:
            self._id = var_count
        else:
            self._id = _id
        self._space = space
        self.idx = self._id
        self._ref = 0
        var_count+=1
    def merge_with(self, var):
        raise NotImplementedError()
        log.debug("merging variables",self,var)
        space_mod1,space_mod2 =(
            max(self._space) - min(self._space) + 1,
            max(var._space) - min(var._space) + 1
        )
        new_mod = space_mod1*space_mod2
        return Variable()
    def increment(self):
        self._ref+=1
    def decrement(self):
        self._ref-=1
    def __repr__(s):
        return "_v%s"%(s._id)
        return "_v%s @%s<-%s"%(s._id,s.idx,s._ref)
