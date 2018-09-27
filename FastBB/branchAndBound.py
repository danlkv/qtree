from mpi4py import MPI

DEPTH =0
def getBB(ub_fun,lb_fun,state_iterator):
    def BB(state,ub,lb):
        global DEPTH
        ii=0
        #print(state[0].nodes(),ub,lb,DEPTH)
        for s in state_iterator(state):
            #if DEPTH == 30:
                #print("depth is 38 iter is",ii,"bounds are",ub,lb)
            ii+=1
            lb = lb_fun(s,ub,lb)
            ub = ub_fun(s,ub,lb)
            if ub<=lb:
                #print('cont',s[1])
                continue
            else:
                DEPTH+=1
                print(ub,lb,DEPTH,ii, '\t',s[1])
                ub,lb = BB(s,ub,lb)
                DEPTH-=1
        return ub,lb
    return BB

def performBB(
        bb_fun,
        state_iterator,
        inital_state):
    print("inital ordering",inital_state[1])
    #ub_init = minwidth(inital_state[0].copy())[0]
    ub,lb= bb_fun(inital_state,20,0)
    print("ub,lb",ub,lb)

