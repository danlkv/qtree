#!/usr/bin/python3.6
from mpi4py import MPI
comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()
if rank==0:
    i = input("zero:Enter a message\n")
    print("sending %s to one"%i)
    req = comm.send(i ,dest=1,tag=42)
elif rank ==1:
    print("im one, now receving")

    data  = comm.recv(source=0,tag=42)
    print("recived",data)
else:
    print("im other")

comm.Disconnect()

