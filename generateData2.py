import numpy as np
import math
import torch

def sampleFromCdomain(params,xi,device=torch.device("cpu")):
    """

    """
    # x=2*np.random.rand(n)-1
    # y=2*np.random.rand(n)-1
    # samplefromdomain=np.stack((x,y),axis=-1)
    # xo=0.05*np.random.rand(n)-0.025
    # yo=0.05*np.random.rand(n)-0.025
    # samplefromdomain2=np.stack((xo,yo),axis=-1)
    # samplefromorigin=np.concatenate((samplefromdomain,samplefromdomain2),axis=0)
    d=params["d"]
    c1=params["c1"]
    c2=params["c2"]
    N1=params["bodyBatch1"]
    N2=params["bodyBatch2"]
    N3=params["bodyBatch3"]
    s=params["s"]
    sample1 = 2*c1*s*(torch.rand(d,N1).to(device)) + (xi-c1*s)
    sample2 = 2*c2*s*(torch.rand(d,N2).to(device)) + (xi-c2*s)
    tmp1 = torch.abs(sample2[0,:]-xi[0])>c1*s
    tmp2 = torch.abs(sample2[1,:]-xi[1])>c1*s
    indices = torch.logical_and(tmp1,tmp2).nonzero()
    indices = torch.squeeze(indices)
    sample2 = sample2[:,indices]
    sample3 = 2*(torch.rand(d,N3).to(device)) + (xi-1)
    tmp1 = torch.abs(sample3[0,:]-xi[0])>c2*s
    tmp2 = torch.abs(sample3[1,:]-xi[1])>c2*s
    indices = torch.logical_and(tmp1,tmp2).nonzero()
    indices = torch.squeeze(indices)
    sample3 = sample3[:,indices]
    return torch.cat((sample1,sample2,sample3),dim=1)


def sampleFromBoundary(n):
    # For simplicity, consider a square with a hole.
    # Square: [-1,1]*[-1,1]
    # Hole: c = (0.3,0.0), r = 0.3
    c = np.array([0.3,0.0])
    r = 0.3
    length = 4*2+2*math.pi*r
    interval1 = np.array([0.0,2.0/length])
    interval2 = np.array([2.0/length,4.0/length])
    interval3 = np.array([4.0/length,6.0/length])
    interval4 = np.array([6.0/length,8.0/length])
    interval5 = np.array([8.0/length,1.0])

    array = np.zeros([n,2])

    for i in range(n):
        rand0 = np.random.rand()
        rand1 = np.random.rand()

        point1 = np.array([rand1*2.0-1.0,-1.0])
        point2 = np.array([rand1*2.0-1.0,+1.0])
        point3 = np.array([-1.0,rand1*2.0-1.0])
        point4 = np.array([+1.0,rand1*2.0-1.0])
        point5 = np.array([c[0]+r*math.cos(2*math.pi*rand1),c[1]+r*math.sin(2*math.pi*rand1)])

        array[i] = myFun(rand0,interval1)*point1 + myFun(rand0,interval2)*point2 + \
            myFun(rand0,interval3)*point3 + myFun(rand0,interval4)*point4 + \
                myFun(rand0,interval5)*point5
 
    return array

def myFun(x,interval):
    if interval[0] <= x <= interval[1]:
        return 1.0
    else: return 0.0


def sampleFrombond(n):
    #boundary of Omega:[-1,1]*[-1,1]
    x1=2*(np.random.rand(n)-0.5)
    y1=-1*np.ones(n)
    p1=np.stack((x1,y1),axis=-1)
    x2=2*(np.random.rand(n)-0.5)
    y2=np.ones(n)
    p2=np.stack((x2,y2),axis=-1)
    x3=np.ones(n)
    y3=2*(np.random.rand(n)-0.5)
    p3=np.stack((x3,y3),-1)
    p4=np.concatenate((p1,p2),axis=0)
    p5=np.concatenate((p3,p4),axis=0)
    return p5

