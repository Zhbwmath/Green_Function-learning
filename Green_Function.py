import numpy as np
import torch
import generateData2
import matplotlib.pyplot as plt
import math

class PINN(torch.nn.Module):
    # modified PINN, the input layer is speically designed
    def __init__(self, params):
        super(PINN, self).__init__()
        self.params = params
        self.linearIn = torch.nn.Linear(3*self.params["d"], self.params["width"])
        self.linear = torch.nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(torch.nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = torch.nn.Linear(self.params["width"], self.params["dd"])
    
    def forward(self, x):
        x = torch.tanh(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x_temp = torch.tanh(layer(x))
            x = x_temp
        
        return self.linearOut(x)

def lossfun(model,dataint,databdry,xi,params):
    """ There are 3 parts of loss function: 
    MSE of differnential operator minus (approx) Dirac Delta, 
    MSE of inconsitency with boundary condtion, and
    MSE of assmetry between x and xi.
    """
    cntint = dataint.shape[0]
    cntbdry = databdry.shape[0]
    Xi_int = xi.transpose(0,1).repeat(dataint.shape[0],1)
    Xi_bdry = xi.transpose(0,1).repeat(databdry.shape[0],1)
    Gint = model(torch.cat((dataint,Xi_int,dataint-Xi_int),dim=1))
    Gbdry = model(torch.cat((databdry,Xi_bdry,databdry-Xi_bdry),dim=1))
    Gsym = model(torch.cat((Xi_int,dataint,Xi_int-dataint),dim=1))
    d2fdx2,d2fdy2 = laplacian(dataint,Gint)
    LGint = - d2fdx2 - d2fdy2
    LGint = torch.squeeze(LGint)
    s = params["s"]
    d = params["d"]
    rho = torch.exp(torch.pow(torch.linalg.norm(dataint-Xi_int,ord=2,dim=1),2) / (2*s**2)) / torch.pow(torch.sqrt(torch.tensor(2*math.pi))*s,d)
    is_inf = torch.logical_not(torch.isinf(rho))
    dataint = dataint[is_inf,:]
    rho = rho[is_inf]
    LGint = LGint[is_inf]
    lossint = torch.sum(torch.abs(rho-LGint)) / cntint
    lossbdry = torch.sum(Gbdry**2) / cntbdry
    losssym = torch.sum((Gint-Gsym)**2) / cntint
    weightbd = params["weightbd"]
    weightsym = params["weightsym"]
    weightint = params["weightint"]
    return weightint * lossint + weightbd * lossbdry + weightsym * losssym

def errfun_green(model,dataint,xi):
    """
    return the L2 error of fundamental solution minus the numerical approximation
    """
    Green = torch.log(torch.norm(dataint,dim=1))
    Xi_int = xi.transpose(0,1).repeat(dataint.shape[0],1)
    predict = model(torch.cat((dataint,Xi_int,dataint-Xi_int),dim=1))
    d2fdx2,d2fdy2 = laplacian(dataint,Green)
    pd2fdx2,pd2fdy2 = laplacian(dataint,predict)
    err = torch.sum((d2fdx2-pd2fdx2)**2 + (d2fdy2-pd2fdy2)**2)
    return err

def train(model,device,params,optimizer,scheduler,xi):
    # training process
    dataint = generateData2.sampleFromCdomain(params,xi,device).detach().to(device).transpose(0,1)
    dataint.requires_grad = True
    databdry = torch.from_numpy(generateData2.sampleFrombond(params["bdryBatch"])).float().to(device)
    databdry.requires_grad = True
    for step in range(params["trainStep"]):
        model.zero_grad()
        loss = lossfun(model,dataint,databdry,xi,params)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if 10*(step+1)%params["trainStep"] == 0:
            trainerror = errfun_green(model,dataint,xi)
            print("Training Error at Step %s is %s."%(step+1,trainerror))
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))   
    torch.save(model.state_dict(), "last_model.pt")

def laplacian(dataint,output):
    """
    Laplacian operator, smooth form
    """
    df = torch.autograd.grad(outputs=output,inputs=dataint,grad_outputs=torch.ones_like(output),create_graph=True,retain_graph=True,only_inputs=True)[0]
    dfdx = df[:,0:1]
    dfdy = df[:,1:2]
    d2fdx2 = torch.autograd.grad(outputs=dfdx,inputs=dataint,grad_outputs=torch.ones_like(dfdx),create_graph=True,retain_graph=True,only_inputs=True)[0]
    d2fdy2 = torch.autograd.grad(outputs=dfdy,inputs=dataint,grad_outputs=torch.ones_like(dfdy),create_graph=True,retain_graph=True,only_inputs=True)[0]
    d2fdx2 = d2fdx2[:,0:1]
    d2fdy2 = d2fdy2[:,1:2]
    return d2fdx2,d2fdy2

