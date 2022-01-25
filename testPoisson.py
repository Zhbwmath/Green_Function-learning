import torch
import math
import Green_Function
import scipy.io
import matplotlib.pyplot as plt

def convolution(model,x,node,elem,area,xi):
    """
    discrete covolution, i.e., gaussian quadrature in elements and edges.
    in Poisson problem, Green's function has the form G(x,s) = G(x-s)
    """
    Gauss_int_std = torch.tensor([[1/3,1/3],[2/15,11/15],[2/15,2/15],[11/15,2/15]],dtype=float)
    Gauss_w_std = torch.tensor([-27/48,25/48,25/48,25/48])
    Gauss_bd_std = torch.tensor([[0,0],[-torch.sqrt(torch.tensor(1/3)),0],[torch.sqrt(torch.tensor(1/3)),0]],dtype=float)
    Gauss_w_bd = torch.tensor([8/9,5/9,5/9],dtype=float)
    elem = elem-1
    elem = elem.long()
    x1 = torch.index_select(node,0,elem[:,0])[:,0]
    x2 = torch.index_select(node,0,elem[:,1])[:,0]
    x3 = torch.index_select(node,0,elem[:,2])[:,0]
    y1 = torch.index_select(node,0,elem[:,0])[:,1]
    y2 = torch.index_select(node,0,elem[:,1])[:,1]
    y3 = torch.index_select(node,0,elem[:,2])[:,1]
    # affine transformation
    quadpoints_x1 = (1-Gauss_int_std[0,0]-Gauss_int_std[0,1])*x1 + Gauss_int_std[0,0]*x2 + Gauss_int_std[0,1]*x3
    quadpoints_x2 = (1-Gauss_int_std[1,0]-Gauss_int_std[1,1])*x1 + Gauss_int_std[1,0]*x2 + Gauss_int_std[1,1]*x3
    quadpoints_x3 = (1-Gauss_int_std[2,0]-Gauss_int_std[2,1])*x1 + Gauss_int_std[2,0]*x2 + Gauss_int_std[2,1]*x3
    quadpoints_x4 = (1-Gauss_int_std[3,0]-Gauss_int_std[3,1])*x1 + Gauss_int_std[3,0]*x2 + Gauss_int_std[3,1]*x3
    quadpoints_y1 = (1-Gauss_int_std[0,0]-Gauss_int_std[0,1])*y1 + Gauss_int_std[0,0]*y2 + Gauss_int_std[0,1]*y3
    quadpoints_y2 = (1-Gauss_int_std[1,0]-Gauss_int_std[1,1])*y1 + Gauss_int_std[1,0]*y2 + Gauss_int_std[1,1]*y3
    quadpoints_y3 = (1-Gauss_int_std[2,0]-Gauss_int_std[2,1])*y1 + Gauss_int_std[2,0]*y2 + Gauss_int_std[2,1]*y3
    quadpoints_y4 = (1-Gauss_int_std[3,0]-Gauss_int_std[3,1])*y1 + Gauss_int_std[3,0]*y2 + Gauss_int_std[3,1]*y3
    quadpoints_x1 = quadpoints_x1[:,None]
    quadpoints_x2 = quadpoints_x2[:,None]
    quadpoints_x3 = quadpoints_x3[:,None]
    quadpoints_x4 = quadpoints_x4[:,None]
    quadpoints_y1 = quadpoints_y1[:,None]
    quadpoints_y2 = quadpoints_y2[:,None]
    quadpoints_y3 = quadpoints_y3[:,None]
    quadpoints_y4 = quadpoints_y4[:,None]
    # points for quadrature (order-3)
    quadpoint1 = torch.cat((quadpoints_x1,quadpoints_y1),1)
    quadpoint2 = torch.cat((quadpoints_x2,quadpoints_y2),1)
    quadpoint3 = torch.cat((quadpoints_x3,quadpoints_y3),1)
    quadpoint4 = torch.cat((quadpoints_x4,quadpoints_y4),1)
    # input for NN model
    Xi = xi.transpose(0,1).repeat(quadpoint1.shape[0],1)
    input1 = torch.cat((x-quadpoint1,Xi,x-quadpoint1-Xi),1)
    input2 = torch.cat((x-quadpoint2,Xi,x-quadpoint2-Xi),1)
    input3 = torch.cat((x-quadpoint3,Xi,x-quadpoint3-Xi),1)
    input4 = torch.cat((x-quadpoint4,Xi,x-quadpoint4-Xi),1)
    # return evaluated u(x), where x scalar
    quadint = torch.sum(area*(Gauss_w_std[0]*ffun(quadpoint1)*model(input1)+ \
    Gauss_w_std[1]*ffun(quadpoint2)*model(input2)+Gauss_w_std[2]*ffun(quadpoint3)*model(input3)+Gauss_w_std[3]*ffun(quadpoint4)*model(input4)))
    return quadint

def ffun(data):
    """
    r.h.s. of the equation, i.e. force term
    """
    fterm=8*(math.pi)**2*(torch.sin(2*data[:,0])*torch.sin(2*data[:,1]))
    return fterm

def errfun(data,predict):
    """
    L2 error of numerical solution and analytic real solution
    """
    u_ex = (torch.sin(2*data[:,0])*torch.sin(2*data[:,1]))
    err = torch.norm(u_ex-predict)
    return err

def drawgraph(model,device,opt,xi,predict):
    # plot the solution
        with torch.no_grad():
            x1 = torch.linspace(-1, 1, 201)
            x2 = torch.linspace(-1, 1, 201)
            X, Y = torch.meshgrid(x1, x2)
            Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
            Z = Z.to(device)
            if opt =='Green':
                Xi = xi.transpose(0,1).repeat(Z.shape[0],1)
                pred = model(torch.cat((Z,Xi,Z-Xi),1)).to(torch.device('cpu'))
                Z = Z.to(torch.device('cpu')) 
            else:
                # pred = torch.zeros(Z.shape[0],1).to(device)
                # for i in range(predict.shape[0]):
                #     pred[i] = convolution(model,data[i,:],node,elem,area,xi)
                # pred = pred.to(torch.device('cpu'))
                Z = Z.to(torch.device('cpu'))
                pred = predict.to(torch.device('cpu'))
                # pred = pred[:,None] 

        pred = pred.cpu().numpy()
        pred = pred.reshape(201, 201)

        plt.figure(0)
        ax = plt.subplot(1, 1, 1)
        h1 = plt.imshow(pred, interpolation='nearest', cmap='rainbow',
                    extent=[0, 1, 0, 1],
                    origin='lower', aspect='auto')
        
        plt.colorbar(h1)
        plt.show()

if __name__ == '__main__':
    device = torch.device("cuda")
    offlinedevice = torch.device("cpu")
    params = dict()
    params["d"] = 2 # 2D
    params["dd"] = 1 # Scalar field
    params["bodyBatch1"] = 512 # Batch size
    params["bodyBatch2"] = 128
    params["bodyBatch3"] = 128
    params["bdryBatch"] = 1024 # Batch size for the boundary integral
    params["lr"] = 0.0001 # Learning rate
    params["width"] = 16 # Width of layers
    params["depth"] = 5 # Depth of the network: depth+2
    params["trainStep"] = 100000
    params["step_size"] = 5000
    params["decay"] = 0.0001
    params["gamma"] = 0.5
    params["weightbd"] = 400
    params["weightsym"] = 1
    params["weightint"] = 1e-10
    params["c1"] = 5
    params["c2"] = 10
    params["s"] = 1e-1
    h = 1e-2
    xi = torch.tensor([[0,],[0.]]).to(device)
    xi.requires_grad = True

    model = Green_Function.PINN(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=params["step_size"],gamma=params["gamma"])
    try:
        model.load_state_dict(torch.load("last_model.pt"))
    except:
        Green_Function.train(model,device,params,optimizer,scheduler,xi)
    model.to(device)
    
    mat = scipy.io.loadmat('D:\Program Files\MATLAB\Matlab Files\Twolevel additive precondioner for eigenvalue problems\h_elem3.mat')
    node = mat['node']
    elem = mat['elem']
    area = mat['area']
    node = torch.from_numpy(node).float().to(device)
    area = torch.from_numpy(area).float().to(device)
    elem = torch.from_numpy(elem).to(device)
    Xi_int = xi.transpose(0,1).repeat(node.shape[0],1)
    Gauss_predict = model(torch.cat((node,Xi_int,node-Xi_int),dim=1))
    x = torch.linspace(-1,1,int(2/h)+1)
    datax,datay = torch.meshgrid(x,x)
    data = torch.cat((datax.reshape(-1,1),datay.reshape(-1,1)),dim=1).to(device)
    data.requires_grad = False
    predict = torch.zeros(data.shape[0],1).to(device)
    with torch.no_grad():
        for i in range(predict.shape[0]):
            predict[i] = convolution(model,data[i,:],node,elem,area,xi)
    data = data.to(offlinedevice)
    predict = predict.to(offlinedevice)
    drawgraph(model,device,'solution',xi,predict)
    print(errfun(data,predict))