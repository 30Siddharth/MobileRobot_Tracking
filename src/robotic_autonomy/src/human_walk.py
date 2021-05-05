import numpy as np 
from matplotlib import pyplot as plt
import pdb

def getPose(nk,dt,X,U):
    Xpose = []
    Ypose = []
    Thpose = []
    for k in range(1,nk):
        Xpose.append(X[0])
        Ypose.append(X[1])
        Thpose.append(X[2])
        U[0] = U[0] + 0.02*np.random.rand(1,1)
        U[1] = U[1] + 0.02*np.random.rand(1,1)
        # U[0] = U[0] + 0.02*np.random.normal(0,1,1)
        # U[1] = U[1] + 0.02*np.random.normal(0,1,1)
        xkm = X[0]
        ykm = X[1]
        thkm = X[2]
        lk = U[0]
        rk = U[1]
        dX = differential_walk(xkm,ykm,thkm,lk,rk)
        X = X + dt*dX
        

    return Xpose, Ypose, Thpose


def differential_walk(xkm,ykm,llv,rlv):
    
    R = 0.30 # The distance between the two feet. This acts as a radius of turn.
    thk = np.arctan2(ykm,xkm)
    dxk = (llv + rlv)*0.5*np.cos(thk)
    dyk = (llv + rlv)*0.5*np.sin(thk)
    dthk = (llv - rlv)/R

    return dxk,dyk



if __name__ == '__main__':

    nk = 10
    dt = 1
    X = np.array([1,1,0])
    U = np.array([0.2,0.2])

    x,y,th = getPose(nk,dt,X,U)

    # p = plt.figure(1)
    # # plt.plot(x[0],y[0],'X')
    # plt.plot(x,y,'--o')
    # plt.plot(x[0],y[0],'X')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend(['Path taken by Human','Starting Point'])
    # plt.axis('equal')

    # plt.show()

    


