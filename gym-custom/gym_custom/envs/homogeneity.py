import numpy as np
from numpy import matlib
import scipy as sc
from scipy import special

def gaussQuad(a,b):
    f = open("GQ100.txt",'r')
    x = f.readline()
    w = f.readline()
    f.close()
    xs = x.split('  ')
    ws = w.split('  ')

    xn = np.zeros((100,1))
    wn = np.zeros((100,1))

    for i in range(0,50):
        xn[i] = np.float_(xs[i])*np.float_(1e-15)
        wn[i] = np.float_(ws[i])*np.float_(1e-15)
        xn[99-i] = 1-xn[i]
        wn[99-i] = wn[i]

    a = np.array(a)
    b = np.array(b)
    x2 = xn*(b-a)+a
    w2 = wn*(b-a)
    return x2, w2

def legendreDiff(n,x):
    x = np.array(x, dtype=np.float64)
    (p,q) = np.shape(x)
    N = n*np.ones((p,q))
    mat1 = np.ones((p,q))
    mat2 = x

    if n==0:
        d = np.zeros((p,q))
    elif n==1:
        d = np.ones((p,q))
    else:
        for i in range(2,n+1):
            matTemp = (2*i-1)/i*x*mat2-(i-1)/i*mat1
            mat1 = mat2
            mat2 = matTemp

    d = N/(x**2-1)*(x*mat2-mat1)
    return d

def loopHC(r,h,I,max):
    pi = sc.pi
    r = np.array(r)
    h = np.array(h)
    mu = np.array(4e-7*pi)
    r0 = np.sqrt(r*r+h*h)
    a = np.arccos(h/r0)

    C = mu/2*I*np.sin(a)*np.sin(a)*legendreDiff(max+1,np.cos(a))/r0**(max+1)
    return C

def solenoidHC(a1,a2,b1,b2,J,idx):
    a1 = np.array(a1, dtype=np.float64)
    a2 = np.array(a2, dtype=np.float64)
    b1 = np.array(b1, dtype=np.float64)
    b2 = np.array(b2, dtype=np.float64)
    J = np.array(J)
    I = J*(a2-a1)*(b2-b1)
    x, wx = gaussQuad(a1,a2)
    z, wz = gaussQuad(b1,b2)

    X = np.matlib.repmat(x,1,100)
    Z = np.matlib.repmat(z,1,100)
    Z = np.transpose(Z)
    WX = np.matlib.repmat(wx,1,100)
    WZ = np.matlib.repmat(wz,1,100)
    WZ = np.transpose(WZ)
    Curr = WX*WZ*I

    HC = np.zeros((idx+1,1))
    # print(np.shape(HC))
    for i in range(0,idx+1):
        Temp = loopHC(X,Z,Curr,i)/(a2-a1)/(b2-b1)
        HC[i]=np.sum(Temp)
    
    return HC

def homogeneity(HC,DSV):
    n = np.size(HC)
    N = 50
    pi = sc.pi
    r = DSV/2*np.ones((N,1))
    theta = np.linspace(0,pi/2,num=N)
    phi = pi/4*np.ones((N,1))

    Map = np.ones((N,n**2))
    for i in range(0,n):
        A = np.ones((i+1,N))
        if i > 0:
            for j in range(0,N):
                p,dp = sc.special.lpmn(i,i,np.cos(theta[j]))
                A[:,j:(j+1)] = p[:,i:(i+1)]

        if i==0:
            TE = np.ones((N,1))
            LE = np.transpose(A)
        else:
            TE = np.ones((N,2*i+1))
            for j in range(0,n):
                TE[:,(2*j+1):(2*j+2)] = np.cos((j+1)*phi)
                TE[:,(2*j+2):(2*j+3)] = np.sin((j+1)*phi)
            temp = np.zeros((2*i+2,i+1))
            for j in range(0,i+1):
                temp[2*j,j:j+1]=1
                temp[2*j+1,j:j+1]=1
            mat = temp[1:(2*i+2),:]
            LE = np.transpose(mat@A)
        
        Map[:,(i**2):((i+1)**2)] = (r**i)*LE*TE
    
    HC2 = np.zeros((n**2,1))
    for i in range(0,n):
        HC2[i**2] = HC[i]

    Bz = Map@HC2
    Bmax = np.max(Bz)
    Bmin = np.min(Bz)
    h = (Bmax-Bmin)/(Bmax+Bmin)*1e6

    return h

