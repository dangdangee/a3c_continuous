import numpy as np
import scipy as sc

def field(a1,a2,b1,b2,J,r,z):
    pi = sc.pi
    N = 8
    x = [-0.960289856497536, -0.796666477413627, -0.525532409916329, -0.183434642495650, 0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536]
    w = [0.101228536290376, 0.222381034453374, 0.313706645877887, 0.362683783378362, 0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376]
    x = np.array(x)
    w = np.array(w)

    if (r>a1) and (r<a2):
        case = 'in'
    else:
        case = 'out'

    if case == 'in':
        Br = np.array(0)
        Bz = np.array(0)
        xJ = np.zeros((2,1))
        xb = np.zeros((2,1))
        xJ[0] = J
        xJ[1] = -J
        xb[0] = b1
        xb[1] = b2
        for iSUB in range(0,2):
            if iSUB == 0:
                a22 = r
                a11 = a1
            else:
                a22 = a2
                a11 = r
            for i in range(0,N):
                for k in range(0,2):
                    h = xb[k]
                    a = 0.5*(a11+a22)+0.5*(a22-a11)*x[i]
                    xk = 4*a*r/((r+a)**2+(z-h)**2)
                    aa0 = 1
                    bb0 = np.sqrt(1-xk)
                    s0 = xk
                    sG = 0.5*(1-bb0)**2
                    cSQ = 4*a*r/(a+r)**2
                    cPSQ = ((a-r)**2)/((a+r)**2)
                    zeta0 = 0
                    eps0 = cSQ/cPSQ
                    delta0 = cPSQ/bb0
                    ii = 0
                    ERref = 1e-10
                    ResER = 1
                    while (ResER>=ERref) or (abs(1-delta0)>=ERref):
                        aa1 = (aa0+bb0)/2
                        bb1 = np.sqrt(aa0*bb0)
                        c1 = (aa0-bb0)/2
                        delta1 = bb1*(2+delta0+1/delta0)/4/aa1
                        eps1 = (delta0*eps0+zeta0)/(1+delta0)
                        zeta1 = (eps0+zeta0)/2
                        sG = sG+(2**ii)*((aa1-bb1)**2)
                        ii = ii+1
                        s0 = s0+(2**ii)*(c1**2)
                        aa0 = aa1
                        bb0 = bb1
                        delta0 = delta1
                        eps0 = eps1
                        zeta0 = zeta1
                        ResER = (2**ii)*(c1**2)
                    sINF = sG
                    alphaINF = aa0
                    zetaINF = zeta0
                    temp = pi*1e-7*np.sqrt((a+r)**2+(z-h)**2)*sINF
                    
                    Br = Br-0.5*(a22-a11)*w[i]*temp*xJ[k]/2/r/alphaINF

                    temp1 = pi*1e-7*(z-h)
                    temp2 = 2*a+(a-r)*zetaINF
                    temp3 = alphaINF*(r+a)*np.sqrt((a+r)**2+(z-h)**2)

                    Bz = Bz+0.5*(a22-a11)*w[i]*xJ[k]*temp1*temp2/temp3

    elif case == 'out':
        Br = np.array(0)
        Bz = np.array(0)
        xJ = np.zeros((2,1))
        xb = np.zeros((2,1))
        xJ[0] = J
        xJ[1] = -J
        xb[0] = b1
        xb[1] = b2
        for i in range(0,N):
            for k in range(0,2):
                h = xb[k]
                a = 0.5*(a1+a2)+0.5*(a2-a1)*x[i]
                xk = 4*a*r/((r+a)**2+(z-h)**2)
                aa0 = 1
                bb0 = np.sqrt(1-xk)
                s0 = xk
                sG = 0.5*(1-bb0)**2
                cSQ = 4*a*r/(a+r)**2
                cPSQ = ((a-r)**2)/((a+r)**2)
                zeta0 = 0
                eps0 = cSQ/cPSQ
                delta0 = cPSQ/bb0
                ii = 0
                ERref = 1e-10
                ResER = 1

                while (ResER>=ERref) or (abs(1-delta0)>=ERref):
                    aa1 = (aa0+bb0)/2
                    bb1 = np.sqrt(aa0*bb0)
                    c1 = (aa0-bb0)/2
                    delta1 = bb1*(2+delta0+1/delta0)/4/aa1
                    eps1 = (delta0*eps0+zeta0)/(1+delta0)
                    zeta1 = (eps0+zeta0)/2
                    sG = sG+(2**ii)*((aa1-bb1)**2)
                    ii = ii+1
                    s0 = s0+(2**ii)*(c1**2)
                    aa0 = aa1
                    bb0 = bb1
                    delta0 = delta1
                    eps0 = eps1
                    zeta0 = zeta1
                    ResER = (2**ii)*(c1**2)

                sINF = sG
                alphaINF = aa0
                zetaINF = zeta0

                temp = pi*1e-7*np.sqrt((a+r)**2+(z-h)**2)*sINF
                
                Br = Br-0.5*(a2-a1)*w[i]*temp*xJ[k]/2/r/alphaINF

                temp1 = pi*1e-7*(z-h)
                temp2 = 2*a+(a-r)*zetaINF
                temp3 = alphaINF*(r+a)*np.sqrt((a+r)**2+(z-h)**2)

                Bz = Bz+0.5*(a2-a1)*w[i]*xJ[k]*temp1*temp2/temp3

    return Br, Bz