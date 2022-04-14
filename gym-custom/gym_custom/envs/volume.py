import numpy as np
import scipy as sc

def volume(a1,a2,b1,b2):
    pi = sc.pi
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)

    V = (a2**2-a1**2)*(b2-b1)*pi

    return V