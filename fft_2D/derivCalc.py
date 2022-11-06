import numpy as np
import globalData as glob

# Temporary array for storing derivatives
tmp = np.zeros([glob.Nx + 2, glob.Nz + 2])

xSt, xEn = 1, glob.Nx + 1
x0 = slice(xSt, xEn)
xm1 = slice(xSt-1, xEn-1)
xp1 = slice(xSt+1, xEn+1)

zSt, zEn = 1, glob.Nz + 1
z0 = slice(zSt, zEn)
zm1 = slice(zSt-1, zEn-1)
zp1 = slice(zSt+1, zEn+1)

# Central difference with 2nd order accuracy
def dfx(F):
    tmp[x0, z0] = (F[xp1, z0] - F[xm1, z0]) * glob.xi_x[x0] * glob.i2hx

    return tmp

def dfz(F):
    tmp[x0, z0] = (F[x0, zp1] - F[x0, zm1]) * glob.zt_z[z0] * glob.i2hz

    return tmp

