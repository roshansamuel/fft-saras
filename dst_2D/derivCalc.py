import numpy as np
from globalData import Nx, Nz, xi_x, zt_z, i2hx, i2hz

# Temporary array for storing derivatives
tmp = np.zeros([Nx + 2, Nz + 2])

xSt, xEn = 1, Nx + 1
x0 = slice(xSt, xEn)
xm1 = slice(xSt-1, xEn-1)
xp1 = slice(xSt+1, xEn+1)

zSt, zEn = 1, Nz + 1
z0 = slice(zSt, zEn)
zm1 = slice(zSt-1, zEn-1)
zp1 = slice(zSt+1, zEn+1)

# Central difference with 2nd order accuracy
def dfx(F):
    tmp[x0, :] = (F[xp1, :] - F[xm1, :]) * xi_x[x0]
    tmp[ 0, :] = (-3.0*F[ 0, :] + 4*F[ 1, :] - F[ 2, :]) * xi_x[ 0]
    tmp[-1, :] = ( 3.0*F[-1, :] - 4*F[-2, :] + F[-3, :]) * xi_x[-1]

    return tmp * i2hx

def dfz(F):
    tmp[:, z0] = (F[:, zp1] - F[:, zm1]) * zt_z[z0]
    tmp[:,  0] = (-3.0*F[:,  0] + 4*F[:,  1] - F[:,  2]) * zt_z[ 0]
    tmp[:, -1] = ( 3.0*F[:, -1] - 4*F[:, -2] + F[:, -3]) * zt_z[-1]

    return tmp * i2hz

