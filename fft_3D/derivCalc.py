import numpy as np
import globalData as glob

# Temporary array for storing derivatives
tmp = np.zeros([glob.Nx + 2, glob.Ny + 2, glob.Nz + 2])

xSt, xEn = 1, glob.Nx + 1
x0 = slice(xSt, xEn)
xm1 = slice(xSt-1, xEn-1)
xp1 = slice(xSt+1, xEn+1)

ySt, yEn = 1, glob.Ny + 1
y0 = slice(ySt, yEn)
ym1 = slice(ySt-1, yEn-1)
yp1 = slice(ySt+1, yEn+1)

zSt, zEn = 1, glob.Nz + 1
z0 = slice(zSt, zEn)
zm1 = slice(zSt-1, zEn-1)
zp1 = slice(zSt+1, zEn+1)

# Central difference with 2nd order accuracy
def dfx(F):
    tmp[x0, y0, z0] = (F[xp1, y0, z0] - F[xm1, y0, z0]) * glob.xi_x[x0] * glob.i2hx

    return tmp

def dfy(F):
    tmp[x0, y0, z0] = (F[x0, yp1, z0] - F[x0, ym1, z0]) * glob.et_y[y0] * glob.i2hy

    return tmp

def dfz(F):
    tmp[x0, y0, z0] = (F[x0, y0, zp1] - F[x0, y0, zm1]) * glob.zt_z[z0] * glob.i2hz

    return tmp

