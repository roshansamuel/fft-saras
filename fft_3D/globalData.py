import numpy as np

inputDir = "../input/data_3D/"

# Global variables
Lx, Ly, Lz = 1.0, 1.0, 1.0
Nx, Ny, Nz = 64, 64, 64
btX, btY, btZ = 0.0, 0.0, 1.3
U, V, W, T, X, Y, Z = 1, 1, 1, 1, 1, 1, 1

# Should transfer function be computed?
cmpTrn = False
# Read existing FFT vs compute anew
readFile = False
# Shells to skip - for very large files
shSkip = 1

nGrid = np.array([Nx, Ny, Nz])
kFactor = np.array([2.0*np.pi/Lx, 2.0*np.pi/Ly, 2.0*np.pi/Lz])
kInt = min(kFactor)*shSkip

minRad = np.sqrt(np.dot((nGrid//2)*kFactor, (nGrid//2)*kFactor))

# Generate kShells
kShell = np.arange(0, minRad, kInt)

dk = np.diff(kShell)
arrLim = kShell.shape[0]

tVol = Lx*Ly*Lz

dXi = 1.0/(Nx)
dEt = 1.0/(Ny)
dZt = 1.0/(Nz)

i2hx = 1.0/(2.0*dXi)
i2hy = 1.0/(2.0*dEt)
i2hz = 1.0/(2.0*dZt)

def genGrid(N, L, bt):
    vPts = np.linspace(0.0, 1.0, N+1)
    xi = np.pad((vPts[1:] + vPts[:-1])/2.0, (1, 1), 'constant', constant_values=(0.0, 1.0))
    if bt:
        xPts = np.array([L*(1.0 - np.tanh(bt*(1.0 - 2.0*i))/np.tanh(bt))/2.0 for i in xi])
        xi_x = np.array([np.tanh(bt)/(L*bt*(1.0 - ((1.0 - 2.0*k/L)*np.tanh(bt))**2.0)) for k in xPts])
        xixx = np.array([-4.0*(np.tanh(bt)**3.0)*(1.0 - 2.0*k/L)/(L**2*bt*(1.0 - (np.tanh(bt)*(1.0 - 2.0*k/L))**2.0)**2.0) for k in xPts])
        xix2 = np.array([k*k for k in xi_x])
    else:
        xPts = L*xi
        xi_x = np.ones_like(xPts)
        xix2 = xi_x**2
        xixx = np.zeros_like(xPts)

    return xi, xPts, xi_x, xixx, xix2

xi, xPts, xi_x, xixx, xix2 = genGrid(Nx, Lx, btX)
et, yPts, et_y, etyy, ety2 = genGrid(Ny, Ly, btY)
zt, zPts, zt_z, ztzz, ztz2 = genGrid(Nz, Lz, btZ)

npax = np.newaxis
xi_x, xixx, xix2 = xi_x[:, npax, npax], xixx[:, npax, npax], xix2[:, npax, npax]
et_y, etyy, ety2 = et_y[:, npax], etyy[:, npax], ety2[:, npax]
