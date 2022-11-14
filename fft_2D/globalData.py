import numpy as np
import yamlRead as yr

dataDir = "../data/data_2D/1_Ra_1e11/"

# Is YAML file available?
readYAML = True

# Global variables
Lx, Lz = 1.0, 1.0
Nx, Nz = 100, 100
btX, btZ = 0.0, 0.0
U, W, T, X, Z = 1, 1, 1, 1, 1
nlx, nlz, nlT = 1, 1, 1

# Limit kShells
kLim = True

# Should transfer function be computed?
cmpTrn = True

# If transfer function is computed, should nlin term be computed in real space?
realNLin = False

# If computing nlin term in real space, should conservative form be used?
consNLin = True

# Read existing FFT vs compute anew
readFile = False

# If YAML file is available, parse it
if readYAML:
    yr.parseYAML(dataDir)

nGrid = np.array([Nx, Nz])
kFactor = np.array([2.0*np.pi/Lx, 2.0*np.pi/Lz])
kInt = min(kFactor)

minRad = np.sqrt(np.dot((nGrid//2)*kFactor, (nGrid//2)*kFactor))

# Generate kShells
kShell = np.arange(0, minRad, kInt)
arrLim = kShell.shape[0]

# If kShells must be limited, change
minStr = 32
if kLim:
    sInd = 0
    indJump = 1
    kNew = []
    while True:
        kNew.append(kShell[sInd])
        sInd += indJump

        if kNew[-1] > 200:
            if len(kNew)%minStr == 0:
                indJump += 2

            if sInd >= arrLim:
                break

    kShell = np.array(kNew)

dk = np.diff(kShell)
arrLim = kShell.shape[0]

tVol = Lx*Lz

dXi = 1.0/(Nx)
dZt = 1.0/(Nz)

i2hx = 1.0/(2.0*dXi)
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
zt, zPts, zt_z, ztzz, ztz2 = genGrid(Nz, Lz, btZ)

npax = np.newaxis
xi_x, xixx, xix2 = xi_x[:, npax], xixx[:, npax], xix2[:, npax]
