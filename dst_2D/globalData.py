import numpy as np
import yamlRead as yr
from scipy.optimize import fsolve as fsol

dataDir = "../data/data_2D/2_Ra_1e9_pySaras/"

# Is YAML file available?
readYAML = True

# Global variables
Lx, Lz = 1.0, 1.0
Nx, Nz = 100, 100
btX, btZ = 0.0, 0.0
U, W, T, X, Z = 1, 1, 1, 1, 1
nlx, nlz, nlT = 1, 1, 1

# Variables to be considered - useful for very large files/small memory
# 0 -> both V and T
# 1 -> only V
# 2 -> only T
varMode = 0

# Limit kShells
kLim = False

# If limiting shells, set number of shells
nShell = 300

# Should transfer function be computed?
cmpTrn = True

# If transfer function is computed, should nlin term be computed in real space?
realNLin = True

# If computing nlin term in real space, should conservative form be used?
consNLin = False

# Use theta (temperature fluctuation) instead of T (temperature)
useTheta = True

# Read existing FFT vs compute anew
readFile = False

# Specify starting time for calculations. Use 0 to include all in timeList.dat
startTime = 0.0

# Specify ending time for calculations. Use Inf to include all in timeList.dat
stopTime = float('Inf')

# If YAML file is available, parse it
if readYAML:
    yr.parseYAML(dataDir)

# Generate kShells
# Basis - FS
# With F along X and S along Z, max k is halved only along X direction, and kFactor is 2*pi/L for X and pi/L for Z
nGrid = np.array([Nx//2, Nz])
kFactor = np.array([2.0*np.pi/Lx, np.pi/Lz])

kInt = min(kFactor)
maxRad = np.sqrt(np.dot(nGrid*kFactor, nGrid*kFactor))

# Function to find beta factor for given max shell radius, minimum spacing and number of shells
def minTanSpace(bt, L, N, delK):
    return delK - L*(1.0 - np.tanh(bt*(N - 2)/(N - 1))/np.tanh(bt))

# If kShells must be limited, use tan-hyp spacing
if kLim:
    beta = fsol(minTanSpace, 1.0, (maxRad, nShell, kInt))[0]
    xi = np.linspace(0.0, 1.0, nShell)
    kShell = maxRad*(1.0 - np.tanh(beta*(1.0 - xi))/np.tanh(beta))

    # Reset first 2 shells to remove floating point errors
    kShell[1] = kInt
    kShell[2] = 2.0*kInt
else:
    kShell = np.arange(0, maxRad, kInt)

dk = np.diff(kShell)
arrLim = kShell.shape[0]

# Kinetic energy spectrum
ekx = np.zeros_like(kShell)
ekz = np.zeros_like(kShell)
Eku = np.zeros_like(kShell)

# KE transfer function and flux
Tku = np.zeros_like(kShell)
Pku = np.zeros_like(kShell)

# KE forcing and dissipation
Fku = np.zeros_like(kShell)
Dku = np.zeros_like(kShell)

# Thermal energy spectrum
EkT = np.zeros_like(kShell)

# TE transfer function and flux
TkT = np.zeros_like(kShell)
PkT = np.zeros_like(kShell)

# TE dissipation
DkT = np.zeros_like(kShell)

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
        xi_x = np.ones_like(xPts)/L
        xix2 = xi_x**2
        xixx = np.zeros_like(xPts)

    return xi, xPts, xi_x, xixx, xix2

xi, xPts, xi_x, xixx, xix2 = genGrid(Nx, Lx, btX)
zt, zPts, zt_z, ztzz, ztz2 = genGrid(Nz, Lz, btZ)

npax = np.newaxis
xi_x, xixx, xix2 = xi_x[:, npax], xixx[:, npax], xix2[:, npax]
