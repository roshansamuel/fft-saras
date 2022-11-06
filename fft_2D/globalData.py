import numpy as np

inputDir = "../data_2D/"

# Global variables
Lx, Lz = 2.0, 1.0
Nx, Nz = 2048, 1024
btX, btZ = 0.0, 1.3
U, W, T, X, Z = 1, 1, 1, 1, 1

# Should transfer function be computed?
cmpTrn = False
# Read existing FFT vs compute anew
readFile = False

nGrid = np.array([Nx, Nz])
kFactor = np.array([2.0*np.pi/Lx, 2.0*np.pi/Lz])
kInt = min(kFactor)

minRad = np.sqrt(np.dot((nGrid//2)*kFactor, (nGrid//2)*kFactor))
arrLim = int(minRad/kInt)

# Generate kShell
#kShell = np.arange(0, minRad, kInt)
def genShells():
    k = kInt
    kS = np.zeros(arrLim + 1)
    shInd = 1
    kS[0] = 0
    while k < minRad:
        kS[shInd] = k
        shInd += 1
        k += kInt

    return kS

kShell = genShells()

tVol = Lx*Lz

# Generate index list
def genIndex():
    kx = np.arange(0, Nx, 1)
    kz = np.arange(0, Nz//2 + 1, 1)

    # Shift the wavenumbers
    kx[Nx//2+1:] = kx[Nx//2+1:] - Nx

    kx = kx*kFactor[0]
    kz = kz*kFactor[1]

    kX, kZ = np.meshgrid(kx, kz, indexing='ij')

    kSqr = kX**2 + kZ**2

    indList = []

    k = kInt
    shInd = 0
    while k < minRad + kInt:
        indList.append(np.where((kSqr > (k - kInt)**2) & (kSqr <= k**2)))
        shInd += 1

    print(len(indList))
    exit()

    return indList

#indexList = genIndex()

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
