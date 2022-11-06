import numpy as np
import globalData as glob
from globalData import Nx, Nz
import pyfftw.interfaces.numpy_fft as fp


def fft_2d(Ak):
    return fp.rfftn(Ak)/(Nx*Nz)


def calcShellSpectrum(fk, temp, kSqr, nlin):
    al = glob.arrLim

    ek = np.zeros(al + 1)
    if glob.cmpTrn:
        Tk = np.zeros(al + 1)
    else:
        Tk = 0

    temp[:,:] = fk[:,:]
    temp[-1:Nx//2:-1, 0] = complex(0, 0)

    ek[0] = (abs(fk[0,0])**2)/2
    if glob.cmpTrn:
        Tk[0] = -(nlin[0,0]*np.conjugate(fk[0,0])).real

    '''
    k = glob.kInt
    shInd = 1
    while k < glob.minRad:
        index = np.where((kSqr > (k - glob.kInt)**2) & (kSqr <= k**2))
        ek[shInd] = np.sum(np.abs(temp[index])**2)
        if glob.cmpTrn:
            Tk[shInd] = -2.0*np.sum(((nlin*np.conjugate(temp)).real)[index])

        #print("\t\tCompleted for k = {0:9.3f} out of {1:9.3f}".format(k, glob.minRad+glob.kInt))
        shInd += 1
        k += glob.kInt
    '''

    for k in range(1, al+1):
        index = np.where((kSqr > glob.kShell[k-1]**2) & (kSqr <= glob.kShell[k]**2))
        ek[k] = np.sum(np.abs(temp[index])**2)
        if glob.cmpTrn:
            Tk[k] = -2.0*np.sum(((nlin*np.conjugate(temp)).real)[index])

        #print("\t\tCompleted for k = {0:9.3f} out of {1:9.3f}".format(k, glob.minRad+glob.kInt))

    return ek, Tk


def computeFFT(nlx, nlz):
    temp = np.zeros((Nx, Nz//2 + 1), dtype="complex")

    kx = np.arange(0, Nx, 1)
    kz = np.arange(0, Nz//2 + 1, 1)

    # Shift the wavenumbers
    kx[Nx//2+1:] = kx[Nx//2+1:] - Nx

    kx = kx*glob.kFactor[0]
    kz = kz*glob.kFactor[1]

    kX, kZ = np.meshgrid(kx, kz, indexing='ij')

    kSqr = kX**2 + kZ**2

    uk = fft_2d(glob.U)
    wk = fft_2d(glob.W)
    tk = fft_2d(glob.T)

    if glob.cmpTrn:
        nlinx = fft_2d(nlx)
        nlinz = fft_2d(nlz)
    else:
        nlinx, nlinz = 0, 0

    print("\tCalculating shell spectrum")
    # Calculate shell spectrum
    ekx, Tkx = calcShellSpectrum(uk, temp, kSqr, nlinx)
    ekz, Tkz = calcShellSpectrum(wk, temp, kSqr, nlinz)

    return ekx, ekz, Tkx, Tkz

