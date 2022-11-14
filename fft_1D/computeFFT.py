import numpy as np
import globalData as glob
from globalData import Nx, Nz
import pyfftw.interfaces.numpy_fft as fp


def fft_1d(Ak):
    return fp.rfft(Ak, axis=0)/Nx


def calcSpectrum(fk, temp, kSqr, nlin):
    ek = np.abs(fk**2)
    Tk = -(nlin*np.conjugate(fk)).real

    '''
    al = glob.arrLim
    ek = np.zeros((al, fk.shape[1]))
    Tk = np.zeros((al, fk.shape[1]))

    temp[:,:] = fk[:,:]

    ek[0,:] = (abs(fk[0,:])**2)/2
    Tk[0,:] = -(nlin[0,:]*np.conjugate(fk[0,:])).real
    for k in range(1, al):
        index = np.where((kSqr > glob.kShell[k-1]**2) & (kSqr <= glob.kShell[k]**2))
        ek[k,:] = np.sum(np.abs(temp[index,:])**2, axis=0)/glob.dk[k-1]
        Tk[k,:] = -2.0*np.sum(((nlin*np.conjugate(temp)).real)[index,:], axis=0)/glob.dk[k-1]
    '''

    return ek, Tk


def computeFFT(uu):
    temp = np.zeros((Nx//2 + 1, Nz), dtype="complex")

    kx = np.arange(0, Nx//2 + 1, 1)
    kx = kx*glob.kFactor

    kSqr = kx**2

    uk = fft_1d(glob.U)
    wk = fft_1d(glob.W)
    tk = fft_1d(glob.T)

    nlin = 1j*(kx[:,np.newaxis]*fft_1d(uu/2.0))

    print("\tCalculating spectrum")
    # Calculate shell spectrum
    Ek, Tk = calcSpectrum(uk, temp, kSqr, nlin)

    return Ek, Tk

