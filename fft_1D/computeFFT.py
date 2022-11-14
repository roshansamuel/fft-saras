import numpy as np
import globalData as glob
from globalData import Nx, Nz
import pyfftw.interfaces.numpy_fft as fp


def fft_1d(Ak):
    return fp.rfft(Ak, axis=0)/Nx


def calcSpectrum(fk, nlin):
    ek = np.abs(fk**2)
    Tk = -(nlin*np.conjugate(fk)).real

    return ek, Tk


def computeFFT(uu):
    kx = np.arange(0, Nx//2 + 1, 1)
    kx = kx*glob.kFactor

    uk = fft_1d(glob.U)
    wk = fft_1d(glob.W)
    tk = fft_1d(glob.T)

    nlin = 1j*(kx[:,np.newaxis]*fft_1d(uu/2.0))

    print("\tCalculating spectrum")
    # Calculate shell spectrum
    Ek, Tk = calcSpectrum(uk, nlin)

    return Ek, Tk

