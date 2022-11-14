import numpy as np
import globalData as glob
from globalData import Nx, Nz
import pyfftw.interfaces.numpy_fft as fp


def fft_2d(Ak):
    return fp.rfftn(Ak)/(Nx*Nz)


def calcShellSpectrum(fk, temp, kSqr, nlin):
    al = glob.arrLim

    ek = np.zeros(al)
    if glob.cmpTrn:
        Tk = np.zeros(al)
    else:
        Tk = 0

    temp[:,:] = fk[:,:]
    temp[-1:Nx//2:-1, 0] = complex(0, 0)

    ek[0] = (abs(fk[0,0])**2)/2
    if glob.cmpTrn:
        Tk[0] = -(nlin[0,0]*np.conjugate(fk[0,0])).real

    for k in range(1, al):
        index = np.where((kSqr > glob.kShell[k-1]**2) & (kSqr <= glob.kShell[k]**2))
        ek[k] = np.sum(np.abs(temp[index])**2)/glob.dk[k-1]
        if glob.cmpTrn:
            Tk[k] = -2.0*np.sum(((nlin*np.conjugate(temp)).real)[index])/glob.dk[k-1]

    return ek, Tk


def computeFFT(uu, uw, ww, uT, wT):
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
        if glob.realNLin:
            nlinx = fft_2d(glob.nlx)
            nlinz = fft_2d(glob.nlz)
            nlinT = fft_2d(glob.nlT)
        else:
            nlinx = 1j*(kx[:, np.newaxis]*fft_2d(uu) + kz*fft_2d(uw))
            nlinz = 1j*(kx[:, np.newaxis]*fft_2d(uw) + kz*fft_2d(ww))
            nlinT = 1j*(kx[:, np.newaxis]*fft_2d(uT) + kz*fft_2d(wT))
    else:
        nlinx, nlinz, nlinT = 0, 0, 0

    print("\tCalculating shell spectrum")
    # Calculate shell spectrum
    ekx, Tkx = calcShellSpectrum(uk, temp, kSqr, nlinx)
    ekz, Tkz = calcShellSpectrum(wk, temp, kSqr, nlinz)
    ekT, TkT = calcShellSpectrum(tk, temp, kSqr, nlinT)

    return ekx, ekz, Tkx, Tkz, ekT, TkT

