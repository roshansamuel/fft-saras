import numpy as np
import globalData as glob
from globalData import Nx, Ny, Nz
import pyfftw.interfaces.numpy_fft as fp


def fft_3d(Ak):
    return fp.rfftn(Ak)/(Nx*Ny*Nz)


def calcShellSpectrum(fk, temp, kSqr, nlin):
    al = glob.arrLim

    ek = np.zeros(al)
    if glob.cmpTrn:
        Tk = np.zeros(al)
    else:
        Tk = 0

    temp[:,:,:] = fk[:,:,:]
    temp[-1:Nx//2:-1, :, 0] = complex(0, 0)
    temp[0, Ny-1:Ny//2:-1, 0] = complex(0, 0)

    ek[0] = (abs(fk[0,0,0])**2)/2
    if glob.cmpTrn:
        Tk[0] = -(nlin[0,0,0]*np.conjugate(fk[0,0,0])).real

    for k in range(1, al):
        index = np.where((kSqr > glob.kShell[k-1]**2) & (kSqr <= glob.kShell[k]**2))
        ek[k] = np.sum(np.abs(temp[index])**2)/glob.dk[k-1]
        if glob.cmpTrn:
            Tk[k] = -2.0*np.sum(((nlin*np.conjugate(temp)).real)[index])/glob.dk[k-1]

        #print("\t\tCompleted for k = {0:3d} out of {1:3d}".format(k, al+1))

    return ek, Tk


def computeFFT(uu, vv, ww, uv, vw, wu):
    temp = np.zeros((Nx, Ny, Nz//2 + 1), dtype="complex")

    kx = np.arange(0, Nx, 1)
    ky = np.arange(0, Ny, 1)
    kz = np.arange(0, Nz//2 + 1, 1)

    # Shift the wavenumbers
    kx[Nx//2+1:] = kx[Nx//2+1:] - Nx
    ky[Ny//2+1:] = ky[Ny//2+1:] - Ny

    kx = kx*glob.kFactor[0]
    ky = ky*glob.kFactor[1]
    kz = kz*glob.kFactor[2]

    kX, kY, kZ = np.meshgrid(kx, ky, kz, indexing='ij')

    kSqr = kX**2 + kY**2 + kZ**2

    uk = fft_3d(glob.U)
    vk = fft_3d(glob.V)
    wk = fft_3d(glob.W)
    tk = fft_3d(glob.T)

    if glob.cmpTrn:
        if glob.realNLin:
            nlinx = fft_3d(glob.nlx)
            nliny = fft_3d(glob.nly)
            nlinz = fft_3d(glob.nlz)
        else:
            nlinx = 1j*(kx[:, np.newaxis, np.newaxis]*fft_2d(uu) + ky[:, np.newaxis]*fft_2d(uv) + kz*fft_2d(wu))
            nliny = 1j*(kx[:, np.newaxis, np.newaxis]*fft_2d(uv) + ky[:, np.newaxis]*fft_2d(vv) + kz*fft_2d(vw))
            nlinz = 1j*(kx[:, np.newaxis, np.newaxis]*fft_2d(wu) + ky[:, np.newaxis]*fft_2d(vw) + kz*fft_2d(ww))
    else:
        nlinx, nliny, nlinz = 0, 0, 0

    print("\tCalculating shell spectrum")
    # Calculate shell spectrum
    ekx, Tkx = calcShellSpectrum(uk, temp, kSqr, nlinx)
    eky, Tky = calcShellSpectrum(vk, temp, kSqr, nliny)
    ekz, Tkz = calcShellSpectrum(wk, temp, kSqr, nlinz)

    return ekx, eky, ekz, Tkx, Tky, Tkz

