import numpy as np
import globalData as glob
from globalData import Nx, Ny, Nz
import pyfftw.interfaces.numpy_fft as fp


def fft_3d(Ak):
    return fp.rfftn(Ak)/(Nx*Ny*Nz)


def calcShellSpectrum(fk, temp, kSqr, nlin):
    mr = glob.minRad
    al = glob.arrLim

    kS = np.zeros(al + 2)
    ek = np.zeros(al + 2)
    if glob.cmpTrn:
        Tk = np.zeros(al + 2)
    else:
        Tk = 0

    temp[:,:,:] = fk[:,:,:]
    temp[-1:Nx//2:-1, :, 0] = complex(0, 0)
    temp[0, Ny-1:Ny//2:-1, 0] = complex(0, 0)

    kS[0] = 0
    ek[0] = (abs(fk[0,0,0])**2)/2
    if glob.cmpTrn:
        Tk[0] = -(nlin[0,0,0]*np.conjugate(fk[0,0,0])).real

    k = glob.kInt
    shInd = 1
    while k < mr + glob.kInt:
        kS[shInd] = k
        index = np.where((kSqr > (k - glob.kInt)**2) & (kSqr <= k**2))
        ek[shInd] = np.sum(np.abs(temp[index])**2)
        if glob.cmpTrn:
            Tk[shInd] = -2.0*np.sum(((nlin*np.conjugate(temp)).real)[index])

        #print("\t\tCompleted for k = {0:9.3f} out of {1:9.3f}".format(k, mr+glob.kInt))
        shInd += 1
        k += glob.kInt

    return kS, ek, Tk


def computeFFT(nlx, nly, nlz):
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
        nlinx = fft_3d(nlx)
        nliny = fft_3d(nly)
        nlinz = fft_3d(nlz)
    else:
        nlinx, nliny, nlinz = 0, 0, 0

    print("\tCalculating shell spectrum")
    # Calculate shell spectrum
    kShell, ekx, Tkx = calcShellSpectrum(uk, temp, kSqr, nlinx)
    kShell, eky, Tky = calcShellSpectrum(vk, temp, kSqr, nliny)
    kShell, ekz, Tkz = calcShellSpectrum(wk, temp, kSqr, nlinz)

    return kShell, ekx, eky, ekz, Tkx, Tky, Tkz

