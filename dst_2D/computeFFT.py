import numpy as np
import globalData as glob
from globalData import Nx, Nz
from mpi4py_fft import fftw as fft


def cos_2d(f):
    a = fft.aligned((Nx, Nz), dtype=np.float)
    b = fft.aligned_like(a)
    c = fft.aligned((Nx//2+1, Nz), dtype=np.complex)

    zFFT = fft.dctn(a, axes=[1])
    xFFT = fft.rfftn(b, axes=[0])

    a[...] = f
    b = zFFT(a, b)/(Nz*2)
    c = xFFT(b, c)/Nx

    return c


def sin_2d(f):
    a = fft.aligned((Nx, Nz), dtype=np.float)
    b = fft.aligned_like(a)
    c = fft.aligned((Nx//2+1, Nz), dtype=np.complex)

    zFFT = fft.dstn(a, axes=[1])
    xFFT = fft.rfftn(b, axes=[0])

    a[...] = f
    b = zFFT(a, b)/(Nz*2)
    c = xFFT(b, c)/Nx

    return c


def calcForceSpectrum(fk, temp, kSqr):
    al = glob.arrLim

    Fk = np.zeros(al)

    temp[:,:] = fk[:,:]

    Fk[0] = abs(fk[0,0])/2
    for k in range(1, al):
        index = np.where((kSqr > glob.kShell[k-1]**2) & (kSqr <= glob.kShell[k]**2))
        Fk[k] = np.sum(temp[index])/glob.dk[k-1]

    return Fk


def calcShellSpectrum(fk, temp, kSqr, nlin):
    al = glob.arrLim

    ek = np.zeros(al)
    if glob.cmpTrn:
        Tk = np.zeros(al)
    else:
        Tk = 0

    temp[...] = fk[...]
    mFactor = 2*np.ones_like(temp, dtype=np.float)
    mFactor[0, :] = 1.0

    ek[0] = abs(temp[0,0])**2
    if glob.cmpTrn:
        Tk[0] = -(nlin[0,0]*np.conjugate(temp[0,0])).real

    for k in range(1, al):
        index = np.where((kSqr > glob.kShell[k-1]**2) & (kSqr <= glob.kShell[k]**2))
        ek[k] = np.sum(mFactor[index]*np.abs(temp[index])**2)/glob.dk[k-1]
        if glob.cmpTrn:
            Tk[k] = -np.sum(((mFactor*(nlin[...]*np.conjugate(temp[...]))).real)[index])/glob.dk[k-1]

    # This has been disabled since the energy equation in SF basis may already have this factor
    #ek = ek/2

    return ek, Tk


def computeFFT(uu, uw, ww, uT, wT):
    if glob.varMode == 0:
        uk = cos_2d(glob.U)
        wk = sin_2d(glob.W)
        tk = sin_2d(glob.T)
    elif glob.varMode == 1:
        uk = cos_2d(glob.U)
        wk = sin_2d(glob.W)
    elif glob.varMode == 2:
        wk = sin_2d(glob.W)
        tk = sin_2d(glob.T)

    kxf = np.arange(0, Nx//2 + 1, 1)
    kzs = np.arange(0, Nz, 1) + 1
    kzc = np.arange(0, Nz, 1)

    kxf = kxf*glob.kFactor[0]
    kzs = kzs*glob.kFactor[1]
    kzc = kzc*glob.kFactor[1]

    if glob.cmpTrn:
        if glob.realNLin:
            if glob.varMode == 0:
                nlinx = cos_2d(glob.nlx)
                nlinz = sin_2d(glob.nlz)
                nlinT = sin_2d(glob.nlT)
            elif glob.varMode == 1:
                nlinx = cos_2d(glob.nlx)
                nlinz = sin_2d(glob.nlz)
            elif glob.varMode == 2:
                nlinT = sin_2d(glob.nlT)
        else:
            # The below formulae are wrong. Always use realNLin = True in SF basis.
            if glob.varMode == 0:
                nlinx = 1j*(kxf[:, np.newaxis]*cos_2d(uu)) + kzs*sin_2d(uw)
                nlinz = 1j*(kxf[:, np.newaxis]*sin_2d(uw)) - kzc*cos_2d(ww)
                nlinT = 1j*(kxf[:, np.newaxis]*sin_2d(uT)) - kzc*cos_2d(wT)
            elif glob.varMode == 1:
                nlinx = 1j*(kxf[:, np.newaxis]*cos_2d(uu)) + kzs*sin_2d(uw)
                nlinz = 1j*(kxf[:, np.newaxis]*sin_2d(uw)) - kzc*cos_2d(ww)
            elif glob.varMode == 2:
                nlinT = 1j*(kxf[:, np.newaxis]*sin_2d(uT)) - kzc*cos_2d(wT)
    else:
        nlinx, nlinz, nlinT = 0, 0, 0

    # Calculate shell spectrum
    print("\tCalculating shell spectrum")
    temp = np.zeros((Nx//2 + 1, Nz), dtype="complex")

    if glob.varMode == 0:
        kX, kZ = np.meshgrid(kxf, kzc, indexing='ij')
        kSqr = kX**2 + kZ**2
        glob.ekx, Tkx = calcShellSpectrum(uk, temp, kSqr, nlinx)

        kX, kZ = np.meshgrid(kxf, kzs, indexing='ij')
        kSqr = kX**2 + kZ**2
        glob.ekz, Tkz = calcShellSpectrum(wk, temp, kSqr, nlinz)
        glob.EkT, glob.TkT = calcShellSpectrum(tk, temp, kSqr, nlinT)

        glob.Eku = glob.ekx + glob.ekz
        glob.Tku = Tkx + Tkz
    elif glob.varMode == 1:
        kX, kZ = np.meshgrid(kxf, kzc, indexing='ij')
        kSqr = kX**2 + kZ**2
        glob.ekx, Tkx = calcShellSpectrum(uk, temp, kSqr, nlinx)

        kX, kZ = np.meshgrid(kxf, kzs, indexing='ij')
        kSqr = kX**2 + kZ**2
        glob.ekz, Tkz = calcShellSpectrum(wk, temp, kSqr, nlinz)

        glob.Eku = glob.ekx + glob.ekz
        glob.Tku = Tkx + Tkz
    elif glob.varMode == 2:
        kX, kZ = np.meshgrid(kxf, kzs, indexing='ij')
        kSqr = kX**2 + kZ**2
        glob.EkT, glob.TkT = calcShellSpectrum(tk, temp, kSqr, nlinT)

    if glob.varMode in [0, 2]:
        print("\tCalculating F(k)")
        temp = np.zeros((Nx//2 + 1, Nz))
        glob.Fku = calcForceSpectrum((np.conjugate(wk)*tk).real, temp, kSqr)

    print("\tCalculating D(k)")
    if glob.varMode == 0:
        glob.Dku = (glob.kShell**2)*glob.Eku
        glob.DkT = (glob.kShell**2)*glob.EkT
    elif glob.varMode == 1:
        glob.Dku = (glob.kShell**2)*glob.Eku
    elif glob.varMode == 2:
        glob.DkT = (glob.kShell**2)*glob.EkT

