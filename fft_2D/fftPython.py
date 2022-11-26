import sys
import h5py as hp
import numpy as np
import plotData as plt
import nlinCalc as nlin
import computeFFT as fft
import globalData as glob
from scipy import interpolate
from globalData import Nx, Nz
import scipy.integrate as integrate

print()

def loadData(fileName):
    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    glob.U = np.pad(np.array(sFile["Vx"]), 1)
    glob.W = np.pad(np.array(sFile["Vz"]), 1)
    glob.T = np.pad(np.array(sFile["T"]), 1)

    glob.X = np.pad(np.array(sFile["X"]), (1, 1), 'constant', constant_values=(0, glob.Lx))
    glob.Z = np.pad(np.array(sFile["Z"]), (1, 1), 'constant', constant_values=(0, glob.Lz))

    sFile.close()

    imposeBCs()

    # Subtract mean profile
    if glob.useTheta:
        glob.T -= (1 - glob.Z)


def periodicBC(f):
    f[0,:], f[-1,:] = f[-2,:], f[1,:]


def imposeBCs():
    # Periodic along X
    glob.X[0], glob.X[-1] = -glob.X[1], glob.Lx + glob.X[1]
    periodicBC(glob.U)
    periodicBC(glob.W)
    periodicBC(glob.T)

    # RBC
    glob.T[:,0], glob.T[:,-1] = 1.0, 0.0


def interpolateData(f, xO, zO):
    intFunct = interpolate.interp1d(glob.Z, f, kind='cubic', axis=1)
    f = intFunct(zO)
    intFunct = interpolate.interp1d(glob.X, f, kind='cubic', axis=0)
    f = intFunct(xO)

    return f


def uniformInterp():
    xU = np.linspace(0.0, glob.Lx, Nx)
    zU = np.linspace(0.0, glob.Lz, Nz)

    glob.U = interpolateData(glob.U, xU, zU)
    glob.W = interpolateData(glob.W, xU, zU)
    glob.T = interpolateData(glob.T, xU, zU)

    if glob.cmpTrn and glob.realNLin:
        glob.nlx = interpolateData(glob.nlx, xU, zU)
        glob.nlz = interpolateData(glob.nlz, xU, zU)
        glob.nlT = interpolateData(glob.nlT, xU, zU)

    glob.X = xU
    glob.Z = zU


def energyCheck(Ek):
    ke = (glob.U**2 + glob.W**2)/2.0
    keInt = integrate.simps(integrate.simps(ke, glob.Z), glob.X)/glob.tVol
    print("\t\tReal field energy =     {0:10.8f}".format(keInt))

    keInt = np.sum(np.dot(Ek[1:], glob.dk)) + Ek[0]
    print("\t\tShell spectrum energy = {0:10.8f}".format(keInt))


def readFFT(tVal):
    fileName = glob.dataDir + "output/FFT_{0:09.4f}.h5".format(tVal)

    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    kShell = np.array(sFile["kShell"])
    ekx = np.array(sFile["ekx"])
    ekz = np.array(sFile["ekz"])
    EkT = np.array(sFile["EkT"])

    if glob.cmpTrn:
        Tku = np.array(sFile["Tku"])
        Pku = np.array(sFile["Pku"])

        TkT = np.array(sFile["TkT"])
        PkT = np.array(sFile["PkT"])
    else:
        Tku = np.zeros_like(ekx)
        Pku = np.zeros_like(ekx)
        TkT = np.zeros_like(ekx)
        PkT = np.zeros_like(ekx)

    sFile.close()

    return kShell, ekx, ekz, Tku, Pku, EkT, TkT, PkT


def writeFFT(tVal, ekx, ekz, Tku, Pku, Fku, Dku, EkT, TkT, PkT, DkT):
    fileName = glob.dataDir + "output/FFT_{0:09.4f}.h5".format(tVal)

    print("\tWriting into file ", fileName)
    sFile = hp.File(fileName, 'a')

    if "kShell" not in sFile:
        dset = sFile.create_dataset("kShell", data = glob.kShell)

    dset = sFile.create_dataset("ekx", data = ekx)
    dset = sFile.create_dataset("ekz", data = ekz)
    dset = sFile.create_dataset("EkT", data = EkT)
    dset = sFile.create_dataset("Fku", data = Fku)
    dset = sFile.create_dataset("Dku", data = Dku)

    if glob.cmpTrn:
        dset = sFile.create_dataset("Tku", data = Tku)
        dset = sFile.create_dataset("Pku", data = Pku)

        dset = sFile.create_dataset("TkT", data = TkT)
        dset = sFile.create_dataset("PkT", data = PkT)

    sFile.close()


def main():
    # Set some global variables from CLI arguments
    argList = sys.argv[1:]
    if argList and len(argList) == 2:
        glob.startTime = float(argList[0])
        glob.stopTime = float(argList[1])

    # Load timelist
    tList = np.loadtxt(glob.dataDir + "output/timeList.dat", comments='#')

    for i in range(tList.shape[0]):
        tVal = tList[i]
        if tVal > glob.startTime and tVal < glob.stopTime:
            if glob.readFile:
                kShell, ekx, ekz, Tku, Pku, EkT, TkT, PkT = readFFT(tVal)

                Eku = ekx + ekz

            else:
                fileName = glob.dataDir + "output/Soln_{0:09.4f}.h5".format(tVal)
                loadData(fileName)

                # Compute non-linear terms
                if glob.cmpTrn and glob.realNLin:
                    print("\tComputing non-linear term")
                    glob.nlx, glob.nlz, glob.nlT = nlin.computeNLin()

                    periodicBC(glob.nlx)
                    periodicBC(glob.nlz)
                    periodicBC(glob.nlT)
                else:
                    glob.nlx, glob.nlz, glob.nlT = 0, 0, 0

                # Interpolate data to uniform grid
                print("\tInterpolating to uniform grid")
                uniformInterp()

                print("\tComputing FFT")
                if glob.realNLin:
                    ekx, ekz, Tkx, Tkz, EkT, TkT, Fku, Dku, DkT = fft.computeFFT(0, 0, 0, 0, 0)
                else:
                    ekx, ekz, Tkx, Tkz, EkT, TkT, Fku, Dku, DkT = fft.computeFFT(glob.U*glob.U, glob.U*glob.W, glob.W*glob.W, glob.U*glob.T, glob.W*glob.T)

                Eku = ekx + ekz
                if glob.cmpTrn:
                    Tku = Tkx + Tkz

                    Pku = np.zeros_like(Tku)
                    Pku[0] = -Tku[0]
                    Pku[1:] = -np.cumsum(Tku[1:]*glob.dk, axis=0) + Pku[0]

                    PkT = np.zeros_like(TkT)
                    PkT[0] = -TkT[0]
                    PkT[1:] = -np.cumsum(TkT[1:]*glob.dk, axis=0) + PkT[0]
                else:
                    Tku = np.zeros_like(Eku)
                    Pku = np.zeros_like(Eku)
                    PkT = np.zeros_like(EkT)

                writeFFT(tVal, ekx, ekz, Tku, Pku, Fku, Dku, EkT, TkT, PkT, DkT)

                print("\tChecking energy balance")
                energyCheck(Eku)

    #plt.plotStuff((10, 12), [0])

main()
