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

    if glob.varMode != 1:
        glob.T = np.pad(np.array(sFile["T"]), 1)

    glob.X = np.pad(np.array(sFile["X"]), (1, 1), 'constant', constant_values=(0, glob.Lx))
    glob.Z = np.pad(np.array(sFile["Z"]), (1, 1), 'constant', constant_values=(0, glob.Lz))

    sFile.close()

    imposeBCs()

    # Subtract mean profile
    if glob.varMode in [0, 2] and glob.useTheta:
        glob.T -= (1 - glob.Z)


def periodicBC(f):
    f[0,:], f[-1,:] = f[-2,:], f[1,:]


def imposeBCs():
    # Periodic along X
    glob.X[0], glob.X[-1] = -glob.X[1], glob.Lx + glob.X[1]

    periodicBC(glob.U)
    periodicBC(glob.W)

    if glob.varMode != 1:
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
    xU = np.linspace(0.0, glob.Lx, Nx, endpoint=False)

    zO = np.linspace(0.0, glob.Lz, Nz+1)
    zU = (zO[1:] + zO[:-1])/2

    glob.U = interpolateData(glob.U, xU, zU)
    glob.W = interpolateData(glob.W, xU, zU)
    if glob.varMode != 1:
        glob.T = interpolateData(glob.T, xU, zU)

    if glob.cmpTrn and glob.realNLin:
        if glob.varMode == 0:
            glob.nlx = interpolateData(glob.nlx, xU, zU)
            glob.nlz = interpolateData(glob.nlz, xU, zU)
            glob.nlT = interpolateData(glob.nlT, xU, zU)
        elif glob.varMode == 1:
            glob.nlx = interpolateData(glob.nlx, xU, zU)
            glob.nlz = interpolateData(glob.nlz, xU, zU)
        elif glob.varMode == 2:
            glob.nlT = interpolateData(glob.nlT, xU, zU)

    glob.X = xU
    glob.Z = zU


def energyCheck():
    ke = (glob.U**2 + glob.W**2)/2.0
    keInt = integrate.simps(integrate.simps(ke, glob.Z), glob.X)/glob.tVol
    print("\t\tReal field energy =     {0:10.8f}".format(keInt))

    keInt = np.sum(np.dot(glob.Eku[1:], glob.dk)) + glob.Eku[0]
    print("\t\tShell spectrum energy = {0:10.8f}".format(keInt))


def readFFT(tVal):
    fileName = glob.dataDir + "output/FFT_{0:09.4f}.h5".format(tVal)

    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    glob.kShell = np.array(sFile["kShell"])

    if glob.varMode == 0:
        glob.ekx = np.array(sFile["ekx"])
        glob.ekz = np.array(sFile["ekz"])
        glob.Eku = np.array(sFile["Eku"])
        glob.Tku = np.array(sFile["Tku"])
        glob.Pku = np.array(sFile["Pku"])
        glob.Fku = np.array(sFile["Fku"])
        glob.Dku = np.array(sFile["Dku"])

        glob.EkT = np.array(sFile["EkT"])
        glob.TkT = np.array(sFile["TkT"])
        glob.PkT = np.array(sFile["PkT"])
        glob.DkT = np.array(sFile["DkT"])
    elif glob.varMode == 1:
        glob.ekx = np.array(sFile["ekx"])
        glob.ekz = np.array(sFile["ekz"])
        glob.Eku = np.array(sFile["Eku"])
        glob.Tku = np.array(sFile["Tku"])
        glob.Pku = np.array(sFile["Pku"])
        glob.Dku = np.array(sFile["Dku"])
    elif glob.varMode == 2:
        glob.EkT = np.array(sFile["EkT"])
        glob.TkT = np.array(sFile["TkT"])
        glob.PkT = np.array(sFile["PkT"])
        glob.DkT = np.array(sFile["DkT"])
        glob.Fku = np.array(sFile["Fku"])

    sFile.close()


def addDataset(fileHandle, setName, setData):
    dset = 0
    if setName not in fileHandle:
        dset = fileHandle.create_dataset(setName, data = setData)

    return dset


def writeFFT(tVal):
    fileName = glob.dataDir + "output/FFT_{0:09.4f}.h5".format(tVal)

    print("\tWriting into file ", fileName)
    sFile = hp.File(fileName, 'a')

    dset = addDataset(sFile, "kShell", glob.kShell)

    if glob.varMode == 0:
        dset = addDataset(sFile, "ekx", glob.ekx)
        dset = addDataset(sFile, "ekz", glob.ekz)
        dset = addDataset(sFile, "Eku", glob.Eku)
        dset = addDataset(sFile, "Tku", glob.Tku)
        dset = addDataset(sFile, "Pku", glob.Pku)
        dset = addDataset(sFile, "Fku", glob.Fku)
        dset = addDataset(sFile, "Dku", glob.Dku)

        dset = addDataset(sFile, "EkT", glob.EkT)
        dset = addDataset(sFile, "TkT", glob.TkT)
        dset = addDataset(sFile, "PkT", glob.PkT)
        dset = addDataset(sFile, "DkT", glob.DkT)
    elif glob.varMode == 1:
        dset = addDataset(sFile, "ekx", glob.ekx)
        dset = addDataset(sFile, "ekz", glob.ekz)
        dset = addDataset(sFile, "Eku", glob.Eku)
        dset = addDataset(sFile, "Tku", glob.Tku)
        dset = addDataset(sFile, "Pku", glob.Pku)
        dset = addDataset(sFile, "Dku", glob.Dku)
    elif glob.varMode == 2:
        dset = addDataset(sFile, "EkT", glob.EkT)
        dset = addDataset(sFile, "TkT", glob.TkT)
        dset = addDataset(sFile, "PkT", glob.PkT)
        dset = addDataset(sFile, "DkT", glob.DkT)
        dset = addDataset(sFile, "Fku", glob.Fku)

    sFile.close()


def main():
    # Set some global variables from CLI arguments
    argList = sys.argv[1:]
    if argList and len(argList) == 3:
        glob.varMode = int(argList[0])
        glob.startTime = float(argList[1])
        glob.stopTime = float(argList[2])

    # Load timelist
    tList = np.loadtxt(glob.dataDir + "output/timeList.dat", comments='#')

    for i in range(tList.shape[0]):
        tVal = tList[i]
        if tVal > glob.startTime and tVal < glob.stopTime:
            if glob.readFile:
                readFFT(tVal)

            else:
                fileName = glob.dataDir + "output/Soln_{0:09.4f}.h5".format(tVal)
                loadData(fileName)

                # Compute non-linear terms
                if glob.cmpTrn and glob.realNLin:
                    print("\tComputing non-linear term")

                    if glob.varMode == 0:
                        glob.nlx, glob.nlz, glob.nlT = nlin.computeNLin()
                        periodicBC(glob.nlx)
                        periodicBC(glob.nlz)
                        periodicBC(glob.nlT)
                    elif glob.varMode == 1:
                        glob.nlx, glob.nlz = nlin.computeNLin()
                        periodicBC(glob.nlx)
                        periodicBC(glob.nlz)
                    elif glob.varMode == 2:
                        glob.nlT = nlin.computeNLin()
                        periodicBC(glob.nlT)
                else:
                    glob.nlx, glob.nlz, glob.nlT = 0, 0, 0

                # Interpolate data to uniform grid
                print("\tInterpolating to uniform grid")
                uniformInterp()

                print("\tComputing FFT")
                if glob.realNLin:
                    fft.computeFFT(0, 0, 0, 0, 0)
                else:
                    fft.computeFFT(glob.U*glob.U, glob.U*glob.W, glob.W*glob.W, glob.U*glob.T, glob.W*glob.T)

                if glob.cmpTrn:
                    glob.Pku[0] = -glob.Tku[0]
                    glob.Pku[1:] = -np.cumsum(glob.Tku[1:]*glob.dk, axis=0) + glob.Pku[0]

                    glob.PkT[0] = -glob.TkT[0]
                    glob.PkT[1:] = -np.cumsum(glob.TkT[1:]*glob.dk, axis=0) + glob.PkT[0]

                writeFFT(tVal)

                if glob.varMode < 2:
                    print("\tChecking energy balance")
                    energyCheck()

    plt.plotStuff([[0, 1], [4, 5]])

main()
