import h5py as hp
import numpy as np
import computeFFT as fft
import matplotlib as mpl
import globalData as glob
from scipy import interpolate
import matplotlib.pyplot as plt
from globalData import Nx, Nz
import scipy.integrate as integrate

mpl.style.use('classic')

# Pyplot-specific directives
plt.rcParams["font.family"] = "serif"

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


def imposeBCs():
    # Periodic along X
    glob.X[0], glob.X[-1] = -glob.X[1], glob.Lx + glob.X[1]
    glob.U[0,:], glob.U[-1,:] = glob.U[-2,:], glob.U[1,:]
    glob.W[0,:], glob.W[-1,:] = glob.W[-2,:], glob.W[1,:]
    glob.T[0,:], glob.T[-1,:] = glob.T[-2,:], glob.T[1,:]

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

    glob.X = xU
    glob.Z = zU


def energyCheck(Ek):
    ke = (glob.U**2 + glob.W**2)/2.0
    keInt = integrate.simps(integrate.simps(ke, glob.Z), glob.X)/glob.tVol
    print("\t\tReal field energy =     {0:10.8f}".format(keInt))

    keInt = np.sum(np.dot(Ek[1:], glob.dk)) + Ek[0]
    print("\t\tShell spectrum energy = {0:10.8f}".format(keInt))


def readFFT(tVal):
    fileName = glob.inputDir + "FFT_{0:09.4f}.h5".format(tVal)

    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    kShell = np.array(sFile["kShell"])
    Ek = np.array(sFile["Ek"])

    sFile.close()

    return kShell, Ek


def writeFFT(tVal, Ek):
    fileName = glob.inputDir + "FFT_{0:09.4f}.h5".format(tVal)

    print("\tWriting into file ", fileName)
    sFile = hp.File(fileName, 'w')

    dset = sFile.create_dataset("kShell", data = glob.kShell)
    dset = sFile.create_dataset("Ek", data = Ek)

    sFile.close()


def main():
    # Load timelist
    tList = np.loadtxt(glob.inputDir + "timeList.dat", comments='#')

    tList = [tList]
    for i in range(1):
    #for i in range(tList.shape[0]):
        tVal = tList[i]
        if glob.readFile:
            kShell, Ek = readFFT(tVal)

        else:
            fileName = glob.inputDir + "Soln_{0:09.4f}.h5".format(tVal)
            loadData(fileName)

            # Interpolate data to uniform grid
            print("\tInterpolating to uniform grid")
            uniformInterp()

            print("\tComputing FFT")
            Ek, Tk = fft.computeFFT(glob.U*glob.U)

            #writeFFT(tVal, Ek)

            #print("\tChecking energy balance")
            #energyCheck(Ek)

    Pk = np.zeros_like(Tk)
    Pk[0,:] = -Tk[0,:]
    #Pk[1:,:] = -np.cumsum(Tk[1:,:]*glob.dk, axis=0) + Pk[0,:]
    Pk[1:,:] = -np.cumsum(Tk[1:,:], axis=0) + Pk[0,:]

    #Ek = np.mean(Ek[:,:], axis=1)
    #Tk = np.mean(Tk[:,:], axis=1)
    #Pk = np.mean(Pk[:,:], axis=1)
    Ek = np.mean(Ek[:, 200:-200], axis=1)
    Tk = np.mean(Tk[:, 200:-200], axis=1)
    Pk = np.mean(Pk[:, 200:-200], axis=1)

    showPlot = 3
    if showPlot == 1:
        plt.loglog(glob.kShell, Ek)
        plt.ylabel("E(k)")
        plt.xlabel("k")
        plt.show()
    elif showPlot == 2:
        plt.plot(glob.kShell, Tk)
        plt.xscale("log")
        plt.yscale("symlog", linthreshy=1e-10)
        plt.ylabel("T(k)")
        #plt.ylim(-5e-3, 5e-3)
        #plt.yticks([-5e-3, 0, 5e-3])
        plt.xlabel("k")
        plt.show()
    elif showPlot == 3:
        plt.plot(glob.kShell, Pk)
        plt.xscale("log")
        plt.yscale("symlog", linthreshy=1e-10)
        plt.ylabel(r"$\Pi(k)$")
        #plt.ylim(-5e-3, 5e-3)
        #plt.yticks([-5e-3, 0, 5e-3])
        plt.xlabel("k")
        plt.show()
    elif showPlot == 4:
        plt.plot(glob.kShell, Pk)
        plt.xscale("log")
        plt.yscale("symlog", linthreshy=1e-10)
        plt.ylabel(r"$\Pi(k)$")
        #plt.ylim(-5e-3, 5e-3)
        #plt.yticks([-5e-3, 0, 5e-3])
        plt.xlabel("k")
        plt.show()

main()
