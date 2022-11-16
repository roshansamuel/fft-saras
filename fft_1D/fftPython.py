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


def readFFT(tVal):
    fileName = glob.dataDir + "output/FFT_{0:09.4f}.h5".format(tVal)

    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    kShell = np.array(sFile["kShell"])
    Ek = np.array(sFile["Ek"])

    sFile.close()

    return kShell, Ek


def writeFFT(tVal, Ek):
    fileName = glob.dataDir + "output/FFT_{0:09.4f}.h5".format(tVal)

    print("\tWriting into file ", fileName)
    sFile = hp.File(fileName, 'w')

    dset = sFile.create_dataset("kShell", data = glob.kShell)
    dset = sFile.create_dataset("Ek", data = Ek)

    sFile.close()


def main():
    # Load timelist
    tList = np.loadtxt(glob.dataDir + "output/timeList.dat", comments='#')

    for i in range(tList.shape[0]):
        tVal = tList[i]
        if glob.readFile:
            kShell, Ek = readFFT(tVal)

        else:
            fileName = glob.dataDir + "output/Soln_{0:09.4f}.h5".format(tVal)
            loadData(fileName)

            # Interpolate data to uniform grid
            print("\tInterpolating to uniform grid")
            uniformInterp()

            print("\tComputing FFT")
            Ek, Tk = fft.computeFFT(glob.U*glob.U)

            #writeFFT(tVal, Ek)

    Pk = np.zeros_like(Tk)
    Pk[0,:] = -Tk[0,:]
    #Pk[1:,:] = -np.cumsum(Tk[1:,:]*glob.dk, axis=0) + Pk[0,:]
    Pk[1:,:] = -np.cumsum(Tk[1:,:], axis=0) + Pk[0,:]

    #Ek = np.mean(Ek[:,:], axis=1)
    #Tk = np.mean(Tk[:,:], axis=1)
    #Pk = np.mean(Pk[:,:], axis=1)

    #Ek = np.mean(Ek[:, 200:-200], axis=1)
    #Tk = np.mean(Tk[:, 200:-200], axis=1)
    #Pk = np.mean(Pk[:, 200:-200], axis=1)

    #np.savetxt("out.dat", np.stack((glob.kShell, Ek, Tk, Pk), axis=1))
    showPlot = 4
    if showPlot == 1:
        Ek = np.mean(Ek[:, 200:-200], axis=1)
        Tk = np.mean(Tk[:, 200:-200], axis=1)
        Pk = np.mean(Pk[:, 200:-200], axis=1)

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
        figSize = (17, 7)
        fig = plt.figure(figsize=figSize)

        ax = fig.add_subplot(1, 2, 1)
        plt.loglog(glob.kShell, Ek[:, 1], label="Z Point = {0:3d}".format(1))

        A = Ek[2, 1]*(glob.kShell[2]**(5.0/3.0))
        inRangeStr = 1e1
        inRangeEnd = 5e2
        inRange = np.linspace(inRangeStr, inRangeEnd, 100)
        kolmLin = A*inRange**(-5.0/3.0)
        ax.loglog(inRange, kolmLin, linestyle='--', linewidth=2.5, color='black', label=r"$k^{-{11/5}}$")

        ax = fig.add_subplot(1, 2, 2)
        plt.loglog(glob.kShell, Ek[:, 50], label="Z Point = {0:3d}".format(50))

        A = 1.2
        inRangeStr = 1e1
        inRangeEnd = 5e2
        inRange = np.linspace(inRangeStr, inRangeEnd, 100)
        kolmLin = A*inRange**(-11.0/5.0)
        ax.loglog(inRange, kolmLin, linestyle='--', linewidth=2.5, color='black', label=r"$k^{-{11/5}}$")

        plt.ylabel("E(k)")
        plt.xlabel("k")
        plt.legend()
        plt.show()

main()
