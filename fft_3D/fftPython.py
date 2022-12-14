import h5py as hp
import numpy as np
import nlinCalc as nlin
import computeFFT as fft
import matplotlib as mpl
import globalData as glob
from scipy import interpolate
import matplotlib.pyplot as plt
from globalData import Nx, Ny, Nz
import scipy.integrate as integrate

mpl.style.use('classic')

# Pyplot-specific directives
plt.rcParams["font.family"] = "serif"

print()

def loadData(fileName):
    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    glob.U = np.pad(np.array(sFile["Vx"]), 1)
    glob.V = np.pad(np.array(sFile["Vy"]), 1)
    glob.W = np.pad(np.array(sFile["Vz"]), 1)
    glob.T = np.pad(np.array(sFile["T"]), 1)

    glob.X = np.pad(np.array(sFile["X"]), (1, 1), 'constant', constant_values=(0, glob.Lx))
    glob.Y = np.pad(np.array(sFile["Y"]), (1, 1), 'constant', constant_values=(0, glob.Ly))
    glob.Z = np.pad(np.array(sFile["Z"]), (1, 1), 'constant', constant_values=(0, glob.Lz))

    sFile.close()

    imposeBCs()


def imposeBCs():
    # Periodic along X
    glob.X[0], glob.X[-1] = -glob.X[1], glob.Lx + glob.X[1]
    glob.U[0,:,:], glob.U[-1,:,:] = glob.U[-2,:,:], glob.U[1,:,:]
    glob.V[0,:,:], glob.V[-1,:,:] = glob.V[-2,:,:], glob.V[1,:,:]
    glob.W[0,:,:], glob.W[-1,:,:] = glob.W[-2,:,:], glob.W[1,:,:]
    glob.T[0,:,:], glob.T[-1,:,:] = glob.T[-2,:,:], glob.T[1,:,:]

    # Periodic along Y
    glob.Y[0], glob.Y[-1] = -glob.Y[1], glob.Ly + glob.Y[1]
    glob.U[:,0,:], glob.U[:,-1,:] = glob.U[:,-2,:], glob.U[:,1,:]
    glob.V[:,0,:], glob.V[:,-1,:] = glob.V[:,-2,:], glob.V[:,1,:]
    glob.W[:,0,:], glob.W[:,-1,:] = glob.W[:,-2,:], glob.W[:,1,:]
    glob.T[:,0,:], glob.T[:,-1,:] = glob.T[:,-2,:], glob.T[:,1,:]

    # RBC
    glob.T[:,:,0], glob.T[:,:,-1] = 1.0, 0.0


def interpolateData(f, xO, yO, zO):
    intFunct = interpolate.interp1d(glob.Z, f, kind='cubic', axis=2)
    f = intFunct(zO)
    intFunct = interpolate.interp1d(glob.Y, f, kind='cubic', axis=1)
    f = intFunct(yO)
    intFunct = interpolate.interp1d(glob.X, f, kind='cubic', axis=0)
    f = intFunct(xO)

    return f


def uniformInterp():
    xU = np.linspace(0.0, glob.Lx, Nx, endpoint=False)
    yU = np.linspace(0.0, glob.Ly, Ny, endpoint=False)
    zU = np.linspace(0.0, glob.Lz, Nz, endpoint=False)

    glob.U = interpolateData(glob.U, xU, yU, zU)
    glob.V = interpolateData(glob.V, xU, yU, zU)
    glob.W = interpolateData(glob.W, xU, yU, zU)
    glob.T = interpolateData(glob.T, xU, yU, zU)

    if glob.cmpTrn:
        glob.nlx = interpolateData(glob.nlx, xU, yU, zU)
        glob.nly = interpolateData(glob.nly, xU, yU, zU)
        glob.nlz = interpolateData(glob.nlz, xU, yU, zU)

    glob.X = xU
    glob.Y = yU
    glob.Z = zU


def energyCheck(Ek):
    ke = (glob.U**2 + glob.V**2 + glob.W**2)/2.0
    keInt = integrate.simps(integrate.simps(integrate.simps(ke, glob.Z), glob.Y), glob.X)/glob.tVol
    print("\t\tReal field energy =     {0:10.8f}".format(keInt))

    keInt = np.sum(np.dot(Ek[1:], glob.dk)) + Ek[0]
    print("\t\tShell spectrum energy = {0:10.8f}".format(keInt))


def readFFT(tVal):
    fileName = glob.inputDir + "FFT_{0:09.4f}.h5".format(tVal)

    print("\nReading from file ", fileName)
    sFile = hp.File(fileName, 'r')

    kShell = np.array(sFile["kShell"])
    ekx = np.array(sFile["ekx"])
    eky = np.array(sFile["eky"])
    ekz = np.array(sFile["ekz"])
    if glob.cmpTrn:
        Tk = np.array(sFile["Tk"])
        Pk = np.array(sFile["Pk"])
    else:
        Tk = np.zeros_like(ekx)
        Pk = np.zeros_like(ekx)

    sFile.close()

    return kShell, ekx, eky, ekz, Tk, Pk


def writeFFT(tVal, ekx, eky, ekz, Tk, Pk):
    fileName = glob.inputDir + "FFT_{0:09.4f}.h5".format(tVal)

    print("\tWriting into file ", fileName)
    sFile = hp.File(fileName, 'w')

    dset = sFile.create_dataset("kShell", data = glob.kShell)
    dset = sFile.create_dataset("ekx", data = ekx)
    dset = sFile.create_dataset("eky", data = eky)
    dset = sFile.create_dataset("ekz", data = ekz)

    if glob.cmpTrn:
        dset = sFile.create_dataset("Tk", data = Tk)
        dset = sFile.create_dataset("Pk", data = Pk)

    sFile.close()


def main():
    # Load timelist
    tList = np.loadtxt(glob.inputDir + "timeList.dat", comments='#')

    tList = [tList]
    for i in range(1):
    #for i in range(tList.shape[0]):
        tVal = tList[i]
        if glob.readFile:
            kShell, ekx, eky, ekz, Tk, Pk = readFFT(tVal)

            Ek = ekx + eky + ekz

        else:
            fileName = glob.inputDir + "Soln_{0:09.4f}.h5".format(tVal)
            loadData(fileName)

            # Compute non-linear terms
            if glob.cmpTrn and glob.realNLin:
                print("\tComputing non-linear term")
                glob.nlx, glob.nly, glob.nlz = nlin.computeNLin()
            else:
                glob.nlx, glob.nly, glob.nlz = 0, 0, 0

            # Interpolate data to uniform grid
            print("\tInterpolating to uniform grid")
            uniformInterp()

            print("\tComputing FFT")
            if glob.realNLin:
                ekx, eky, ekz, Tkx, Tky, Tkz = fft.computeFFT(0, 0, 0, 0, 0, 0)
            else:
                ekx, eky, ekz, Tkx, Tky, Tkz = fft.computeFFT(glob.U*glob.U, glob.V*glob.V, glob.W*glob.W,
                                                              glob.U*glob.V, glob.V*glob.W, glob.W*glob.U)

            Ek = ekx + eky + ekz
            if glob.cmpTrn:
                Tk = Tkx + Tky + Tkz

                Pk = np.zeros_like(Tk)
                Pk[0] = -Tk[0]
                Pk[1:] = -np.cumsum(Tk[1:]*glob.dk, axis=0) + Pk[0]
            else:
                Tk = np.zeros_like(Ek)
                Pk = np.zeros_like(Ek)

            writeFFT(tVal, ekx, eky, ekz, Tk, Pk)

            print("\tChecking energy balance")
            energyCheck(Ek)

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


main()
