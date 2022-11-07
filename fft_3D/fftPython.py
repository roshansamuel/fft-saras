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


def uniformInterp(nlx, nly, nlz):
    xU = np.linspace(0.0, glob.Lx, Nx)
    yU = np.linspace(0.0, glob.Ly, Ny)
    zU = np.linspace(0.0, glob.Lz, Nz)

    glob.U = interpolateData(glob.U, xU, yU, zU)
    glob.V = interpolateData(glob.V, xU, yU, zU)
    glob.W = interpolateData(glob.W, xU, yU, zU)
    glob.T = interpolateData(glob.T, xU, yU, zU)

    if glob.cmpTrn:
        nlx = interpolateData(nlx, xU, yU, zU)
        nly = interpolateData(nly, xU, yU, zU)
        nlz = interpolateData(nlz, xU, yU, zU)
    else:
        nlx, nly, nlz = 0, 0, 0

    glob.X = xU
    glob.Y = yU
    glob.Z = zU

    return nlx, nly, nlz


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
    Tkx = np.array(sFile["Tkx"])
    Tky = np.array(sFile["Tky"])
    Tkz = np.array(sFile["Tkz"])

    sFile.close()

    return kShell, ekx, eky, ekz, Tkx, Tky, Tkz


def writeFFT(tVal, ekx, eky, ekz, Tkx, Tky, Tkz):
    fileName = glob.inputDir + "FFT_{0:09.4f}.h5".format(tVal)

    print("\tWriting into file ", fileName)
    sFile = hp.File(fileName, 'w')

    dset = sFile.create_dataset("kShell", data = glob.kShell)
    dset = sFile.create_dataset("ekx", data = ekx)
    dset = sFile.create_dataset("eky", data = eky)
    dset = sFile.create_dataset("ekz", data = ekz)
    dset = sFile.create_dataset("Tkx", data = Tkx)
    dset = sFile.create_dataset("Tky", data = Tky)
    dset = sFile.create_dataset("Tkz", data = Tkz)

    sFile.close()


def main():
    # Load timelist
    tList = np.loadtxt(glob.inputDir + "timeList.dat", comments='#')

    tList = [tList]
    for i in range(1):
    #for i in range(tList.shape[0]):
        tVal = tList[i]
        if glob.readFile:
            kShell, ekx, eky, ekz, Tkx, Tky, Tkz = readFFT(tVal)

            Ek = ekx + eky + ekz
            if glob.cmpTrn:
                Tk = Tkx + Tky + Tkz

        else:
            fileName = glob.inputDir + "Soln_{0:09.4f}.h5".format(tVal)
            loadData(fileName)

            # Compute non-linear terms
            if glob.cmpTrn:
                print("\tComputing non-linear term")
                nlx, nly, nlz = nlin.computeNLin()
            else:
                nlx, nly, nlz = 0, 0, 0

            # Interpolate data to uniform grid
            print("\tInterpolating to uniform grid")
            nlx, nly, nlz = uniformInterp(nlx, nly, nlz)

            print("\tComputing FFT")
            ekx, eky, ekz, Tkx, Tky, Tkz = fft.computeFFT(nlx, nly, nlz)

            writeFFT(tVal, ekx, eky, ekz, Tkx, Tky, Tkz)

            Ek = ekx + eky + ekz
            if glob.cmpTrn:
                Tk = Tkx + Tky + Tkz

            print("\tChecking energy balance")
            energyCheck(Ek)

    #np.savetxt("Ek.dat", Ek)
    plt.loglog(glob.kShell, Ek)
    plt.ylabel("E(k)")
    plt.xlabel("k")
    #plt.savefig("plot.png")
    plt.show()


main()
