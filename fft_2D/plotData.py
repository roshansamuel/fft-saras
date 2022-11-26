import numpy as np
import matplotlib as mpl
import globalData as glob
import matplotlib.pyplot as plt

mpl.style.use('classic')

# Pyplot-specific directives
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 22


def plotUSpectrum(ax):
    ax.loglog(glob.kShell, glob.Eku)
    ax.set_ylabel(r"$E_{u}(k)$")
    ax.set_xlabel(r"$k$")

def plotTSpectrum(ax):
    ax.loglog(glob.kShell, glob.EkT)
    ax.set_ylabel(r"$E_{\theta}(k)$")
    ax.set_xlabel(r"$k$")

def plotUTransfer(ax):
    ax.plot(glob.kShell, glob.Tku)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthreshy=1e-10)

    ax.set_ylabel(r"$T(k)$")
    ax.set_xlabel(r"$k$")

    #ax.set_ylim(-5e-3, 5e-3)
    #ax.set_yticks([-5e-3, 0, 5e-3])

def plotTTransfer(ax):
    ax.plot(glob.kShell, glob.TkT)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthreshy=1e-10)

    ax.set_ylabel(r"$T(k)$")
    ax.set_xlabel(r"$k$")

    #ax.set_ylim(-5e-3, 5e-3)
    #ax.set_yticks([-5e-3, 0, 5e-3])

def plotUFlux(ax):
    ax.plot(glob.kShell, DTk)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthreshy=1e-5)

    ax.set_ylabel(r"$\Pi(k)$")
    ax.set_xlabel("k")

    #ax.set_ylim(-5e-3, 5e-3)
    #ax.set_yticks([-5e-3, 0, 5e-3])

def plotStuff(figSize, stList):
    fig = plt.figure(figsize=figSize)
    ax = fig.add_subplot(1, 1, 1)
    plotTSpectrum(ax)

    ax.tick_params(axis='both', labelsize=24)
    plt.show()
