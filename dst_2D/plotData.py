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

    ax.set_ylim(1e-15, 1e-1)
    ax.set_ylabel(r"$E_{u}(k)$")
    ax.set_xlabel(r"$k$")

def plotTSpectrum(ax):
    ax.loglog(glob.kShell, glob.EkT)

    ax.set_ylim(1e-15, 1e-2)
    ax.set_ylabel(r"$E_{\theta}(k)$")
    ax.set_xlabel(r"$k$")

def plotUTransfer(ax):
    ax.plot(glob.kShell, glob.Tku)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthreshy=1e-4)

    ax.set_ylabel(r"$T(k)$")
    ax.set_xlabel(r"$k$")

    #ax.set_ylim(-5e-3, 5e-3)
    #ax.set_yticks([-5e-3, 0, 5e-3])

def plotTTransfer(ax):
    ax.plot(glob.kShell, glob.TkT)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthreshy=1e-4)

    ax.set_ylabel(r"$T_{\theta}(k)$")
    ax.set_xlabel(r"$k$")

    #ax.set_ylim(-5e-3, 5e-3)
    #ax.set_yticks([-5e-3, 0, 5e-3])

def plotUFlux(ax):
    ax.plot(glob.kShell, glob.Pku)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthreshy=1e-4)

    ax.set_ylabel(r"$\Pi(k)$")
    ax.set_xlabel("k")

    ax.set_ylim(-1e-2, 1e-2)
    #ax.set_yticks([-5e-3, 0, 5e-3])

def plotTFlux(ax):
    ax.plot(glob.kShell, glob.PkT)
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthreshy=1e-4)

    ax.set_ylabel(r"$\Pi_{\theta}(k)$")
    ax.set_xlabel("k")

    ax.set_ylim(0, 1e-2)
    #ax.set_yticks([-5e-3, 0, 5e-3])

def plotStuff(stList):
    nrow = len(stList)
    ncol = len(stList[0])

    nPlots = nrow*ncol
    figSize = (ncol*7, nrow*5)
    fig = plt.figure(figsize=figSize)

    count = 0
    axList = []
    for i in range(nrow):
        for j in range(ncol):
            axList.append(fig.add_subplot(nrow, ncol, count+1))
            fv = stList[i][j]
            if fv == 0:
                plotUSpectrum(axList[count])
            elif fv == 1:
                plotTSpectrum(axList[count])
            elif fv == 2:
                plotUTransfer(axList[count])
            elif fv == 3:
                plotTTransfer(axList[count])
            elif fv == 4:
                plotUFlux(axList[count])
            elif fv == 5:
                plotTFlux(axList[count])
            
            count += 1

    for ax in axList:
        ax.tick_params(axis='both', labelsize=24)

    plt.tight_layout()
    #plt.savefig("plot.png")
    plt.show()
