import derivCalc as df
import globalData as glob

def nlinTerm(F):
    return glob.U*df.dfx(F) + glob.V*df.dfy(F) + glob.W*df.dfz(F)

def computeNLin():
    return nlinTerm(glob.U), nlinTerm(glob.V), nlinTerm(glob.W)
