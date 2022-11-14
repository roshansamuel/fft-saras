import derivCalc as df
import globalData as glob

def nlinTerm(F):
    if glob.consNLin:
        return df.dfx(glob.U*F) + df.dfz(glob.W*F)
    else:
        return glob.U*df.dfx(F) + glob.W*df.dfz(F)

def computeNLin():
    return nlinTerm(glob.U), nlinTerm(glob.W), nlinTerm(glob.T)
