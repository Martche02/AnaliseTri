from numpy import exp, pi
from scipy.integrate import quad
def nota(X, Csi,Xq, A,questoes=45,  db=dict()):
    def P_x_thetaCsi(theta,a,b,c,x):
        return [1-(c+(1-c)*(exp(-a*(theta-b)))/(1+ exp(-a*(theta-b))) ), c+(1-c)*(exp(-a*(theta-b)))/(1+ exp(-a*(theta-b)))][x]
        chave = str([theta, a, b, c])
        if chave not in db:
            db[chave] = [1-c-(1-c)*(exp(a*(theta-b)))/(1+exp(a*(theta-b))), c+(1-c)*(exp(a*(theta-b)))/(1+ exp(a*(theta-b)))]
        return db[chave][x]
    def P_xvec_thetaCsi(x, theta, csi):
        p = 1
        for i in range(questoes):
            p*=P_x_thetaCsi(theta, csi[i][0], csi[i][1], csi[i][2], x[i])
        return p
    def aproxTheta(x, csi):
        n, d=0,0
        for i in range(40):
            n+=Xq[i]*P_xvec_thetaCsi(x, Xq[i],csi)*A[i]#*exp(-0.5*((Xq[i]))**2)
            d+=P_xvec_thetaCsi(x, Xq[i],csi)*A[i]#*exp(-0.5*((Xq[i]))**2)
        return n/d
    return aproxTheta(X, Csi)
    def integrandoN(theta, x, csi):
        return theta*P_xvec_thetaCsi(x, theta, csi)*exp(-0.5*((theta)/1)**2)
    def integrandoD(theta, x, csi):
        return P_xvec_thetaCsi(x, theta, csi)*exp(-0.5*((theta)/1)**2)
    return quad(integrandoN, -1, 1,(X, Csi),points=40)[0]/quad(integrandoD, -1, 1,(X, Csi),points=40)[0]
    