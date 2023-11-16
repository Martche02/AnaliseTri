import math
import numpy
def nota(X, Csi, questoes=45, pontos_de_quadratura=40):
    def P_x_thetaCsi(theta,a,b,c,x):
        p =[1-c-(1-c)/(1+math.exp(a*(theta-b))), c+(1-c)/(1+math.exp(a*(theta-b)))][x]
        if p<0:
            print(p,theta,a,b,c,x)
        return p
    def P_xvec_thetaCsi(x, theta, csi):
        p = 1
        for i in range(questoes):
            p*=P_x_thetaCsi(theta, csi[i][0], csi[i][1], csi[i][2], x[i])
        return p
    def aproxTheta(x, csi):
        n, d=0,0
        Xq, A = numpy.polynomial.legendre.leggauss(pontos_de_quadratura)
        Xq = [Xq[i] for i in range(len(Xq))]
        for i in range(pontos_de_quadratura):
            n+=(Xq[i]+1)*500*P_xvec_thetaCsi(x, Xq[i],csi)*A[i]
            d+=P_xvec_thetaCsi(x, Xq[i],csi)*A[i] 

        return n/d
    return aproxTheta(X, Csi)