from numpy import exp
def nota(X, Csi,Xq, A,questoes=45,  db=dict()):
    def P_x_thetaCsi(theta,a,b,c,x):
        chave = str([theta, a, b, c])
        if chave not in db:
            db[chave] = [1-c-(1-c)/(1+exp(a*(theta-b))), c+(1-c)/(1+exp(a*(theta-b)))]
        return db[chave][x]
        return [1-c-(1-c)/(1+exp(a*(theta-b))), c+(1-c)/(1+ exp(a*(theta-b)))][x]
    def P_xvec_thetaCsi(x, theta, csi):
        p = 1
        for i in range(questoes):
            p*=P_x_thetaCsi(theta, csi[i][0], csi[i][1], csi[i][2], x[i])
        return p
    def aproxTheta(x, csi):
        n, d=0,0
        for i in range(40):
            n+=(Xq[i]+1)*500*P_xvec_thetaCsi(x, Xq[i],csi)*A[i]
            d+=P_xvec_thetaCsi(x, Xq[i],csi)*A[i] 
        return n/d
    return aproxTheta(X, Csi)