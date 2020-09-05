from scipy import integrate, optimize
import matplotlib.pyplot as plt
import numpy as np
import random
import math

if __name__ == "__main__":
    ALPHA = 0.25
    p_alpha = 0.75
    N = 25000

    def SEIRM(y, x, alpha, alpha_r, beta, gamma_r, gamma_u, mu):
        alpha_u = 1.0*(1-alpha)/alpha * alpha_r
        
        dS = -1.0*beta*y[0]*(y[2]+y[3])/N
        dE = beta*y[0]*(y[2]+y[3])/N - alpha_r*y[1] - alpha_u*y[1]
        dI_r = alpha_r*y[1] - gamma_r*y[2] - mu*y[2]
        dI_u = alpha_u*y[1] - gamma_u*y[3]
        dR = gamma_r*y[2] + gamma_u*y[3]
        dM = mu*y[2]
        
        return dS, dE, dI_r, dI_u, dR, dM

    def FitSEIRM(x, alpha, alpha_r, beta, gamma_r, gamma_u, mu, E0):        
        S0 = N - E0
        
        ret =  integrate.odeint(func = SEIRM, y0 = (S0, E0, 0, 0, 0 ,0), t = x, args = (alpha, alpha_r, beta, gamma_r, gamma_u, mu))

        I = N - ret[:,0] - ret[:,1]
        Ir = alpha*np.gradient(I)
        M = ret[:,5]
        M = np.gradient(M)
        IM = np.hstack((Ir,M))

        return IM

    def FitSEIRM_p(x, alpha_r, beta, gamma_r, gamma_u, mu, E0):   
        S0 = N - E0
        
        ret =  integrate.odeint(func = SEIRM, y0 = (S0, E0, 0, 0, 0 ,0), t = x, args = (p_alpha, alpha_r, beta, gamma_r, gamma_u, mu))

        I = N - ret[:,0] - ret[:,1]
        Ir = p_alpha*np.gradient(I)
        M = ret[:,5]
        M = np.gradient(M)
        IM = np.hstack((Ir,M))

        return IM

    def FitSEIRM_pp(x, alpha, alpha_r, beta, gamma_r, gamma_u, mu, E0): 
        S0 = N - E0
        
        ret =  integrate.odeint(func = SEIRM, y0 = (S0, E0, 0, 0, 0 ,0), t = x, args = (alpha, alpha_r, beta, gamma_r, gamma_u, mu))
        
        I = N - ret[:,0] - ret[:,1]
        I = np.gradient(I)
        Ir = alpha*I
        M = ret[:,5]
        M = np.gradient(M)
        IM = np.hstack((Ir,I,M))

        return IM

    XData = list(range(1, 90+1))
    XData = np.array(XData, dtype = float)
    
    Data = integrate.odeint(func = SEIRM, y0 = (N-10, 10, 0, 0, 0 ,0), t = XData, args = (ALPHA, 0.1, 0.36, 0.095, 0.2, 0.005))

    '''    
    figure = plt.figure(figsize = (10.24, 7.68))
    plt.plot(Data[:,0], label = 'S')
    plt.plot(Data[:,1], label = 'E')
    plt.plot(Data[:,2], label = 'I_r')
    plt.plot(Data[:,3], label = 'I_u')
    plt.plot(Data[:,4], label = 'R')
    plt.plot(Data[:,5], label = 'M')
    plt.legend()
    plt.show()
    '''
    
    YCases = N - Data[:,0] - Data[:,1]
    YCases = np.gradient(YCases)
    YReports = ALPHA * YCases
    YDeaths = Data[:,5]
    YDeaths = np.gradient(YDeaths)

    YDataReport = np.hstack((YReports, YDeaths))
    YDataReport = np.array(YDataReport, dtype = float)
    YData = np.hstack((YReports, YCases, YDeaths))
    YData = np.array(YData, dtype = float)

    ParaOpt_p, ParaCov_p = optimize.curve_fit(f = FitSEIRM_p, xdata = XData, ydata = YDataReport, maxfev = 10000, p0 = (1, 1, 1, 1, 1, 10), bounds = [[0,0,0,0,0,0], [1,1,1,1,1,N]])
    
    print('p:')
    print('report rate:',p_alpha)
    print('alpha_reported:',ParaOpt_p[0])
    print('alpha_unreported:',ParaOpt_p[0]*(1-p_alpha)/p_alpha)
    print('beta:',ParaOpt_p[1])
    print('gamma_reported:',ParaOpt_p[2])
    print('gamma_unreported:',ParaOpt_p[3])
    print('mu:',ParaOpt_p[4])
    print('S0:',N-ParaOpt_p[5])
    print('E0:',ParaOpt_p[5])
    print('Iu0:',0)
    print('Ir0:',0)
    print('R0:',0)
    print('M0:',0)

    Parameter_p = [p_alpha, ParaOpt_p[0], ParaOpt_p[1], ParaOpt_p[2], ParaOpt_p[3], ParaOpt_p[4], ParaOpt_p[5], N - ParaOpt_p[5], 0, 0, 0, 0]

    Result_p = FitSEIRM(XData, *Parameter_p[0:7])

    figure = plt.figure(figsize = (15.36,10.24))
    plt.subplot(2,3,1)
    plt.plot(XData, YReports, label='D_reported')
    plt.plot(XData, Result_p[:len(YCases)], label='D_OS_reported(p)')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p -> Calibrate(OS, D_reported)")
    plt.legend()
    
    plt.subplot(2,3,2)
    plt.plot(XData, YDeaths, label='D_mortality')
    plt.plot(XData, Result_p[len(YCases):], label='D_OS_mortality(p)')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p -> Calibrate(OS, D_reported)")
    plt.legend()
    
    plt.subplot(2,3,3)
    plt.plot(XData, YCases, label='D_ground_truth')
    plt.plot(XData, Result_p[:len(YCases)]/Parameter_p[0], label='D_OS(p)')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p -> Calibrate(OS, D_reported)")
    plt.legend()
    
    ParaOpt_pp, ParaCov_pp = optimize.curve_fit(f = FitSEIRM_pp, xdata = XData, ydata = YData, maxfev = 10000, p0 = (1, 1, 1, 1, 1, 1, 10), bounds = [[0,0,0,0,0,0,0], [1,1,1,1,1,1,N]])

    print('p\':')
    print('report rate:',ParaOpt_pp[0])
    print('alpha_reported:',ParaOpt_pp[1])
    print('alpha_unreported:',ParaOpt_pp[1]*(1-ParaOpt_pp[0])/ParaOpt_pp[0])
    print('beta:',ParaOpt_pp[2])
    print('gamma_reported:',ParaOpt_pp[3])
    print('gamma_unreported:',ParaOpt_pp[4])
    print('mu:',ParaOpt_pp[5])
    print('S0:',N-ParaOpt_pp[6])
    print('E0:',ParaOpt_pp[6])
    print('Iu0:',0)
    print('Ir0:',0)
    print('R0:',0)
    print('M0:',0)

    Parameter_pp = [ParaOpt_pp[0], ParaOpt_pp[1], ParaOpt_pp[2], ParaOpt_pp[3], ParaOpt_pp[4], ParaOpt_pp[5], ParaOpt_pp[6], N - ParaOpt_pp[6], 0, 0, 0, 0]

    Result_pp = FitSEIRM_pp(XData, *Parameter_pp[0:7])

    plt.subplot(2,3,4)
    plt.plot(XData, YReports, label='D_reported')
    plt.plot(XData, Result_pp[:len(YCases)], label='D_OS_reported(p\')')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p' -> Calibrate(OS, D_reported, D_unreported)")
    plt.legend()
    
    plt.subplot(2,3,5)
    plt.plot(XData, YDeaths, label='D_mortality')
    plt.plot(XData, Result_pp[2*len(YCases):], label='D_OS_mortality(p\')')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p' -> Calibrate(OS, D_reported, D_unreported)")
    plt.legend()
    
    plt.subplot(2,3,6)
    plt.plot(XData, YCases, label='D_ground_truth')
    plt.plot(XData, Result_pp[len(YCases):2*len(YCases)], label='D_OS(p\')')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p' -> Calibrate(OS, D_reported, D_unreported)")
    plt.legend()

    plt.savefig("result.png")
    plt.show()
    
    print (np.linalg.norm(np.cumsum(YReports)-np.cumsum(Result_p[:len(YCases)])))
    print (np.linalg.norm(np.cumsum(YCases)-np.cumsum(Result_p[:len(YCases)])/Parameter_p[0]))
    print (np.linalg.norm(np.cumsum(YReports)-np.cumsum((Result_pp[:len(YCases)]))))
    print (np.linalg.norm(np.cumsum(YCases)-np.cumsum(Result_pp[len(YCases):2*len(YCases)])))
    
    def IntegerCost(x):
        Sum = math.log(2.865,2)+1
        if (abs(x) > 1.01):
            add = math.log(abs(x),2)
            while (add > 1.01):
                Sum = Sum + add
                add = math.log(add,2)
        return Sum

    def RealNumberCost(x):
        delta = 0.1
        if (math.floor(abs(x)) == 0):
            return 1-math.log(delta,2) 
        else:
            return IntegerCost(math.floor(abs(x)))-math.log(delta,2)+1 

    def VectorCost(x):
        Sum = 0
        for xnow in x:
            Sum = Sum + RealNumberCost(xnow)
        return Sum
    
    def TimeSeriesCost(x):
        Sum = 0
        Length = x.shape[0]
        x_grad = np.around(np.gradient(x))
        T, Tcount = np.unique(x_grad, return_counts = True)
        XDictionary = {}
        for counter in range(len(T)):
            XDictionary[int(T[counter])] = int(Tcount[counter]) 
        for keys in XDictionary.keys():
            Sum = Sum + 1.0*XDictionary[keys]/Length*math.log(1.0*Length/XDictionary[keys],2)
        Sum = Length * Sum
        for keys in XDictionary.keys():
            Sum = Sum + IntegerCost(int(keys))
        return Sum

    Parameter_p = np.array(Parameter_p)
    Parameter_pp = np.array(Parameter_pp)

    ModelCost1 = VectorCost(Parameter_p)
    ModelCost2 = VectorCost(Parameter_p - Parameter_p)
    ModelCost3 = TimeSeriesCost(p_alpha*Result_p[:len(YCases)]/p_alpha-Result_p[:len(YCases)])
    ModelCost = ModelCost1 + ModelCost2 + ModelCost3
    DataCost = TimeSeriesCost((YReports/p_alpha)-(Result_p[:len(YCases)]/p_alpha))
    MDLCost = ModelCost + DataCost

    print ('MDL(D_jiaming(p))')
    print ('ModelCost1:',ModelCost1)
    print ('ModelCost2:',ModelCost2)
    print ('ModelCost3:',ModelCost3)
    print ('ModelCost:',ModelCost)
    print ('DataCost:',DataCost)
    print ('MDLCost:',MDLCost)

    ModelCost1 = VectorCost(Parameter_p)
    ModelCost2 = VectorCost(Parameter_pp - Parameter_p)
    ModelCost3 = TimeSeriesCost(Parameter_pp[0]*YCases-Result_p[:len(YCases)])
    ModelCost = ModelCost1 + ModelCost2 + ModelCost3
    DataCost = TimeSeriesCost((YReports/Parameter_pp[0])-(Result_pp[len(YCases):2*len(YCases)]))
    MDLCost = ModelCost + DataCost

    print ('MDL(D_ground_truth))')
    print ('ModelCost1:',ModelCost1)
    print ('ModelCost2:',ModelCost2)
    print ('ModelCost3:',ModelCost3)
    print ('ModelCost:',ModelCost)
    print ('DataCost:',DataCost)
    print ('MDLCost:',MDLCost)
