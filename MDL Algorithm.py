from scipy import integrate, optimize
import matplotlib.pyplot as plt
import numpy as np
import random
import math

if __name__ == "__main__":
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
        M = 10*np.gradient(M) # Mortality Calibration Weight
        IM = np.hstack((Ir,M))

        return IM

    def FitSEIRM_p(x, alpha_r, beta, gamma_r, gamma_u, mu, E0):
        p_alpha = 0.75
        
        S0 = N - E0
        
        ret =  integrate.odeint(func = SEIRM, y0 = (S0, E0, 0, 0, 0 ,0), t = x, args = (p_alpha, alpha_r, beta, gamma_r, gamma_u, mu))

        I = N - ret[:,0] - ret[:,1]
        Ir = p_alpha*np.gradient(I)
        M = ret[:,5]
        M = 10*np.gradient(M) # Mortality Calibration Weight
        IM = np.hstack((Ir,M))

        return IM

    def FitSEIRM_pp(x, alpha, alpha_r, beta, gamma_r, gamma_u, mu, E0): 
        S0 = N - E0
        
        ret =  integrate.odeint(func = SEIRM, y0 = (S0, E0, 0, 0, 0 ,0), t = x, args = (alpha, alpha_r, beta, gamma_r, gamma_u, mu))
        
        I = N - ret[:,0] - ret[:,1]
        I = np.gradient(I)
        Ir = alpha*I
        Iu = (1-alpha)*I
        M = ret[:,5]
        M = 10*np.gradient(M) # Mortality Calibration Weight
        IM = np.hstack((Ir,Iu,M))

        return IM

    XData = list(range(1, 90+1))
    XData = np.array(XData, dtype = float)
    
    Data = integrate.odeint(func = SEIRM, y0 = (N-10, 10, 0, 0, 0 ,0), t = XData, args = (0.25, 0.1, 0.36, 0.095, 0.2, 0.005)) # Ground truth alpha
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
    YReports = 0.25 * YCases # Ground truth alpha
    YUnreports = (1-0.25) * YCases # Ground truth alpha
    YDeaths = Data[:,5]
    YDeaths = 10*np.gradient(YDeaths) # Mortality Calibration Weight

    YData = np.hstack((YReports, YDeaths))
    YData = np.array(YData, dtype = float)

    ParaOpt_p, ParaCov_p = optimize.curve_fit(f = FitSEIRM_p, xdata = XData, ydata = YData, maxfev = 10000, p0 = (1, 1, 1, 1, 1, 10), bounds = [[0,0,0,0,0,0], [1,1,1,1,1,N]])
    '''
    print('p:')
    print('report rate:',0.75)
    print('alpha_reported:',ParaOpt_p[0])
    print('alpha_unreported:',ParaOpt_p[0]*(1-0.75)/0.75)
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
    '''
    Parameter_p = [0.75, ParaOpt_p[0], ParaOpt_p[1], ParaOpt_p[2], ParaOpt_p[3], ParaOpt_p[4], ParaOpt_p[5], N - ParaOpt_p[5], 0, 0, 0, 0]

    Result_p = FitSEIRM(XData, *Parameter_p[0:7])
    '''
    figure = plt.figure(figsize = (15.36,3.84))
    plt.subplot(1,3,1)
    plt.plot(XData, YReports, label='D_reported')
    plt.plot(XData, Result_p[:len(YCases)], label='D_OS_reported(p)')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p -> Calibrate(OS, D_reported)")
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(XData, YDeaths, label='D_mortality')
    plt.plot(XData, Result_p[len(YCases):], label='D_OS_mortality(p)')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p -> Calibrate(OS, D_reported)")
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(XData, YCases, label='D_ground_truth')
    plt.plot(XData, Result_p[:len(YCases)]/Parameter_p[0], label='D_OS(p)')
    plt.xlabel('Time')
    plt.ylabel('Cases')
    plt.title("p -> Calibrate(OS, D_reported)")
    plt.legend()
    plt.show()
    '''
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

    def MDL(D, Parameter_p, Parameter_pp):
        Result_pp = FitSEIRM_pp(XData, *Parameter_pp[0:7])
        
        p = np.array(Parameter_p)
        pp = np.array(Parameter_pp)

        ModelCost1 = VectorCost(p)
        ModelCost2 = VectorCost(pp - p)
        ModelCost3 = TimeSeriesCost(pp[0]*D-Result_p[:len(YCases)])
        ModelCost = ModelCost1 + ModelCost2 + ModelCost3
        DataCost = TimeSeriesCost((D-YReports)/(1-Parameter_pp[0])-(Result_pp[:len(YCases)]+Result_pp[len(YCases):2*len(YCases)]))
        MDLCost = ModelCost + DataCost

        return MDLCost

    AlphaList = []
    MDLCostList = []
    MDLCostForwardList = []

    D_alpha = 0.01
    Parameter_pp = Parameter_p[:]
            
    while (True):        
        print (D_alpha)
        D_alpha_new = D_alpha + 0.01
        D_new1 = YReports / D_alpha_new
        D_new2 = YReports / (D_alpha_new + 0.01)
        D_new3 = YReports / (D_alpha_new + 0.02)

        D = YReports / D_alpha
        YDataMore = np.hstack((YReports, D-YReports, YDeaths))
        ParaOpt_pp, ParaCov_pp = optimize.curve_fit(f = FitSEIRM_pp, xdata = XData, ydata = YDataMore, maxfev = 10000, p0 = (1, 1, 1, 1, 1, 1, 10), bounds = [[0,0,0,0,0,0,0], [1,1,1,1,1,1,N]])
        Parameter_pp = [ParaOpt_pp[0], ParaOpt_pp[1], ParaOpt_pp[2], ParaOpt_pp[3], ParaOpt_pp[4], ParaOpt_pp[5], ParaOpt_pp[6], N - ParaOpt_pp[6], 0, 0, 0, 0]

        YDataMore = np.hstack((YReports, D_new1-YReports, YDeaths))
        ParaOpt_pp, ParaCov_pp = optimize.curve_fit(f = FitSEIRM_pp, xdata = XData, ydata = YDataMore, maxfev = 10000, p0 = (1, 1, 1, 1, 1, 1, 10), bounds = [[0,0,0,0,0,0,0], [1,1,1,1,1,1,N]])
        Parameter_pp1 = [ParaOpt_pp[0], ParaOpt_pp[1], ParaOpt_pp[2], ParaOpt_pp[3], ParaOpt_pp[4], ParaOpt_pp[5], ParaOpt_pp[6], N - ParaOpt_pp[6], 0, 0, 0, 0]

        YDataMore = np.hstack((YReports, D_new2-YReports, YDeaths))
        ParaOpt_pp, ParaCov_pp = optimize.curve_fit(f = FitSEIRM_pp, xdata = XData, ydata = YDataMore, maxfev = 10000, p0 = (1, 1, 1, 1, 1, 1, 10), bounds = [[0,0,0,0,0,0,0], [1,1,1,1,1,1,N]])
        Parameter_pp2 = [ParaOpt_pp[0], ParaOpt_pp[1], ParaOpt_pp[2], ParaOpt_pp[3], ParaOpt_pp[4], ParaOpt_pp[5], ParaOpt_pp[6], N - ParaOpt_pp[6], 0, 0, 0, 0]

        YDataMore = np.hstack((YReports, D_new3-YReports, YDeaths))
        ParaOpt_pp, ParaCov_pp = optimize.curve_fit(f = FitSEIRM_pp, xdata = XData, ydata = YDataMore, maxfev = 10000, p0 = (1, 1, 1, 1, 1, 1, 10), bounds = [[0,0,0,0,0,0,0], [1,1,1,1,1,1,N]])
        Parameter_pp3 = [ParaOpt_pp[0], ParaOpt_pp[1], ParaOpt_pp[2], ParaOpt_pp[3], ParaOpt_pp[4], ParaOpt_pp[5], ParaOpt_pp[6], N - ParaOpt_pp[6], 0, 0, 0, 0]

        AlphaList.append(D_alpha)
        MDLCostList.append(MDL(D,Parameter_p,Parameter_pp))
        MDLCostForwardList.append((MDL(D_new1,Parameter_p,Parameter_pp1) + MDL(D_new2,Parameter_p,Parameter_pp2) + MDL(D_new3,Parameter_p,Parameter_pp3))/3)

        if (((MDL(D_new1,Parameter_p,Parameter_pp1) + MDL(D_new2,Parameter_p,Parameter_pp2) + MDL(D_new3,Parameter_p,Parameter_pp3))/3) < MDL(D,Parameter_p,Parameter_pp)):
            D_alpha = D_alpha_new
        else:
            break

    D = YReports / D_alpha
    print (D_alpha, MDL(D,Parameter_p,Parameter_pp))
    
    figure = plt.figure(figsize = (10.24,7.68))
    plt.plot(AlphaList, MDLCostList, label='MDL Cost', color='r', linewidth = 3)
    plt.plot(AlphaList, MDLCostForwardList, label='MDL Cost of next Alpha', color='b', linewidth = 2)
    plt.axvline(x = 0.25,color='#660874',linewidth=1,linestyle="dashed")
    plt.text(0.25,max(MDLCostList),'alpha_ground_truth')
    plt.legend()
    plt.xlabel('alpha (D = D_reported/alpha)')
    plt.ylabel('MDL(D_reported/alpha, p)')
    plt.show()
    
