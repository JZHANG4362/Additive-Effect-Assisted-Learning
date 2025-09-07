#========================================================================================================================
# This module contains two functions: 
# experiment_test: single experiment
# repexperiment_test: independently repeat the same experiment multiple times. 
#========================================================================================================================
import statsmodels.api as sm
import numpy as np
import scipy
import pickle
from scipy.linalg import sqrtm
#=======================function for single experiment====================
def experiment_test(Model, hypothesis, statCal, randomBeta, Lapscale, Xdist, Umatrix, testtype, sig, n, K, ResponseGenVec, fitmodel, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3):

    #========================Generate data========================

    # calculate the total number of coefficients from each group
    p1 = p1_1 + p1_2 + p1_3
    p2 = p2_1 + p2_2 + p2_3
    p3 = p3_1 + p3_2 + p3_3

    # # generate different data-generating model coefficients from normal distributions
    # beta1 = np.concatenate((np.random.normal(loc = 0, scale = v1_1, size = p1_1), np.random.normal(loc = 0, scale = v1_2, size = p1_2), np.random.normal(loc = 0, scale = v1_3, size = p1_3)))

    # # the data from B is independent with the response
    # beta2 = np.zeros(p2)

    # beta3 = np.concatenate((np.random.normal(loc = 0, scale = v3_1, size = p3_1), np.random.normal(loc = 0, scale = v3_2, size = p3_2), np.random.normal(loc = 0, scale = v3_3, size = p3_3)))

    # obtain the coefficients
    if randomBeta == "False":
        beta1 = np.concatenate((np.repeat(v1_1, p1_1), np.repeat(v1_2, p1_2), np.repeat(v1_3, p1_3)))
        if hypothesis == "H0":
            # the data from B is independent with the response
            beta2 = np.zeros(p2)
        elif hypothesis == "H1":
            beta2 = np.concatenate((np.repeat(v2_1, p2_1), np.repeat(v2_2, p2_2), np.repeat(v2_3, p2_3)))

        
        beta3 = np.concatenate((np.repeat(v3_1, p3_1), np.repeat(v3_2, p3_2), np.repeat(v3_3, p3_3)))


        beta = np.concatenate((beta1, beta2, beta3))
    elif randomBeta == "True":
        # generate different data-generating model coefficients from normal distributions
        beta1 = np.concatenate((np.random.normal(loc = 0, scale = v1_1, size = p1_1), np.random.normal(loc = 0, scale = v1_2, size = p1_2), np.random.normal(loc = 0, scale = v1_3, size = p1_3)))
        if hypothesis == "H0":
            # the data from B is independent with the response
            beta2 = np.zeros(p2)
        elif hypothesis == "H1":
            beta2 = np.concatenate((np.random.normal(loc = 0, scale = v2_1, size = p2_1), np.random.normal(loc = 0, scale = v2_2, size = p2_2), np.random.normal(loc = 0, scale = v2_3, size = p2_3)))
        beta3 = np.concatenate((np.random.normal(loc = 0, scale = v3_1, size = p3_1), np.random.normal(loc = 0, scale = v3_2, size = p3_2), np.random.normal(loc = 0, scale = v3_3, size = p3_3)))

        beta = np.concatenate((beta1, beta2, beta3))

    # generate predictor data 
    if Xdist == "normal":
        X_tmp = np.random.normal(loc = 0, scale = 1, size = (n,(p1 + p2 + p3)))
    elif Xdist == "uniform":
        X_tmp = np.random.uniform(low=0, high=1, size = (n,(p1 + p2 + p3)))

    p = p1 + p2 + p3

    # transform X_tmp to have AR1 covariance
    mat_tmp1 = np.tile(np.arange(p), p).reshape(p,p)
    mat_tmp2 = np.transpose(mat_tmp1)
    mat_tmp3 = abs(mat_tmp1 - mat_tmp2)

    mat_tmp4 = sqrtm(sig**mat_tmp3)

    X = np.matmul(X_tmp, mat_tmp4)
    
    # A's unique data
    X1 = X[:, 0:p1]
    # B's unique data
    X2 = X[:, p1:(p1+p2)]
    # shared data
    X3 = X[:, (p1+p2):(p1+p2+p3)]

    # data from A
    XA = np.concatenate((X1, X3), axis = 1)

    # data from B
    XB = np.concatenate((X2, X3), axis = 1)

    # linear predictor from the data-generating model
    LpVec = np.dot(X, beta)

    # generate the response data
    Y = ResponseGenVec(LpVec)

    # add intercept terms that consist of 1 to the predictor datasets
    # XA matrix with intercept
    XAintercept = sm.add_constant(XA)



    if Model == "logcosh":
        #========================define functions for fitting log-cosh====================
        logcoshA = 0.3
        def gradCal(Y, XAintercept, beta0):
            # get the residual
            resid = Y - np.dot(XAintercept, beta0)
            # calculate the gradient
            grad = np.mean((np.tanh(logcoshA * resid).reshape(n,-1)) * XAintercept, axis = 0)
            return(grad)

        def HessCal(Y, XAintercept, beta0):
            # get the residual
            resid = Y - np.dot(XAintercept, beta0)
            # calculate the Hessian
            Hess = logcoshA * n**(-1) * np.dot(np.dot(np.transpose(XAintercept), np.diag(np.cosh(logcoshA * resid)**(-2))), XAintercept)
            return(Hess)
        def fitLogCosh(Y, XAintercept):
            # fit a linear regression as the initial value
            model = sm.GLM(endog = Y, exog = XAintercept, family = fitmodel)
            # get initial value
            beta0 = model.fit().params

            # evaluate the gradient
            grad_updated = gradCal(Y, XAintercept, beta0)
            gradL2 = np.mean(grad_updated**2)

            # set convergence threthold
            thre = 1e-15
            # set counter
            ct = 0
            while (gradL2 > thre) | (ct <100):
                grad = gradCal(Y, XAintercept, beta0)
                Hess = HessCal(Y, XAintercept, beta0)
                beta0 = beta0 + np.dot(np.linalg.inv(Hess), grad)
                grad_updated = gradCal(Y, XAintercept, beta0)
                ct = ct + 1
                gradL2 = np.mean(grad_updated**2)
            return beta0
    if testtype == "likelihoodRatio":
        #========================fit the H0 model========================
        # A fits the initial model
        modelH0 = sm.GLM(endog = Y, exog = XAintercept, family = fitmodel)
        modelH0_results = modelH0.fit()

    # list of p-values
    pVList = []
    # list of rejection results
    rejList = []

    # initialize the predictor data for A
    XAintercept_add = XAintercept
    for i in range(K):
        # obtain the random vector u
        u = Umatrix[:,i]
        uXB_tmp = np.dot(XB, u)
        # add noise
        uXB = uXB_tmp + np.random.laplace(loc=0, scale=Lapscale, size=uXB_tmp.shape)
  
        XAintercept_add_tmp = np.concatenate((XAintercept_add, uXB.reshape(uXB.shape[0],-1)), axis = 1)
        #=============delete linearly dependent columns=============
        q,r = np.linalg.qr(XAintercept_add_tmp.T)

        XAintercept_add = XAintercept_add_tmp[:, np.abs(np.diag(r))>=1e-10]
        # XAintercept_add = np.concatenate((XAintercept_add, uXB), axis = 1)
        #========================fit the H1 model========================
        # A fits the H1 model after receiving the data
        if Model == "logcosh":
            betaFitted = fitLogCosh(Y, XAintercept_add)
        else:
            modelH1 = sm.GLM(endog = Y, exog = XAintercept_add, family = fitmodel)
            modelH1_results = modelH1.fit()
            betaFitted = modelH1_results.params
        # test statistic
        if testtype == "likelihoodRatio":
            stat = 2 * (modelH1_results.llf - modelH0_results.llf)
        elif testtype == "Wald":
            # linear predictor
            lp = np.dot(XAintercept_add, betaFitted)
            
            # prediction result
            if Model == "logcosh":
                prd =  np.dot(XAintercept_add, betaFitted)
            else:
                prd = modelH1_results.predict(XAintercept_add)
            # number of predictors from A
            pA = XAintercept.shape[1]
            # fitted coefficients
            # prediction result
            betaU =betaFitted[pA: ]

            
            stat = statCal(Y, XAintercept_add, pA, prd, betaU, lp)
        df = (XAintercept_add.shape[1] - XAintercept.shape[1])
# #########
#         print(df)
#         df = i + 1
#         # print(modelH1_results.llf)
#         # print(modelH0_results.llf)
#         print(i+1)
# #########
        # p value
        pV = 1 - scipy.stats.chi2.cdf(stat, df)
        pVList.append(pV)
        # rejection
        rej = pV < 0.05
        rejList.append(rej)
    return pVList, rejList


#=============function to repeat the experiment multiple times============
def repexperiment_test(Model, hypothesis, statCal, randomBeta, Lapscale, Xdist, fixedU, testtype, sig, niter, n, K, ResponseGenVec, fitmodel, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3):
    p2 = p2_1 + p2_2 + p2_3
    p3 = p3_1 + p3_2 + p3_3
    p23 = p2 + p3
    #==================If want to fix a set U and apply different n, read the generated U values====
    if fixedU == True:
        infile = open("uArray_" + str(p23) + '_dic.p', 'rb')
        new_dict = pickle.load(infile)
        infile.close()
        
    pV_array = np.empty((0, K))
    rej_array = np.empty((0, K))   

    for j in range(niter):
        # while True:
        #     try: 
        #         if fixedU == True:
        #             Umatrix = new_dict[j, :, :]
        #         elif fixedU == False:
        #             # generate normal random values
        #             u_tmp = np.random.normal(loc = 0, scale = 1, size = (p23, K))
        #             # standardize each column
        #             L2Norms = np.sum(np.abs(u_tmp)**2,axis=0)**(1./2)
        #             Umatrix = u_tmp/L2Norms
        #         print("iter: " + str(j))
        #         pVList, rejList = experiment_test(Model, hypothesis, statCal, randomBeta, Lapscale, Xdist, Umatrix, testtype, sig, n, K, ResponseGenVec, fitmodel, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
        #         break
        #     except BaseException:
        #         print('Errors encountered in model fitting, possibly due to numerical issues. Generate the dataset again')
        if fixedU == True:
            Umatrix = new_dict[j, :, :]
        elif fixedU == False:
            # generate normal random values
            u_tmp = np.random.normal(loc = 0, scale = 1, size = (p23, K))
            # standardize each column
            L2Norms = np.sum(np.abs(u_tmp)**2,axis=0)**(1./2)
            Umatrix = u_tmp/L2Norms
        print("iter: " + str(j))
        pVList, rejList = experiment_test(Model, hypothesis, statCal, randomBeta, Lapscale, Xdist, Umatrix, testtype, sig, n, K, ResponseGenVec, fitmodel, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
        pV_array = np.concatenate((pV_array, np.array(pVList).reshape(1,-1)), axis = 0)
        rej_array = np.concatenate((rej_array, np.array(rejList).reshape(1,-1)), axis = 0)
    return pV_array, rej_array

