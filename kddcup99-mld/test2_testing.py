import pandas as pd
import statsmodels.api as sm
import numpy as np
import math
import scipy.stats
from sklearn import preprocessing
import pickle

#=============================read data=====================================

fname = "data_preprocessed_dic.p"
infile = open(fname, 'rb')
new_dict = pickle.load(infile)
infile.close()



Y = new_dict['Y_tmp']



XA = new_dict['XA_tmp']
XB = new_dict['XB_tmp']
# # drop linearly dependent columns
# q1,r1 = np.linalg.qr(XA_tmp.T)
# XA = XA_tmp[:, np.abs(np.diag(r1))>=1e-10]

# q2,r2 = np.linalg.qr(XB_tmp.T)
# XB = XB_tmp[:, np.abs(np.diag(r2))>=1e-10]

# add intercept terms that consist of 1 to the predictor datasets
# XA matrix with intercept
XAintercept = sm.add_constant(XA)

def TestFun(testtype, Lapscale):
    #=============================specify the model to fit============================
    fitmodel = sm.families.Binomial(link = sm.families.links.logit())

    #========================fit the H0 model========================
    # A fits the initial model
    modelH0 = sm.GLM(endog = Y, exog = XAintercept, family = fitmodel)
    modelH0_results = modelH0.fit()

    # list of p-values
    pVList = []
    # list of rejection results
    rejList = []

    # number of random vectors to send
    K = 3

    # initialize the predictor data for A
    XAintercept_add = XAintercept
    for i in range(K):
        # B generates the random vector u and sends the vector uXB
        u_tmp = np.random.normal(loc = 0, scale = 1, size = (XB.shape[1], 1))

        # standardize each column
        L2Norms = np.sum(np.abs(u_tmp)**2,axis=0)**(1./2)
        u = u_tmp/L2Norms
        # normalize XB
        min_max_scaler = preprocessing.MinMaxScaler()
        XB_normalized = min_max_scaler.fit_transform(XB)


        uXB_tmp = np.dot(XB_normalized, u)

        # add noise
        uXB = uXB_tmp + np.random.laplace(loc=0, scale=Lapscale, size=uXB_tmp.shape)
        
        #XAintercept_add_tmp = np.concatenate((XAintercept_add, uXB), axis = 1)
        XAintercept_add = np.concatenate((XAintercept_add, uXB), axis = 1)
        # #=============delete linearly dependent columns=============
        # q,r = np.linalg.qr(XAintercept_add_tmp.T)

        # XAintercept_add = XAintercept_add_tmp[:, np.abs(np.diag(r))>=1e-10]
        #========================fit the H1 model========================
        # A fits the H1 model after receiving the data
        modelH1 = sm.GLM(endog = Y, exog = XAintercept_add, family = fitmodel)
        modelH1_results = modelH1.fit()

        # test statistic
        if testtype == "likelihoodRatio":
            stat = 2 * (modelH1_results.llf - modelH0_results.llf)
        elif testtype == "Wald":
            # linear predictor
            lp = np.dot(XAintercept_add, modelH1_results.params)
            # calculate the gradient for each observation

            resid =  (Y - modelH1_results.predict(XAintercept_add))
            Grad = resid.reshape(len(resid), -1) * XAintercept_add

            # check
            # print(Grad[2,:])
            # print(resid[2] * XAintercept_add[2,:])

            #calculate the Fisher information matrix
            V2 = np.dot(np.transpose(Grad), Grad)/len(Y)
            c1 = (1 + math.e**(-lp))**(-2) * math.e**(-lp)
            # calculate the negative hessian
            V1 = np.dot(np.dot(np.transpose(XAintercept_add), np.diag(c1)), XAintercept_add)/len(Y)
            V = np.dot(np.dot(np.linalg.inv(V1), V2), np.linalg.inv(V1))
            Vt = V[XAintercept.shape[1]:, XAintercept.shape[1]:]
            betaU = modelH1_results.params[XAintercept.shape[1]: ]
            stat = len(Y) * np.dot(np.dot(np.transpose(betaU), np.linalg.inv(Vt)), betaU)

        df = (XAintercept_add.shape[1] - XA.shape[1])
        
        # p value
        pV = 1 - scipy.stats.chi2.cdf(stat, df)
        pVList.append(pV)
        # rejection
        rej = pV < 0.05
        rejList.append(rej)
        
    print(rejList)
    print(pVList)
    # [True, True, True]

np.random.seed(20)
for Lapscale in [0, 0.1, 0.5]:
    TestFun('Wald', Lapscale)

# [True, True, True]
# [6.309078703914395e-09, 0.0, 0.0]
# [True, True, True]
# [1.3824557769215318e-05, 6.836322952175067e-08, 0.0]
# [True, True, True]
# [0.004105776107113601, 2.7450723527167042e-05, 1.9585049113146624e-05]
# >>> 0.05/3
# 0.016666666666666666