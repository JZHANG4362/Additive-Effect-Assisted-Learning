import pandas as pd
import statsmodels.api as sm
import numpy as np
import math
import scipy.stats
from sklearn import preprocessing

# read the data of angels '1' and '3'
df1Train = pd.read_csv('TrainSpeed_1without_finetune.csv', header=None)
df1Val = pd.read_csv('ValSpeed_1without_finetune.csv', header=None)

df1 = pd.concat([df1Train, df1Val], ignore_index=True)
print(df1.shape)

df2Train = pd.read_csv('TrainSpeed_2without_finetune.csv', header=None)
df2Val = pd.read_csv('ValSpeed_2without_finetune.csv', header=None)

df2 = pd.concat([df2Train, df2Val], ignore_index=True)
print(df2.shape)


# check whether the responses match
print(pd.DataFrame.equals(df1[df1.columns[0]], df2[df2.columns[0]]))


# extract the response data
Y = df1[0].to_numpy()

# predictors data from A
XA = df2[df2.columns[1:20]].to_numpy()
# predictors data from B
XB = df1[df1.columns[1:20]].to_numpy()

# add intercept terms that consist of 1 to the predictor datasets
# XA matrix with intercept
XAintercept = sm.add_constant(XA)




# type of the test statistic
testtype = "Wald"



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
    np.random.seed(10)
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
        
        XAintercept_add_tmp = np.concatenate((XAintercept_add, uXB), axis = 1)
        # #=============delete linearly dependent columns=============
        # q,r = np.linalg.qr(XAintercept_add_tmp.T)

        # XAintercept_add = XAintercept_add_tmp[:, np.abs(np.diag(r))>=1e-10]
        # #========================fit the H1 model========================
        XAintercept_add = XAintercept_add_tmp
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

    # [True, True, True]
    print(rejList)
    print(pVList)


# for Lapscale in [0, 0.25, 0.5]:
#     TestFun('Wald', Lapscale)


np.random.seed(30)
for Lapscale in [0, 0.1, 0.25]:
    TestFun('Wald', Lapscale)

# [True, True, True]
# [1.842581642819141e-11, 3.961719841072409e-12, 1.828104334578029e-11]
# [True, True, True]
# [5.259122032219565e-05, 1.5177294865331703e-08, 3.015325245048217e-08]
# [False, True, True]
# [0.13281409963270752, 0.0016190877977697582, 0.0021335185951580904]

#  0.05/3
# 0.016666666666666666