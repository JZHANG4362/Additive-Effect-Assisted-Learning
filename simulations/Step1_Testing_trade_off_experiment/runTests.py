import statsmodels.api as sm
import numpy as np
import math
import os
import argparse
import pickle
import matplotlib.pyplot as plt

from Experiment_testing import repexperiment_test

parser = argparse.ArgumentParser()
parser.add_argument("--Model", default="logistic", type=str, help="logistic, normal, or poisson")
parser.add_argument("--hypothesis", default="H1", type=str, help="H0 or H1")
parser.add_argument("--mechanism", default="Laplace", type=str, help="Laplace or Gaussian")


args = parser.parse_args()


# obtain the parameter values from the commandline
Model = args.Model

hypothesis = args.hypothesis

mechanism = args.mechanism

# Model = "logcosh"
# hypothesis = "H0"

# # type of the model to test
# Model = "logistic"
# # H0 or H1
# hypothesis = "H0"


#===================================list of parameters===============================
# Whether randomly generate betas
# In this set of experiments, we do not randomly generate beta.
randomBeta = "False"

# distribution to generate covariate observations
# XdistList = ["uniform", "normal"]
XdistList = ["uniform"]
# type of the test statistic
# testtypeList = ["Wald", "likelihoodRatio"]
testtypeList = ["Wald"]
# training sample size
nList = [300, 2000]
# The upper bound of the L2 norm of the covariates
XBound = 1
# Epsilon in local differential privacy protection
#LapEpsilon = 100
LapEpsilon = 20
# fix a set of randomly generated U and change sample size
# fixedUList = [True, False]
fixedUList = [False]
# covariance values
sigList  = [0, 0.1, 0.25]

#===================model pattern settings
# number of different types of coefficients in the model
pNumList = np.array([[6,0,0, 6,0,0, 0,0,0], [3,0,0, 6,0,0, 3,0,0], [3,0,0, 9,0,0, 0,0,0],
                     [3,0,0, 1,5,0, 0,0,0], [3,0,0, 1,5,0, 3,0,0], [3,0,0, 1,8,0, 0,0,0],
                     [3,0,0, 1,5,0, 0,0,0], [3,0,0, 1,5,0, 3,0,0], [3,0,0, 1,8,0, 0,0,0]])
# number of settings
nSettings = pNumList.shape[0]
# list of setting indices. 
IndexList = list(range(1,(nSettings + 1)))

# number of vectors sent
K = 5
# number of iterations
niter = 100


#===========================create folder to store output results=====================
results_path_data = "./" + Model + "_results/results" + hypothesis + "/data"
# Check whether the specified path exists or not
isExist = os.path.exists(results_path_data)

if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(results_path_data)

#===========================model-specific settings=====================

if Model == "logistic":
    #=============================function to calculate the test statistic============================
    def statCal(Y, XAintercept_add, pA, prd, betaU, lp):
        # calculate the gradient for each observation
        resid =  (Y - prd)
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
        Vt = V[pA:, pA:]
        
        stat = len(Y) * np.dot(np.dot(np.transpose(betaU), np.linalg.inv(Vt)), betaU)
        return stat
    #=============================function to calculate the inverse logit link============================
    def invLink(lp):
        pv = 1/(1 + math.exp(-lp))
        return pv

    # vectorize the function
    invLinkVec = np.vectorize(invLink)

    #======================function to generate response data given linear predictors=====================
    def ResponseGen(lp):
        pv = invLink(lp)
        binoRand = np.random.binomial(1, pv, 1)
        return binoRand

    # vectorize the function
    ResponseGenVec = np.vectorize(ResponseGen)

    #=============================specify the model to fit============================
    fitmodel = sm.families.Binomial(link = sm.families.links.logit())

    #======================specify the data-generating coefficients=====================
    # the coefficients of the data-generating model are generated from normal distributions.
    # assign three different values/standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    vList = np.array([[0.5,0,0, 0.5,0,0, 0.5,0,0], [0.5,0,0, 0.5,0,0, 0.5,0,0], [0.5,0,0, 0.5,0,0, 0.5,0,0],
                      [0.5,0,0, 1.2247,0,0, 0.5,0,0], [0.5,0,0, 1.2247,0,0, 0.5,0,0], [0.5,0,0, 1.5,0,0, 0.5,0,0],
                      [0.5,0,0, 2.4494,0,0, 0.5,0,0], [0.5,0,0, 2.4494,0,0, 0.5,0,0], [0.5,0,0, 3,0,0, 0.5,0,0]])
elif Model == "normal":
    #=============================function to calculate the test statistic============================
    def statCal(Y, XAintercept_add, pA, prd, betaU, lp):
        # calculate the gradient for each observation
        resid =  (Y - prd)
        Grad = resid.reshape(len(resid), -1) * XAintercept_add

        # check
        # print(Grad[2,:])
        # print(resid[2] * XAintercept_add[2,:])

        #calculate the Fisher information matrix
        V2 = np.dot(np.transpose(Grad), Grad)/len(Y)
        c1 = np.repeat(1,XAintercept_add.shape[0])
        # calculate the negative hessian
        V1 = np.dot(np.dot(np.transpose(XAintercept_add), np.diag(c1)), XAintercept_add)/len(Y)
        V = np.dot(np.dot(np.linalg.inv(V1), V2), np.linalg.inv(V1))
        Vt = V[pA:, pA:]
        stat = len(Y) * np.dot(np.dot(np.transpose(betaU), np.linalg.inv(Vt)), betaU)
        return stat
    #=============================function to calculate the inverse link============================
    def invLink(lp):
        pv = lp
        return pv

    # vectorize the function
    invLinkVec = np.vectorize(invLink)

    #======================function to generate response data given linear predictors=====================
    def ResponseGen(lp):
        pv = invLink(lp)
        binoRand = np.random.normal(pv, 1)
        return binoRand

    # vectorize the function
    ResponseGenVec = np.vectorize(ResponseGen)

    #=============================specify the model to fit============================
    fitmodel = sm.families.Gaussian(link = sm.families.links.identity())

    #======================specify the data-generating coefficients=====================
    # the coefficients of the data-generating model are generated from normal distributions.
    # assign three different values/standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    
    vList = np.array([[0.5,0,0, 0.5,0,0, 0.5,0,0], [0.5,0,0, 0.5,0,0, 0.5,0,0], [0.5,0,0, 0.5,0,0, 0.5,0,0],
                      [0.5,0,0, 1.2247,0,0, 0.5,0,0], [0.5,0,0, 1.2247,0,0, 0.5,0,0], [0.5,0,0, 1.5,0,0, 0.5,0,0],
                      [0.5,0,0, 2.4494,0,0, 0.5,0,0], [0.5,0,0, 2.4494,0,0, 0.5,0,0], [0.5,0,0, 3,0,0, 0.5,0,0]])
elif Model == "poisson":
    #=============================function to calculate the test statistic============================
    def statCal(Y, XAintercept_add, pA, prd, betaU, lp):
        # calculate the gradient for each observation
        resid =  (Y - prd)
        Grad = resid.reshape(len(resid), -1) * XAintercept_add

        # check
        # print(Grad[2,:])
        # print(resid[2] * XAintercept_add[2,:])

        #calculate the Fisher information matrix
        V2 = np.dot(np.transpose(Grad), Grad)/len(Y)
        c1 = math.e**(lp)
        # calculate the negative hessian
        V1 = np.dot(np.dot(np.transpose(XAintercept_add), np.diag(c1)), XAintercept_add)/len(Y)
        V = np.dot(np.dot(np.linalg.inv(V1), V2), np.linalg.inv(V1))
        Vt = V[pA:, pA:]
        betaU = betaU
        stat = len(Y) * np.dot(np.dot(np.transpose(betaU), np.linalg.inv(Vt)), betaU)
        return stat
    #=============================function to calculate the inverse link============================
    def invLink(lp):
        pv = math.exp(lp)
        return pv

    # vectorize the function
    invLinkVec = np.vectorize(invLink)

    #======================function to generate response data given linear predictors=====================
    def ResponseGen(lp):
        pv = invLink(lp)
        binoRand = np.random.poisson(pv, 1)
        return binoRand

    # vectorize the function
    ResponseGenVec = np.vectorize(ResponseGen)

    #=============================specify the model to fit============================
    fitmodel = sm.families.Poisson(link = sm.families.links.log())

    #======================specify the data-generating coefficients=====================
    # the coefficients of the data-generating model are generated from normal distributions.
    # assign three different values/standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    vList = np.array([[0.2,0,0, 0.2,0,0, 0.2,0,0], [0.2,0,0, 0.2,0,0, 0.2,0,0], [0.2,0,0, 0.2,0,0, 0.2,0,0]])


    vList = np.array([[0.2,0,0, 0.2,0,0, 0.2,0,0], [0.2,0,0, 0.2,0,0, 0.2,0,0], [0.2,0,0, 0.2,0,0, 0.2,0,0],
                      [0.2,0,0, 0.49,0,0, 0.2,0,0], [0.2,0,0, 0.49,0,0, 0.2,0,0], [0.2,0,0, 0.6,0,0, 0.2,0,0],
                      [0.2,0,0, 0.98,0,0, 0.2,0,0], [0.2,0,0, 0.98,0,0, 0.2,0,0], [0.2,0,0, 1.2,0,0, 0.2,0,0]])
    
elif Model == "logcosh":
    logcoshA = 0.3
    #=============================function to calculate the test statistic============================
    def statCal(Y, XAintercept_add, pA, prd, betaU, lp):
        # calculate the gradient for each observation
        resid =  (Y - prd)
        Grad =  -(np.tanh(logcoshA*resid).reshape(n,-1)) * XAintercept_add
        # check
        # print(Grad[2,:])
        # print(resid[2] * XAintercept_add[2,:])

        #calculate the Fisher information matrix
        V2 = np.dot(np.transpose(Grad), Grad)/len(Y)
        # calculate the negative hessian
        V1 = logcoshA*n**(-1) * np.dot(np.dot(np.transpose(XAintercept_add), np.diag(np.cosh(logcoshA*resid)**(-2))), XAintercept_add)
        V = np.dot(np.dot(np.linalg.inv(V1), V2), np.linalg.inv(V1))
        Vt = V[pA:, pA:]
        stat = len(Y) * np.dot(np.dot(np.transpose(betaU), np.linalg.inv(Vt)), betaU)
        return stat
    #=============================function to calculate the inverse link============================
    def invLink(lp):
        pv = lp
        return pv

    # vectorize the function
    invLinkVec = np.vectorize(invLink)

    #======================function to generate response data given linear predictors=====================
    def ResponseGen(lp):
        pv = invLink(lp)
        binoRand = np.random.normal(pv, 1)
        return binoRand

    # vectorize the function
    ResponseGenVec = np.vectorize(ResponseGen)

    #=============================specify the model to fit============================
    fitmodel = sm.families.Gaussian(link = sm.families.links.identity())

    #======================specify the data-generating coefficients=====================
    # the coefficients of the data-generating model are generated from normal distributions.
    # assign three different values/standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    
    vList = np.array([[0.5,0,0, 0.5,0,0, 0.5,0,0], [0.5,0,0, 0.5,0,0, 0.5,0,0], [0.5,0,0, 0.5,0,0, 0.5,0,0],
                      [0.5,0,0, 1.2247,0,0, 0.5,0,0], [0.5,0,0, 1.2247,0,0, 0.5,0,0], [0.5,0,0, 1.5,0,0, 0.5,0,0],
                      [0.5,0,0, 2.4494,0,0, 0.5,0,0], [0.5,0,0, 2.4494,0,0, 0.5,0,0], [0.5,0,0, 3,0,0, 0.5,0,0]])
#==========================================Testing in various settings========================

# set random seed
np.random.seed(30)



for fixedU in fixedUList:
    if Model == "logcosh":
        testtypeList = ["Wald"]
    for testtype in testtypeList:
        for Xdist in XdistList:
            for sig in sigList:
                for n in nList:
                    for Index in IndexList:
                        # obtain the number of each type of coefficients in different settings
                        Id = Index - 1

                        #===================================assign the number of different kinds of coefficients=============================
                        # number of three kinds of coefficients from A
                        p1_1 = pNumList[Id, 0]
                        p1_2 = pNumList[Id, 1]
                        p1_3 = pNumList[Id, 2]

                        # number of three kinds of coefficients from B
                        p2_1 = pNumList[Id, 3]
                        p2_2 = pNumList[Id, 4]
                        p2_3 = pNumList[Id, 5]

                        # number of three kinds of coefficients shared by both A and B
                        p3_1 = pNumList[Id, 6]
                        p3_2 = pNumList[Id, 7]
                        p3_3 = pNumList[Id, 8]

                        #===================================assign the coefficient values/standard deviations=============================
                        # number of three kinds of coefficients from A
                        v1_1 = vList[Id, 0]
                        v1_2 = vList[Id, 1]
                        v1_3 = vList[Id, 2]

                        # number of three kinds of coefficients from B
                        v2_1 = vList[Id, 3]
                        v2_2 = vList[Id, 4]
                        v2_3 = vList[Id, 5]

                        # number of three kinds of coefficients shared by both A and B
                        v3_1 = vList[Id, 6]
                        v3_2 = vList[Id, 7]
                        v3_3 = vList[Id, 8]


                        # obtain the tesing results
                        pV_array, rej_array = repexperiment_test(Model, hypothesis, statCal, randomBeta, mechanism, XBound, LapEpsilon, Xdist, fixedU , testtype, sig, niter, n, K, ResponseGenVec, fitmodel, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
                        
                        # file name
                        fname = Model + "_Setting_" + str(Index) + "_n_" + str(n) + "_mechanism_" + str(mechanism) + "_XBound_" + str(XBound) + "_LapEpsilon_" + str(LapEpsilon) + "_sig_" + str(sig) + "_Xdist_" + str(Xdist) + "_testtype_" + testtype + "_fixedU_" + str(fixedU) + "_randomBeta_" + randomBeta
                        
                        
                        #==============================================export the results===============================================
                        result = {"pV": pV_array, "rej": rej_array}

                        pickle.dump(result, open(results_path_data + "/" + fname + "_dic.p", "wb"))


                            