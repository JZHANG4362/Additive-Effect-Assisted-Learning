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
parser.add_argument("--hypothesis", default="H0", type=str, help="H0 or H1")

args = parser.parse_args()


# obtain the parameter values from the commandline
Model = args.Model

hypothesis = args.hypothesis

# Model = "logcosh"
# hypothesis = "H0"

# # type of the model to test
# Model = "logistic"
# # H0 or H1
# hypothesis = "H0"

#===================================list of parameters===============================
# Whether randomly generate betas
if hypothesis == "H0":
    randomBeta = "False"
elif hypothesis == "H1":
    randomBeta = "True"
# distribution to generate covariate observations
XdistList = ["uniform"]
# type of the test statistic
testtypeList = ["Wald"]
# training sample size
nList = [2000]
# list of the scales of the Laplace noise to control local differential privacy (for Xdist=normal, consider Lapscale=0 only)
# LapscaleList = [0, 0.1, 0.5]
LapscaleList = [0.1]
# fixedUList = [True, False]
fixedUList = [False]
# covariance values
sigList  = [0.1]

#===================model pattern settings
# number of different types of coefficients in the model
pNumList = np.array([[4,0,0, 4,0,0, 4,0,0]])
# number of settings
nSettings = pNumList.shape[0]
# list of setting indices. 
IndexList = list(range(1,(nSettings + 1)))

# number of vectors sent
K = 5
# number of iterations
# niter = 100
niter = 500


#===========================create folder to store output results=====================
results_path_data = "./" + Model + "_results/results" + hypothesis + "/data"
# Check whether the specified path exists or not
isExist = os.path.exists(results_path_data)

if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(results_path_data)

results_path_figures = "./" + Model + "_results/results" + hypothesis + "/Figures"
# Check whether the specified path exists or not
isExist = os.path.exists(results_path_figures)

if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(results_path_figures)
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
    # assign three different standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    # standard deviations to generate the coefficients from A
    v1_1 = 0.5
    v1_2 = 0
    v1_3 = 0

    # standard deviations to generate the coefficients from B
    v2_1 = 0.5
    v2_2 = 0
    v2_3 = 0

    # standard deviations to generate the coefficients from shared variables
    v3_1 = 0.5
    v3_2 = 0
    v3_3 = 0
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
    # assign three different standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    # standard deviations to generate the coefficients from A
    v1_1 = 0.5
    v1_2 = 0
    v1_3 = 0

    # standard deviations to generate the coefficients from B
    v2_1 = 0.5
    v2_2 = 0
    v2_3 = 0

    # standard deviations to generate the coefficients from shared variables
    v3_1 = 0.5
    v3_2 = 0
    v3_3 = 0
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
    # assign three different standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    # standard deviations to generate the coefficients from A
    v1_1 = 0.2
    v1_2 = 0
    v1_3 = 0

    # standard deviations to generate the coefficients from B
    v2_1 = 0.2
    v2_2 = 0
    v2_3 = 0

    # standard deviations to generate the coefficients from shared variables
    v3_1 = 0.2
    v3_2 = 0
    v3_3 = 0
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
        V1 = logcoshA*n**(-1) * np.dot(np.dot(np.transpose(XAintercept_add), np.diag(np.cosh(logcoshA*resid)**(-1))), XAintercept_add)
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
    # assign three different standard deviations for each group (unique variables from A, unique variables from B, and shared variables)

    # standard deviations to generate the coefficients from A
    v1_1 = 0.5
    v1_2 = 0
    v1_3 = 0

    # standard deviations to generate the coefficients from B
    v2_1 = 0.5
    v2_2 = 0
    v2_3 = 0

    # standard deviations to generate the coefficients from shared variables
    v3_1 = 0.5
    v3_2 = 0
    v3_3 = 0


# generate the U matrix from t distribution with degrees of freedom 3, 5, and 10, and uniform(-1, 1)
UgenList = ["t3", "t5", "t10", "Ump1"]
#==========================================Testing in various settings========================

# set random seed
np.random.seed(30)



for fixedU in fixedUList:
    if Model == "logcosh":
        testtypeList = ["Wald"]
    for testtype in testtypeList:
        for Xdist in XdistList:
            if Xdist == "normal":
                # list of the scales of the Laplace noise to control local differential privacy
                LapscaleList = [0]
            for sig in sigList:
                for Lapscale in LapscaleList:   
                    for n in nList:
                        for Index in IndexList:
                            # obtain the number of each type of coefficients in different settings
                            Id = Index - 1
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
                            #==========================================Specify the function to generate the U matrix===============
                            #Ugen = "normal"
                            for Ugen in UgenList:
                                if Ugen == "normal":
                                    def UgenFun(sizeU):
                                        return np.random.normal(loc = 0, scale = 1, size = (sizeU[0], sizeU[1]))
                                # t distribution with degrees of freedom 3
                                elif Ugen == "t3":
                                    def UgenFun(sizeU):
                                        return np.random.standard_t(3, size = (sizeU[0], sizeU[1]))
                                # t distribution with degrees of freedom 5
                                elif Ugen == "t5":
                                    def UgenFun(sizeU):
                                        return np.random.standard_t(5, size = (sizeU[0], sizeU[1]))
                                # t distribution with degrees of freedom 10
                                elif Ugen == "t10":
                                    def UgenFun(sizeU):
                                        return np.random.standard_t(10, size = (sizeU[0], sizeU[1]))
                                elif Ugen == "Ump1":
                                    def UgenFun(sizeU):
                                        return np.random.uniform(low=-1, high=1, size = (sizeU[0], sizeU[1]))


                                # obtain the tesing results
                                pV_array, rej_array = repexperiment_test(UgenFun, Model, hypothesis, statCal, randomBeta, Lapscale, Xdist, fixedU , testtype, sig, niter, n, K, ResponseGenVec, fitmodel, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
                                
                                # file name
                                fname = Model + "_Setting_" + str(Index) + "_n_" + str(n) + "_Lapscale_" + str(Lapscale) + "_sig_" + str(sig) + "_Xdist_" + str(Xdist) + "_testtype_" + testtype + "_fixedU_" + str(fixedU) + "_randomBeta_" + randomBeta + "_Ugen_" + Ugen
                                
                                
                                #==============================================export the results===============================================
                                result = {"pV": pV_array, "rej": rej_array}

                                pickle.dump(result, open(results_path_data + "/" + fname + "_dic.p", "wb"))

                                #===============================================plot the evaluation results=======================================
                                # calculate means and standard deviations of different metrics
                                rej_mean = np.mean(rej_array, axis = 0)
                                rej_std = np.std(rej_array, axis = 0)


                                # plot the evaluation results
                                fit, (ax1) = plt.subplots(1,1, figsize = (8, 8))
                                ax1.errorbar(np.array(range(1, (K+1))), rej_mean , rej_std, marker = '.', color = "b")
                                ax1.set_xlabel("Number of data vectors")
                                ax1.set_ylabel("Rejection rates")
                                ax1.title.set_text(Model + " regression " + hypothesis)

                                # fit.show()
                                plt.savefig(results_path_figures + "/" + fname + '_plot.pdf')

                                