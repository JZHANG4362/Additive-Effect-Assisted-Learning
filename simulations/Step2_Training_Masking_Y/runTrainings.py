import statsmodels.api as sm
import numpy as np
import math
import os
import argparse
import pickle
import matplotlib.pyplot as plt

# import the functions from Experiment_fitting1.py
from Experiment_fitting import repExperiment

 
# parser = argparse.ArgumentParser()
# parser.add_argument("--Model", default="logistic", type=str, help="logistic, normal, or poisson")

# args = parser.parse_args()


# # obtain the parameter values from the commandline
# Model = args.Model

Model = "logistic"
#===================================list of parameters===============================
# Whether randomly generate betas
randomBeta = "True"
# distribution to generate covariate observations
XdistList = ["uniform"]
# training sample size
nList = [2000]
# covariance values
sigList  = [0.1]
# masking probability
ppList = [0.1, 0.2, 0.3, 0.4]

#===================model pattern settings
# number of different types of coefficients in the model
pNumList = np.array([[4,0,0, 4,0,0, 4,0,0]])
# number of settings
nSettings = pNumList.shape[0]
# list of setting indices. 
IndexList = list(range(1,(nSettings + 1)))

# number of transmission iterations
kNum = 5

# number of iterations
niter = 100
# set the size of the evaluation data
nEval = 10**6
#===========================create folder to store output results=====================
results_path_data = "./" + Model + "_results/results/data"
# Check whether the specified path exists or not
isExist = os.path.exists(results_path_data)

if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(results_path_data)

results_path_figures = "./" + Model + "_results/results/Figures"
# Check whether the specified path exists or not
isExist = os.path.exists(results_path_figures)

if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(results_path_figures)

#=======================================================================
#===========================model-specific settings=====================

# if Model == "logistic":
# output AUC
reportAUC = True
# do not report ll 
reportll = False
#========================Define the log-likelihood for evaluation========================
# logistic regression log-likelihood
def loglikeli(YEval, predCombined):
    # # clip the predicted probability to avoid it getting too close to 0 or 1
    # predCombined_clipped = np.clip(predCombined, 1e-15, (1-1e-15))

    predCombined_clipped = predCombined
    ll = np.mean(YEval * np.log(predCombined_clipped ) + (1-YEval) * np.log(1 - predCombined_clipped ))
    return ll
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

#==========================================Testing in various settings========================

# set random seed
np.random.seed(30)



for Xdist in XdistList:
    for sig in sigList:
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
                    
                    for pp in ppList:
                        # file name
                        fname = Model + "_Setting_" + str(Index) + "_n_" + str(n) +  "_sig_" + str(sig) + "_Xdist_" + str(Xdist) +  "_randomBeta_" + randomBeta + "_pp_" + str(pp)
                        if reportAUC:
                            if reportll:
                                # obtain the evaluation results
                                AUC_array, EuDis_array, ll_array, AUC_oracle_array, ll_oracle_array = repExperiment(pp, Model, randomBeta, Xdist, sig, niter, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)

                                #==============================================export the results===============================================
                                result = {"EuDis": EuDis_array, "ll": ll_array, "AUC": AUC_array, "ll_oracle": ll_oracle_array , "AUC_oracle": AUC_oracle_array}
                            else:
                                AUC_array, EuDis_array, AUC_oracle_array =  repExperiment(pp, Model, randomBeta, Xdist, sig, niter, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)

                                #==============================================export the results===============================================
                                result = {"EuDis": EuDis_array, "AUC": AUC_array,  "AUC_oracle": AUC_oracle_array}
                        elif reportll:
                            # obtain the evaluation results
                            EuDis_array, ll_array, ll_oracle_array =  repExperiment(pp, Model, randomBeta, Xdist, sig, niter, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)

                            #==============================================export the results===============================================
                            result = {"EuDis": EuDis_array, "ll": ll_array,  "ll_oracle": ll_oracle_array}
                        else:
                            # obtain the evaluation results
                            EuDis_array =  repExperiment(pp, Model, randomBeta, Xdist, sig, niter, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)

                            #==============================================export the results===============================================
                            result = {"EuDis": EuDis_array}


                        pickle.dump(result, open(results_path_data + "/" + fname + "_dic.p", "wb"))
                                    

                        #===============================================plot the evaluation results=======================================
                        

                        # calculate means and standard error of different metrics
                        EuDis_mean = np.mean(EuDis_array, axis = 0)
                        EuDis_std = np.std(EuDis_array, axis = 0)
                        EuDis_ste = EuDis_std/math.sqrt(niter)

                        if reportAUC:

                            AUC_mean = np.mean(AUC_array, axis = 0)
                            AUC_std = np.std(AUC_array, axis = 0)
                            AUC_ste = AUC_std/math.sqrt(niter)
                            
                            AUC_oracle_mean = np.repeat(np.mean(AUC_oracle_array), (kNum + 1) )
                            AUC_oracle_std = np.repeat(np.std(AUC_oracle_array), (kNum + 1) )
                            AUC_oracle_ste = AUC_oracle_std/math.sqrt(niter)
                        if reportll:
                            ll_mean = np.mean(ll_array, axis = 0)
                            ll_std = np.std(ll_array, axis = 0)
                            ll_ste = ll_std/math.sqrt(niter)


                            ll_oracle_mean = np.repeat(np.mean(ll_oracle_array), (kNum + 1) )
                            ll_oracle_std = np.repeat(np.std(ll_oracle_array), (kNum + 1) )
                            ll_oracle_ste = ll_oracle_std/math.sqrt(niter)
                        


                        if reportAUC:
                            if reportll:
                                # plot the evaluation results
                                fit, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (18, 8))
                                ax1.errorbar(np.array(range(kNum + 1)), EuDis_mean , EuDis_ste, marker = '.', color = "b")
                                ax1.set_xlabel("Number of iterations")
                                ax1.set_ylabel("Euclidean distance")
                                ax1.title.set_text("Euclidean distance between the assisted \n learning estimates and the oracle estimates")

                                ax2.errorbar(np.array(range(kNum + 1)), ll_mean , ll_ste, marker = '.', color = "b")
                                ax2.errorbar(np.array(range(kNum + 1)), ll_oracle_mean , ll_oracle_ste, marker = '.', color = "r")
                                ax2.set_xlabel("Number of iterations")
                                ax2.set_ylabel("Negative log-likelihood")
                                ax2.title.set_text("The log-likelihood of the estimates")
                                ax2.legend(["Assisted learning estimator", "Oracle estimator"], loc = "lower right")

                                ax3.errorbar(np.array(range(kNum + 1)), AUC_mean , AUC_ste, marker = '.', color = "b")
                                ax3.errorbar(np.array(range(kNum + 1)), AUC_oracle_mean , AUC_oracle_ste, marker = '.', color = "r")
                                ax3.set_xlabel("Number of iterations")
                                ax3.set_ylabel("AUC")
                                ax3.title.set_text("AUC of the estimates")
                                ax3.legend(["Assisted learning estimator", "Oracle estimator"], loc = "lower right")
                            else:
                                # plot the evaluation results
                                fit, (ax1, ax3) = plt.subplots(1,2, figsize = (18, 8))
                                ax1.errorbar(np.array(range(kNum + 1)), EuDis_mean , EuDis_ste, marker = '.', color = "b")
                                ax1.set_xlabel("Number of iterations")
                                ax1.set_ylabel("Euclidean distance")
                                ax1.title.set_text("Euclidean distance between the assisted \n learning estimates and the oracle estimates")

                                ax3.errorbar(np.array(range(kNum + 1)), AUC_mean , AUC_ste, marker = '.', color = "b")
                                ax3.errorbar(np.array(range(kNum + 1)), AUC_oracle_mean , AUC_oracle_ste, marker = '.', color = "r")
                                ax3.set_xlabel("Number of iterations")
                                ax3.set_ylabel("AUC")
                                ax3.title.set_text("AUC of the estimates")
                                ax3.legend(["Assisted learning estimator", "Oracle estimator"], loc = "lower right")
                        elif reportll:
                            # plot the evaluation results
                            fit, (ax1, ax2) = plt.subplots(1,2, figsize = (10, 8))
                            ax1.errorbar(np.array(range(kNum + 1)), EuDis_mean , EuDis_std, marker = '.', color = "b")
                            ax1.set_xlabel("Number of iterations")
                            ax1.set_ylabel("Euclidean distance")
                            ax1.title.set_text("Euclidean distance between the assisted \n learning estimates and the oracle estimates")

                            ax2.errorbar(np.array(range(kNum + 1)), ll_mean , ll_std, marker = '.', color = "b")
                            ax2.errorbar(np.array(range(kNum + 1)), ll_oracle_mean , ll_oracle_std, marker = '.', color = "r")
                            ax2.set_xlabel("Number of iterations")
                            ax2.set_ylabel("Negative log-likelihood")
                            ax2.title.set_text("The log-likelihood of the estimates")
                            ax2.legend(["Assisted learning estimator", "Oracle estimator"], loc = "lower right")
                        else:
                            # plot the evaluation results
                            fit, (ax1) = plt.subplots(1,1, figsize = (10, 8))
                            ax1.errorbar(np.array(range(kNum + 1)), EuDis_mean , EuDis_std, marker = '.', color = "b")
                            ax1.set_xlabel("Number of iterations")
                            ax1.set_ylabel("Euclidean distance")
                            ax1.title.set_text("Euclidean distance between the assisted \n learning estimates and the oracle estimates")



                        #fit.show()
                        plt.savefig(results_path_figures + "/" + fname + '_plot.pdf')

