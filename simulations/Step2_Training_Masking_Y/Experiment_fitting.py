#========================================================================================================================
# This module contains two functions: 
# experiment: single experiment
# repExperiment: independently repeat the same experiment multiple times. 
#========================================================================================================================
import statsmodels.api as sm
import numpy as np
from scipy.spatial import distance
from sklearn import metrics
from scipy.linalg import sqrtm
def experiment(pp, Model, randomBeta, Xdist, sig, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3):
    # calculate the total number of coefficients from each group
    p1 = p1_1 + p1_2 + p1_3
    p2 = p2_1 + p2_2 + p2_3
    p3 = p3_1 + p3_2 + p3_3
    #========================Generate data========================

    # obtain the coefficients
    if randomBeta == "False":
        beta1 = np.concatenate((np.repeat(v1_1, p1_1), np.repeat(v1_2, p1_2), np.repeat(v1_3, p1_3)))

        beta2 = np.concatenate((np.repeat(v2_1, p2_1), np.repeat(v2_2, p2_2), np.repeat(v2_3, p2_3)))
        
        beta3 = np.concatenate((np.repeat(v3_1, p3_1), np.repeat(v3_2, p3_2), np.repeat(v3_3, p3_3)))


        beta = np.concatenate((beta1, beta2, beta3))
    elif randomBeta == "True":
        # generate different data-generating model coefficients from normal distributions
        beta1 = np.concatenate((np.random.normal(loc = 0, scale = v1_1, size = p1_1), np.random.normal(loc = 0, scale = v1_2, size = p1_2), np.random.normal(loc = 0, scale = v1_3, size = p1_3)))

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


    #========================generate masked Y for assisted learning==================
    # current version only works for binary classification
    # when pp=0, there is no masking. when pp gets closer to 0.5, noise level gets larger
    Ymasked = np.array([])
    for Yobser in Y:
        if Yobser == 1:
            Ymasked = np.concatenate((Ymasked,np.random.binomial(1, (1 - pp), 1)))
        elif Yobser == 0:
            Ymasked = np.concatenate((Ymasked,np.random.binomial(1, pp, 1)))
        else:
            print("error")
    #print(np.unique((Ymasked - Y), return_counts=True))
    # assign the value of Ymasked to Y
    Y = Ymasked

    # if need to report AUC or ll, generate the evaluation data. Else, they are not needed for the Eculidean distance
    if reportAUC | reportll:
        #=====================================generate evaluation data=======================================
        # generate predictor data 
        XEval_tmp = np.random.normal(loc = 0, scale = 1, size = (nEval,(p1 + p2 + p3)))
        # transform XEval_tmp to have AR1 covariance
        XEval = np.matmul(XEval_tmp, mat_tmp4)
        # add the intercept
        XEvalintercept = sm.add_constant(XEval)

        # linear predictor 
        LpEvalVec = np.dot(XEval, beta)
        # generate the response data
        YEval = ResponseGenVec(LpEvalVec)



    # betaOracle: a 1-d np array of oracle model fitted coefficients
    # betaTemp: a 1-d np array of assisted learning model fitted coefficients
    # Y: predictor data
    # predCombined: prediction from the assisted learning model
    def eval(betaOracle, betaTemp, Y, predCombined, loglikeli):
        # calculate the Euclidean distance between the oracle model and assiste learning model fitted coefficients.
        EuDis = distance.euclidean(betaOracle, betaTemp)
        
        if reportAUC:
        #  calculate the AUC
            fpr, tpr, _ = metrics.roc_curve(Y, predCombined)
            AUC = metrics.auc(fpr, tpr)
        if reportll:
            # calculate the loglikelihood
            ll = loglikeli(Y, predCombined)

        if reportAUC:
            if reportll:
                return EuDis, AUC, ll
            else:
                return EuDis, AUC
        elif reportll:
            return EuDis, ll
        else:
            return EuDis
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
            while (gradL2 > thre) | (ct >100):
                grad = gradCal(Y, XAintercept, beta0)
                Hess = HessCal(Y, XAintercept, beta0)
                beta0 = beta0 + np.dot(np.linalg.inv(Hess), grad)
                grad_updated = gradCal(Y, XAintercept, beta0)
                ct = ct + 1
                gradL2 = np.mean(grad_updated**2)
            return beta0
        # add offset for assisted learning
        def gradCalAssisted(Y, XAintercept, offset, beta0):
            # get the residual
            resid = Y - np.dot(XAintercept, beta0) - offset
            # calculate the gradient
            grad = np.mean((np.tanh(logcoshA * resid).reshape(XAintercept.shape[0],-1)) * XAintercept, axis = 0)
            return(grad)
        def HessCalAssisted(Y, XAintercept, offset, beta0):
            # get the residual
            resid = Y - np.dot(XAintercept, beta0) - offset
            # calculate the Hessian
            Hess = logcoshA * n**(-1) * np.dot(np.dot(np.transpose(XAintercept), np.diag(np.cosh(logcoshA * resid)**(-2))), XAintercept)
            return(Hess)
        def fitLogCoshAssisted(Y, XAintercept, offset):
            # fit a linear regression as the initial value
            model = sm.GLM(endog = Y, exog = XAintercept, family = fitmodel)
            # get initial value
            beta0 = model.fit().params

            # evaluate the gradient
            grad_updated = gradCalAssisted(Y, XAintercept, offset, beta0)
            gradL2 = np.mean(grad_updated**2)

            # set convergence threthold
            thre = 1e-15
            # set counter
            ct = 0
            while (gradL2 > thre) | (ct >100):
                grad = gradCalAssisted(Y, XAintercept, offset, beta0)
                Hess = HessCalAssisted(Y, XAintercept, offset, beta0)
                beta0 = beta0 + np.dot(np.linalg.inv(Hess), grad)
                grad_updated = gradCalAssisted(Y, XAintercept, offset, beta0)
                ct = ct + 1
                gradL2 = np.mean(grad_updated**2)
            return beta0

    # add intercept terms that consist of 1 to the predictor datasets
    # X matrix with intercept
    Xintercept = sm.add_constant(X)

    # XA matrix with intercept
    XAintercept = sm.add_constant(XA)

    # XB matrix with intercept
    XBintercept = sm.add_constant(XB)

    #========================fit oracle model========================
    if Model == "logcosh":
        betaOracle = fitLogCosh(Y, Xintercept)
    else:
        oracle_model = sm.GLM(endog = Y, exog = Xintercept, family = fitmodel)
        oracle_results = oracle_model.fit()
        betaOracle = oracle_results.params

    # evaluation metrics for the oracle model
    if reportAUC | reportll:
        # linear predictor values from the initial value.
        lp_oracle = np.dot(XEvalintercept, betaOracle )

        # fitted probabilities
        predCombined_oracle_tmp =invLinkVec(lp_oracle)

        # calculate the prediction for the original probabilities
        predCombined_oracle =  (predCombined_oracle_tmp - pp)/(1 - 2*pp)

    # evaluation metrics for the oracle model


        

    if Model != "logcosh":
        # evaluate the performance of the oracle model
        if reportAUC:
            if reportll:
                _, AUC_oracle, ll_oracle = eval(betaOracle, betaOracle, YEval, predCombined_oracle, loglikeli)
            else:
                _, AUC_oracle = eval(betaOracle, betaOracle, YEval, predCombined_oracle, loglikeli)
        elif reportll:
            _, ll_oracle = eval(betaOracle, betaOracle, YEval, predCombined_oracle, loglikeli)
    #========================fit assisted model========================
    # obtain initial values
    # fit the model from A
    if Model == "logcosh":
        betafitted = fitLogCosh(Y, XAintercept)
    else:
        modelA = sm.GLM(endog = Y, exog = XAintercept, family = fitmodel)
        resultsA = modelA.fit()
        betafitted = resultsA.params
    # calculate the linear predictor
    lpA = np.dot(XAintercept, betafitted)

    #===============================Evaluation for the initial value=====================
    # obtain the estimated coefficients from the initial values. take 0 for those from B
    betaTemp = np.concatenate((betafitted[0:(p1+1)], np.repeat(0, p2), betafitted[(p1+1):(p1+p3+1)]))

    if reportAUC | reportll:
        # linear predictor values from the initial value.
        lp = np.dot(XEvalintercept, betaTemp)
        # calculate the fitted probabilities from the asissted learning model   
        predCombined_tmp =invLinkVec(lp)
        # calculate the prediction for the original probabilities
        predCombined =  (predCombined_tmp - pp)/(1 - 2*pp)



    # evaluate the performance of the initial value
    if reportAUC:
        if reportll:
            EuDis, AUC, ll = eval(betaOracle, betaTemp, YEval, predCombined, loglikeli)
        else:
            EuDis, AUC = eval(betaOracle, betaTemp, YEval, predCombined, loglikeli)
    elif reportll:
        EuDis, ll = eval(betaOracle, betaTemp, YEval, predCombined, loglikeli)
    else:
        EuDis = distance.euclidean(betaOracle, betaTemp)


    #store the evaluation results
    # a list that stores the Euclidean distance between the assisted learning beta and oracl beta
    EuDisList = [EuDis]

    # a list that stores the AUC values
    if reportAUC:
        AUCList = [AUC]

    # a list that stores the log-likelihoods
    if reportll:
        llList = [ll]

    # an array that stores betaTemp
    betaTempArray = betaTemp.reshape(1,-1)

    #===============================start fitting the assisted learning model with kNum iteration=====================
    for k in range(kNum):
        # B fits the model
        if Model == "logcosh":
            betafittedB = fitLogCoshAssisted(Y, XBintercept, lpA)
        else:
            modelB = sm.GLM(endog = Y, exog = XBintercept, offset =lpA, family = fitmodel)
            resultsB = modelB.fit()
            betafittedB = resultsB.params
        # calculate the linear predictor
        lpB = np.dot(XBintercept, betafittedB)

        # A fits the model
        if Model == "logcosh":
            betafittedA = fitLogCoshAssisted(Y, XAintercept, lpB)
        else:
            modelA = sm.GLM(endog = Y, exog = XAintercept, offset =lpB, family = fitmodel)
            resultsA = modelA.fit()
            betafittedA = resultsA.params
        # calculate the linear predictor
        lpA = np.dot(XAintercept, betafittedA)
        # combine the two coefficients from A and B
        betaTemp = np.concatenate(([betafittedA[0] + betafittedB[0]],betafittedA[1:(p1+1)], betafittedB[1:(p2+1)], betafittedA[(p1+1):(p1+p3+1)] + betafittedB[(p2+1):(p2+p3+1)]))
        betaTempArray = np.concatenate((betaTempArray, betaTemp.reshape(1,-1)), axis = 0)

        if reportAUC | reportll:
            # calculate the the linear predictor value
            lp = np.dot(XEvalintercept, betaTemp)
            # calculate the fitted probabilities from the asissted learning model   
            predCombined_tmp =invLinkVec(lp)
            # calculate the prediction for the original probabilities
            predCombined =  (predCombined_tmp - pp)/(1 - 2*pp)

        # evaluate the performance of the current model
            
        if reportAUC:
            if reportll:
                EuDis, AUC, ll = eval(betaOracle, betaTemp, YEval, predCombined, loglikeli)
            else:
                EuDis, AUC = eval(betaOracle, betaTemp, YEval, predCombined, loglikeli)
        elif reportll:
            EuDis, ll = eval(betaOracle, betaTemp, YEval, predCombined, loglikeli)
        else:
            EuDis = distance.euclidean(betaOracle, betaTemp)    

        if reportAUC:
            AUCList.append(AUC)
        if reportll:
            llList.append(ll)
        EuDisList.append(EuDis)
    # return the evaluation results
    if reportAUC:
        if reportll:
            return AUCList, EuDisList, llList, AUC_oracle, ll_oracle
        else:
            return AUCList, EuDisList, AUC_oracle
    elif reportll:
        return EuDisList, llList, ll_oracle
    else:
        return EuDisList


# function to repeat the experiment multiple times
def repExperiment(pp,Model,randomBeta, Xdist, sig, niter, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3):
    #store the evaluation results from multiple replications
    if reportAUC:
        AUC_array = np.empty((0, (kNum+1)))
        AUC_oracle_array = np.empty((0, (1)))
    if reportll:
        ll_oracle_array = np.empty((0, (1)))  
        ll_array = np.empty((0, (kNum+1)))   
    EuDis_array = np.empty((0, (kNum+1)))
    for j in range(niter):
        print("iter: " + str(j))
        # while True:
        #     try: 
        #         # in order to generate the same set of datasets, fix random seed of each iteration
        #         np.random.seed(j)
        #         # with AUC
        #         if reportAUC:
        #             AUCList, EuDisList, llList, AUC_oracle, ll_oracle  = experiment(batchsize, TrainPattern, mu, Q, etaFun, eta0, randomBeta, Xdist, sig, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
        #             AUC_array = np.concatenate((AUC_array, np.array(AUCList).reshape(1,-1)), axis = 0)
        #             AUC_oracle_array = np.concatenate((AUC_oracle_array, np.array(AUC_oracle).reshape(1,1) ))
        #         # without AUC
        #         else:
        #             EuDisList, llList, ll_oracle  = experiment(batchsize, TrainPattern, mu, Q, etaFun, eta0, randomBeta, Xdist,  sig, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
        #         break
        #     except BaseException:
        #         print('Errors encountered in model fitting, possibly due to numerical issues. Generate the dataset again')
        

        if reportAUC:
            if reportll:
                AUCList, EuDisList, llList, AUC_oracle, ll_oracle  = experiment(pp,Model,randomBeta, Xdist, sig, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
            else:
                AUCList, EuDisList, AUC_oracle  = experiment(pp,Model,randomBeta, Xdist, sig, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
        elif reportll:
            EuDisList, llList, ll_oracle  = experiment(pp,Model,randomBeta, Xdist, sig, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)
        else:
            EuDisList  = experiment(pp,Model,randomBeta, Xdist, sig, n, nEval, invLinkVec, loglikeli, ResponseGenVec, fitmodel, reportAUC, reportll, kNum, v1_1, v1_2, v1_3, v2_1, v2_2, v2_3, v3_1, v3_2, v3_3, p1_1, p1_2, p1_3, p2_1, p2_2, p2_3, p3_1, p3_2, p3_3)

        if reportAUC:
            AUC_array = np.concatenate((AUC_array, np.array(AUCList).reshape(1,-1)), axis = 0)
            AUC_oracle_array = np.concatenate((AUC_oracle_array, np.array(AUC_oracle).reshape(1,1) ))
        if reportll:
            ll_array  = np.concatenate((ll_array, np.array(llList).reshape(1,-1)), axis = 0)   
            ll_oracle_array = np.concatenate((ll_oracle_array, np.array(ll_oracle).reshape(1,1) ))
        EuDis_array = np.concatenate((EuDis_array, np.array(EuDisList).reshape(1,-1)), axis = 0)
    
    if reportAUC: 
        if reportll:
            return AUC_array, EuDis_array, ll_array, AUC_oracle_array, ll_oracle_array
        else:
            return AUC_array, EuDis_array, AUC_oracle_array
    elif reportll:
        return EuDis_array, ll_array, ll_oracle_array
    else: 
        return EuDis_array