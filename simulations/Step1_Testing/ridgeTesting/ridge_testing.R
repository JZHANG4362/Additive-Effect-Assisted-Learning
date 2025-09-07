#######################################################
# Apply the outputs from ridge to calculate the test statistic
# testing results under various settings
#######################################################
library(MASS)
library(expm)
library(VGAM)


datGen <- function(n, Xdist, sigMat, be1, be2, be3){
  # number of predictors from user 1
  p1 <- length(be1)
  # number of predictors from user 2
  p2 <- length(be2)
  
  # number of shared predictors
  p3 <- length(be3)
  
  # total number of predictors
  p <- p1 + p2 + p3
  # generate predictors data
  if (Xdist == "normal"){
    Xdat_tmp <- matrix(rnorm((n*p)), nrow = n, ncol = p)
  }else if (Xdist == "uniform"){
    Xdat_tmp <- matrix(runif((n*p)), nrow = n, ncol = p)
  }
  Xdat <- Xdat_tmp %*% sqrtm(sigMat)

  x1 <- Xdat[,c(1 : p1)]
  x2 <- Xdat[,c((p1+1) : (p1 + p2))]
  if (p3 != 0){
    x3 <- Xdat[,c((p1 + p2 + 1) : p)]
  }
  
  
  
  # generate response data (coefficients are all 1)
  if (p3 != 0){
    eta <-  as.numeric(x1 %*% be1 + x2%*% be2 + x3%*%be3)
  }else{
    eta <-  as.numeric(x1 %*% be1 + x2%*% be2)
  }
  y <- rnorm(n, mean = eta, sd = 1)
  
  
  
  if (p3 != 0){
  # full data set
  dat <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
  # data for user 1
  dat1 <- data.frame(y = y, x1 = x1, x3 = x3)
  # data for user 2
  dat2 <- data.frame(y = y, x2 = x2, x3 = x3)
  }else{
    # full data set
  dat <- data.frame(y = y, x1 = x1, x2 = x2)
  # data for user 1
  dat1 <- data.frame(y = y, x1 = x1)
  # data for user 2
  dat2 <- data.frame(y = y, x2 = x2)
  }

  

  
  return(list(dat = dat, dat1 = dat1, dat2 = dat2))
}
TestFun <- function(n, Lapscale, Umatrix, Xdist, sig, K, nK, be1, be2, be3, testSetting){
  
  # lengths of betas
  p1 <- length(be1)
  p2 <- length(be2)
  p3 <- length(be3)
  
  p <- p1 + p2 + p3
  
  # under H0 betas take fixed values
  # under H1 betas are randomly generated
  if (testSetting == "H1"){
    be1 <- mvrnorm(mu = be1, Sigma = diag(rep(1, length(be1))))
    be2 <- mvrnorm(mu = be2, Sigma = diag(rep(1, length(be2))))
    if (p3 != 0){
      be3 <- mvrnorm(mu = be3, Sigma = diag(rep(1, length(be3))))
    }
  }
  
  
  # covariance matrix of the predictors
  ar1_cor <- function(n, rho) {
    exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - 
                      (1:n - 1))
    rho^exponent
  }
  
  sigMat <- ar1_cor(p, sig)
  

  
  
  datList <- datGen(n = n, Xdist = Xdist, sigMat = sigMat, be1 = be1, be2 = be2, be3 = be3)
  
  dat <- datList$dat
  # data for user 1
  dat1 <- datList$dat1
  dat1_temp <- dat1
  dat1_temp$y <- NULL
  Xmat1 <- as.matrix(dat1_temp)
  # data for user 2
  dat2 <- datList$dat2
  dat2_temp <- dat2
  dat2_temp$y <- NULL
  Xmat2 <- as.matrix(dat2_temp)
  

  
  # initialize the data with dat1 only
  dat3 <- dat1
  
  # vectors to store the outputs
  statList <- numeric(K)
  pVList <- numeric(K)
  lamList <- numeric(K)
  
  for (j in c(1:K)){
    # generate random number to combine predictors from B
    uvec1 <- Umatrix[,j]
    # the data B send to A, add Laplace noise
    if (Lapscale != 0){
      udat1 <- data.frame(Xmat2 %*% uvec1) + rlaplace(n, location = 0, scale = Lapscale)
    }else if (Lapscale == 0)
      udat1 <- data.frame(Xmat2 %*% uvec1) 
    
    
    
    names(udat1) <- paste("U", j, sep = "")
    
    # predictors from A after receiving data from B
    dat3 <- data.frame(cbind(dat3, udat1))
    
    #================ calculate beta=================
    require(glmnet)
    lambMin <- cv.glmnet(as.matrix(dat3[,-1]), dat3$y, nfolds = nK, family = "gaussian", alpha = 0)$lambda.min
    mod <- glmnet(as.matrix(dat3[,-1]), dat3$y, family = "gaussian", lambda = lambMin, alpha = 0)
    coeff <- c(mod$a0, as.numeric(mod$beta))
    
    # calculate the fitted value
    ridge_fitted <- as.matrix(dat3[,-1]) %*% mod$beta + mod$a0
    
    ## check
    #predict(mod, newx = as.matrix(dat3[,-1]) ) - ridge_fitted
    
    resd2 <- as.vector((dat3$y - ridge_fitted )^2)
    
    # remove y and add the intercept
    datMat <- as.matrix(cbind(1, dat3[,-1]))
    #=============================Calculate the test statistic================
    V2 <- t(datMat) %*% diag(resd2) %*% datMat/n
    V1 <- t(datMat) %*%datMat/n
    
    V <- solve(V1) %*% V2 %*% solve(V1)
    Vt <- V[(2 + p1+p3):nrow(V1), (2 + p1+p3):ncol(V1)]
    
    betaU <- coeff[(2 + p1+p3):length(coeff)]
    
    if (j == 1){
      stat <- n * betaU * Vt^(-1) * betaU
    }else{
      stat <- n * t(betaU) %*% solve(Vt) %*% betaU
    }
    
    pV <- pchisq(stat, j, ncp = 0, lower.tail = FALSE, log.p = FALSE)
    
    statList[j] <- stat
    pVList[j] <- pV
    lamList[j] <- lambMin
    
    
  }
  result <- list(stat = statList, pV = pVList, lam = lamList)
  return(result)
}

# Covariance
sigList <- c(0.6, 0.9)

# patterns of data-generating coefficients
betaVec <- rbind(c(12,12,0),  c(4,16,4))

# size of the training data
#nList <- c(200, 2000)
nList <- c(2000)

# number of vectors to send
K <- 5

# number of CV folds in selecting lambda
nK <- 10
# number of replications
nrep <- 100

# distribution of predictors
Xdist = "uniform"

# whether fix a set of u for changing sample size n
fixedU = TRUE

# list of the scales of the Laplace noise to control local differential privacy
LapscaleList <- c(0.1, 0.5)

set.seed(20) 
for (nInd in seq_along(nList)){
  n <- nList[nInd]
  # setting of the test
  for (testSetting in c("H0", "H1")){
    # index of data-generating coefficients pattern
    for (betasetting in c(1:3)){
      for (sig in sigList){
        for (Lapscale in LapscaleList){
          # generate beta vectors
          be1 <- rep(1,betaVec[betasetting,1])
          if (testSetting == "H0"){
            be2 <- rep(0,betaVec[betasetting,2])
          }else if (testSetting == "H1"){
            be2 <- rep(1,betaVec[betasetting,2])
          } 
          be3 <- rep(1,betaVec[betasetting,3])
          
          # number of predictors from B
          p23 <- length(be2) + length(be3)
          
          # matrices to store results
          statMat <- matrix(nrow = nrep, ncol = K)
          pVMat <- matrix(nrow = nrep, ncol = K)
          lamMat <- matrix(nrow = nrep, ncol = K)
          
          if (fixedU == TRUE){
            load(paste("uArray_", p23, ".rda", sep=""))
          }
          for (i  in c(1:100)){
            message(i)
            if (fixedU == TRUE){
              # uArray is from the loaded file uArray_p23.rda
              Umatrix <- uArray[[i]]
            }else if (fixedU == FALSE){
              u_tmp <- matrix(rnorm((p23 * K)), nrow = p23, ncol = K)
              # standardize each column
              L2norm <- colNorms(u_tmp, method = "euclidean")
              Umatrix <- plyr::aaply(u_tmp, 1, "/", L2norm)
            }
            res <- TestFun(n=n, Lapscale=Lapscale, Umatrix=Umatrix, Xdist = Xdist, sig=sig, K=K, nK = nK, be1=be1, be2=be2, be3=be3, testSetting)
            statMat[i,] <- res$stat
            pVMat[i,] <- res$pV
            lamMat[i,] <- res$lam
          }
          #qqplot(pVMat[,1],runif(n))
          filename <- paste("testSetting_", testSetting, "_betasetting_", betasetting, "_n_", n, "_Xdist_",  Xdist, "_sig_", sig, "_fixedU_", fixedU, "_Lapscale_", Lapscale, sep="")
          write.csv(statMat, paste("data/testing/ridgeTestingStat_", filename, ".csv", sep=""))
          write.csv(pVMat, paste("data/testing/ridgeTestingPV_", filename, ".csv", sep=""))
          write.csv(lamMat, paste("data/testing/ridgeTestingLam_", filename, ".csv", sep=""))
        }
      }
      

    }
  }
}
