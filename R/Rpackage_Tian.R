##### Table of Contents:
# Function 1: Expit function
# Function 2: Constraint maximum likelihood (CML) method for logistic regression (binary outcome Y) 
# Function 3: Constraint maximum likelihood (CML) method for linear regression (continuous outcome Y) 
# Function 4: Asymptotic variance-covariance matrix for gamma_Int and gamma_CML for logistic regression (binary outcome Y)
# Function 5: Asymptotic variance-covariance matrix for gamma_Int and gamma_CML for linear regression (continuous outcome Y)
# Function 6: Calculate the empirical Bayes (EB) estimates
# Function 7: Using simulation to obtain the asymptotic variance-covariance matrix of gamma_EB, package corpcor and MASS are required
# Function 8: Obtain the proposed Optimal covariate-Weighted (OCW) estimates, package Rsolnp is required
# Function 9: Obtain the proposed Selective Coefficient-Learner (SC-Learner) estimates



####################################
########## Function 1: Expit function
###################################

#' Expit
#'
#' Standard expit function
#' 
#' y = exp(x) / (1 + exp(x))
#'
#' @param x data to transform
#' @return vector of transformed data
#' 
#' @export
expit <- function(x){
  y = exp(x)/(1+exp(x))
  return(y)
}


##############################################################################################################
########## Function 2: Constraint maximum likelihood (CML) method for logistic regression (binary outcome Y) 
##############################################################################################################
# Full Model: logit[Pr(Y=1|X,B)] = gamma_0 + gamma_1*X + gamma_2*B
# Reduced Model: logit[Pr(Y=1|X)] = beta_0 + beta_1*X
##############################################################################################################
# Reference: Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum likelihood 
# estimation for model calibration using summary-level information from external big data sources. 
# Journal of the American Statistical Association 111, 107-117.
##############################################################################################################
# p: total number of X covariates including the intercept (i.e. p=ncol(X)+1)
# q: total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
# YInt: Outcome vector
# XInt: X covariates that are used in the external models - Do not include intercept
# BInt: Newly added B covariates that are not included in the external models
# betaHatExt: External parameter estimates of the reduced model
# gammaHatInt: Full model parameter estimates using the internal data only
# n: internal data sample size
# tol: convergence criteria e.g. 1e-6
# maxIter: Maximum number of iterations to reach convergence e.g. 400
# factor: the step-halving factor between 0 and 1, if factor=1 then newton-raphson method; decrease if algorithm cannot converge given the maximum iterations
# Output gammaHat follows the order (intercept, XInt, BInt)


#' Constraint maximum likelihood (CML) method for logistic regression (binary outcome Y)
#'
#' @references Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). 
#' Constrained maximum likelihood estimation for model calibration using summary-level 
#' information from external big data sources. Journal of the American Statistical 
#' Association 111, 107-117.
#'
#' @param p total number of X covariates including the intercept (i.e. p=ncol(X)+1)
#' @param q total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
#' @param YInt Outcome vector
#' @param XInt X covariates that are used in the external models - Do not include intercept
#' @param BInt Newly added B covariates that are not included in the external models
#' @param betaHatExt External parameter estimates of the reduced model
#' @param gammaHatExt Full model parameter estimates using the internal data only
#' @param n internal data sample size
#' @param tol convergence criteria e.g. 1e-6
#' @param maxIter Maximum number of iterations to reach convergence e.g. 400
#' @param factor the step-halving factor between 0 and 1, if factor=1 then 
#' newton-raphson method; decrease if algorithm cannot converge given the 
#' maximum iterations
#' 
#' @return gammaHat, in the order (intercept, XInt, BInt)
#'
#' @examples 
#' # Full model: Y|X, B
#' # Reduced model: Y|X
#' # X,B follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X, B follows N(-1-0.5X+0.5B, 1)
#' set.seed(2333)
#' n = 1000
#' data.n = data.frame(matrix(ncol = 3, nrow = n))
#' colnames(data.n) = c('Y', 'X', 'B')
#' data.n[,c('X', 'B')] = MASS::mvrnorm(n, rep(0,2), diag(0.7,2)+0.3)
#' data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X + 0.5*data.n$B))
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = 30000
#' data.m = data.frame(matrix(ncol = 3, nrow = m))
#' names(data.m) = c('Y', 'X', 'B')
#' data.m[,c('X', 'B')] = MASS::mvrnorm(m, rep(0,2), diag(0.7,2)+0.3)
#' data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X + 0.5*data.m$B))
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E = glm(Y ~ X, data = data.m, family = binomial(link='logit'))
#' beta.E = coef(fit.E)
#' names(beta.E) = c('int', 'X')
#' V.E = vcov(fit.E)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = glm(Y ~ X + B, data = data.n, family = binomial(link='logit'))
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates
#' gamma.CML = fxnCC_LogReg(p=2, 
#'                          q=3, 
#'                          YInt=data.n$Y, 
#'                          XInt=data.n$X, 
#'                          BInt=data.n[,'B'], 
#'                          betaHatExt=beta.E, 
#'                          gammaHatInt=gamma.I, 
#'                          n=nrow(data.n), 
#'                          tol=1e-8, 
#'                          maxIter=400,
#'                          factor=1)[["gammaHat"]]
#'
#' @export
#'
fxnCC_LogReg <- function(p, q, YInt, XInt, BInt, betaHatExt, gammaHatInt, n, tol, maxIter, factor) {
  XiMatrix = cbind(1,XInt)
  XBiMatrix = cbind(XiMatrix,BInt)
  gammaHat = matrix(gammaHatInt,q,1)
  betaHatExt = matrix(betaHatExt,p,1)
  lambda = matrix(0,p,1)
  estDiff = 1
  counter = 0
  while(estDiff > tol & counter < maxIter) {
    v1 = v2 = 0
    h11 = matrix(0,q,q)
    h12 = matrix(0,p,q)
    h22 = matrix(0,p,p)
    for(i in 1:n) {
      Xi = matrix(XiMatrix[i,],p,1)
      XBi = matrix(XBiMatrix[i,],q,1)
      y = matrix(YInt[i],1,1)
      pInt = as.numeric(expit(t(XBi)%*%gammaHat))
      pExt = as.numeric(expit(t(Xi)%*%betaHatExt))
      diffP = as.numeric(pInt - pExt)
      pq = pInt*(1-pInt)
      lamTXi = as.numeric(t(lambda)%*%Xi)
      denom = as.numeric(n - lamTXi*diffP)
      v1 = v1 + as.numeric(y - pInt)*XBi + lamTXi*pq*XBi/denom
      v2 = v2 + diffP*Xi/denom
      xiC = as.numeric(denom*pq)
      h11 = h11 + -1*pq*XBi%*%t(XBi) + (xiC*(1-2*pInt)*lamTXi+pq^2*lamTXi^2)*XBi%*%t(XBi)/denom^2
      h12 = h12 + (xiC*Xi%*%t(XBi)+pq*diffP*lamTXi*Xi%*%t(XBi))/denom^2
      h22 = h22 + diffP^2*Xi%*%t(Xi)/denom^2
    }
    V = rbind(v1,v2)
    H = rbind(cbind(h11, t(h12)),cbind(h12, h22))
    pars = matrix(c(gammaHat,lambda),p+q,1)
    pars = pars - factor*solve(H, tol = 1e-1000)%*%V
    estDiff = max(abs(pars - matrix(c(gammaHat,lambda),p+q,1)))
    gammaHat = pars[1:q]
    lambda = pars[(q+1):(q+p)]
    counter = counter + 1
  }
  if(counter == 400) { stop("Convergence Note Met") }
  outList <- list(counter = counter, gammaHat = gammaHat, estDiff = estDiff)
  return(outList)
}

# ############### Example ##############
# # Full model: Y|X, B
# # Reduced model: Y|X
# # X,B follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X, B follows N(-1-0.5X+0.5B, 1)
# set.seed(2333)
# n = 1000
# data.n = data.frame(matrix(ncol = 3, nrow = n))
# colnames(data.n) = c('Y', 'X', 'B')
# data.n[,c('X', 'B')] = MASS::mvrnorm(n, rep(0,2), diag(0.7,2)+0.3)
# data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X + 0.5*data.n$B))
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = 30000
# data.m = data.frame(matrix(ncol = 3, nrow = m))
# names(data.m) = c('Y', 'X', 'B')
# data.m[,c('X', 'B')] = MASS::mvrnorm(m, rep(0,2), diag(0.7,2)+0.3)
# data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X + 0.5*data.m$B))
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E = glm(Y ~ X, data = data.m, family = binomial(link='logit'))
# beta.E = coef(fit.E)
# names(beta.E) = c('int', 'X')
# V.E = vcov(fit.E)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = glm(Y ~ X + B, data = data.n, family = binomial(link='logit'))
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates
# gamma.CML = fxnCC_LogReg(p=2, 
#                          q=3, 
#                          YInt=data.n$Y, 
#                          XInt=data.n$X, 
#                          BInt=data.n[,'B'], 
#                          betaHatExt=beta.E, 
#                          gammaHatInt=gamma.I, 
#                          n=nrow(data.n), 
#                          tol=1e-8, 
#                          maxIter=400,
#                          factor=1)[["gammaHat"]]
# ############### End of Example ##############






##############################################################################################################
########## Function 3: Constraint maximum likelihood (CML) method for linear regression (continuous outcome Y) 
##############################################################################################################
# Full Model: E[Y|X,B] = gamma_0 + gamma_1*X + gamma_2*B
# Reduced Model: E[Y|X] = beta_0 + beta_1*X
##############################################################################################################
# Reference: Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum likelihood 
# estimation for model calibration using summary-level information from external big data sources. 
# Journal of the American Statistical Association 111, 107-117.
##############################################################################################################
# Reference: Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum likelihood 
# estimation for model calibration using summary-level information from external big data sources. 
# Journal of the American Statistical Association 111, 107-117.
##############################################################################################################
# p: total number of X covariates including the intercept (i.e. p=ncol(X)+1)
# q: total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
# YInt: Outcome vector
# XInt: X covariates that are used in the external models - Do not include intercept
# BInt: Newly added B covariates that are not included in the external models
# betaHatExt: External parameter estimates of the reduced model
# gammaHatInt: Full model parameter estimates using the internal data only
# n: internal data sample size
# tol: convergence criteria e.g. 1e-6
# maxIter: Maximum number of iterations to reach convergence e.g. 400
# factor: the step-halving factor between 0 and 1, if factor=1 then newton-raphson method; decrease if algorithm cannot converge given the maximum iterations
# Output gammaHat follows the order (intercept, XInt, BInt)

#' Constraint maximum likelihood (CML) method for linear regression (continuous outcome Y) 
#'
#' @references Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). 
#' Constrained maximum likelihood estimation for model calibration using summary-level 
#' information from external big data sources. Journal of the American Statistical 
#' Association 111, 107-117.
#'
#' @param p total number of X covariates including the intercept (i.e. p=ncol(X)+1)
#' @param q total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
#' @param YInt Outcome vector
#' @param XInt X covariates that are used in the external models - Do not include intercept
#' @param BInt Newly added B covariates that are not included in the external models
#' @param betaHatExt External parameter estimates of the reduced model
#' @param gammaHatExt Full model parameter estimates using the internal data only
#' @param n internal data sample size
#' @param tol convergence criteria e.g. 1e-6
#' @param maxIter Maximum number of iterations to reach convergence e.g. 400
#' @param factor the step-halving factor between 0 and 1, if factor=1 then 
#' newton-raphson method; decrease if algorithm cannot converge given the 
#' maximum iterations
#' 
#' @return gammaHat, in the order (intercept, XInt, BInt)
#'
#' @examples 
#' # Full model: Y|X, B
#' # Reduced model: Y|X
#' # X,B follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X, B follows N(-1-0.5X+0.5B, 1)
#' set.seed(2333)
#' n = 800
#' data.n = data.frame(matrix(ncol = 3, nrow = n))
#' colnames(data.n) = c('Y', 'X', 'B')
#' data.n[,c('X', 'B')] = MASS::mvrnorm(n, rep(0,2), diag(0.7,2)+0.3)
#' data.n$Y = rnorm(n, -1 - 0.5*data.n$X + 0.5*data.n$B, 1)
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = 30000
#' data.m = data.frame(matrix(ncol = 3, nrow = m))
#' names(data.m) = c('Y', 'X', 'B')
#' data.m[,c('X', 'B')] = MASS::mvrnorm(m, rep(0,2), diag(0.7,2)+0.3)
#' data.m$Y = rnorm(m, -1 - 0.5*data.m$X + 0.5*data.m$B, 1)
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E = lm(Y ~ X, data = data.m)
#' beta.E = coef(fit.E)
#' names(beta.E) = c('int', 'X')
#' V.E = vcov(fit.E)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = lm(Y ~ X + B, data = data.n)
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates
#' gamma.CML = fxnCC_LinReg(p=2, 
#'                          q=3, 
#'                          YInt=data.n$Y, 
#'                          XInt=data.n[,'X'], 
#'                          BInt=data.n[,'B'], 
#'                          betaHatExt=beta.E, 
#'                          gammaHatInt=gamma.I, 
#'                          n=nrow(data.n), 
#'                          tol=1e-8, 
#'                          maxIter=400,
#'                          factor=1)[["gammaHat"]]
#'
#'
#'
#' @export
#' 
fxnCC_LinReg <- function(p, q, YInt, XInt, BInt, betaHatExt, gammaHatInt, n, tol, maxIter, factor) {
  XiMatrix = cbind(1,XInt)
  XBiMatrix = cbind(XiMatrix,BInt)
  gammaHat = matrix(gammaHatInt,q,1)
  lambda = matrix(0,p,1)
  sigma2 = 1/n*sum((YInt-XBiMatrix%*%gammaHat)^2)
  sigma2Ext = 1/n*sum((YInt-XiMatrix%*%betaHatExt)^2)
  estDiff = 1
  counter = 0
  while(estDiff > tol & counter < maxIter) {
    v1 = v2 = 0
    h11 = matrix(0,q,q)
    h12 = matrix(0,p,q)
    h22 = matrix(0,p,p)
    for(i in 1:n) {
      Xi = matrix(XiMatrix[i,],p,1)
      XBi = matrix(XBiMatrix[i,],q,1)
      y = matrix(YInt[i],1,1)
      resid = y-t(XBi)%*%gammaHat
      denom = as.numeric(n - (sigma2Ext)^(-1)*t(lambda)%*%Xi*(t(XBi)%*%gammaHat - t(Xi)%*%betaHatExt))
      v1 = v1 + 1/sigma2*XBi%*%resid + (sigma2Ext)^(-1)*XBi%*%t(Xi)%*%lambda/denom
      v2 = v2 + (sigma2Ext)^(-1)*Xi%*%(t(XBi)%*%gammaHat - t(Xi)%*%betaHatExt)/denom
      h11 = h11 + -1/sigma2*XBi%*%t(XBi) + (sigma2Ext)^(-2)*XBi%*%t(Xi)%*%lambda%*%t(XBi%*%t(Xi)%*%lambda)/denom^2
      h12 = h12 + (denom*(sigma2Ext)^(-1)*Xi%*%t(XBi)+as.numeric((sigma2Ext)^(-2)*(t(XBi)%*%gammaHat - t(Xi)%*%betaHatExt))*Xi%*%t(lambda)%*%Xi%*%t(XBi))/denom^2
      h22 = h22 + as.numeric((sigma2Ext)^(-2)*(t(XBi)%*%gammaHat - t(Xi)%*%betaHatExt)^2)*Xi%*%t(Xi)/denom^2
    }
    V = rbind(v1,v2)
    H = rbind(cbind(h11, t(h12)),cbind(h12, h22))
    pars = matrix(c(gammaHat,lambda),p+q,1)
    pars = pars - factor*solve(H, tol = 1e-1000)%*%V
    estDiff = max(abs(pars - matrix(c(gammaHat,lambda),p+q,1)))
    gammaHat = pars[1:q]
    lambda = pars[(q+1):(q+p)]
    counter = counter + 1
    sigma2 = 1/n*sum((YInt-XBiMatrix%*%gammaHat)^2)
  }
  if(counter == 800) { stop("Convergence Note Met") }
  outList <- list(counter = counter, gammaHat = gammaHat, sigma2 = sigma2, estDiff = estDiff)
  return(outList)
}

# ############### Example ##############
# # Full model: Y|X, B
# # Reduced model: Y|X
# # X,B follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X, B follows N(-1-0.5X+0.5B, 1)
# set.seed(2333)
# n = 800
# data.n = data.frame(matrix(ncol = 3, nrow = n))
# colnames(data.n) = c('Y', 'X', 'B')
# data.n[,c('X', 'B')] = MASS::mvrnorm(n, rep(0,2), diag(0.7,2)+0.3)
# data.n$Y = rnorm(n, -1 - 0.5*data.n$X + 0.5*data.n$B, 1)
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = 30000
# data.m = data.frame(matrix(ncol = 3, nrow = m))
# names(data.m) = c('Y', 'X', 'B')
# data.m[,c('X', 'B')] = MASS::mvrnorm(m, rep(0,2), diag(0.7,2)+0.3)
# data.m$Y = rnorm(m, -1 - 0.5*data.m$X + 0.5*data.m$B, 1)
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E = lm(Y ~ X, data = data.m)
# beta.E = coef(fit.E)
# names(beta.E) = c('int', 'X')
# V.E = vcov(fit.E)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = lm(Y ~ X + B, data = data.n)
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates
# gamma.CML = fxnCC_LinReg(p=2, 
#                          q=3, 
#                          YInt=data.n$Y, 
#                          XInt=data.n[,'X'], 
#                          BInt=data.n[,'B'], 
#                          betaHatExt=beta.E, 
#                          gammaHatInt=gamma.I, 
#                          n=nrow(data.n), 
#                          tol=1e-8, 
#                          maxIter=400,
#                          factor=1)[["gammaHat"]]
# ############### End of Example ##############







######################################################################################################################################
########## Function 4: Asymptotic variance-covariance matrix for gamma_Int and gamma_CML for logistic regression (binary outcome Y)
######################################################################################################################################
# Full Model: logit[Pr(Y=1|X,B)] = gamma_0 + gamma_1*X + gamma_2*B
# Reduced Model: logit[Pr(Y=1|X)] = beta_0 + beta_1*X
# gamma_I: Full model parameter estimates using the internal data only (MLE from direct regression)
# gamma_CML: Full model parameter estimates using the internal data and the external reduced model parameters (Chatterjee et al. 2016)
######################################################################################################################################
# Reference 1: Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum likelihood 
# estimation for model calibration using summary-level information from external big data sources. 
# Journal of the American Statistical Association 111, 107-117.
######################################################################################################################################
# Reference 2: Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction framework to integrate multiple regression 
# models into a current study. Manuscript in preparation.
######################################################################################################################################
# k: number of external models
# p: total number of all X covariates that is used at least once in the external model, including the intercept (i.e. p=ncol(X)+1)
# q: total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
# YInt: Outcome vector
# XInt: X covariates that are used in the external models - Do not include intercept
# BInt: Newly added B covariates that are not included in the external models
# gammaHatInt: Internal parameter estimates of the full model using the internal data
# betaHatExt_list: a list of k items, each item is a vector of the external parameter estimates (beta). Vector name is required for each 
#                  covariate, and has to be as consistent as the full model 
# CovExt_list: a list of k items, each item is the variance-covariance matrix of the external parameter estimates (beta) of the reduced model
# rho: a list of k items, each item is the sample size ratio, n/m (the internal sampel size n over the external sample size m)
# ExUncertainty: logic indicator, if TRUE then considering the external model uncertainty in the algorithm; if FALSE then ignoring the external model uncertainty

#' Asymptotic variance-covariance matrix for gamma_Int and gamma_CML for logistic regression (binary outcome Y)
#'
#' @references 
#' Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum 
#' likelihood estimation for model calibration using summary-level information from 
#' external big data sources. Journal of the American Statistical Association 111, 107-117.
#' 
#' Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction 
#' framework to integrate multiple regression models into a current study. 
#' Manuscript in preparation.
#' 
#' @param k number of external models
#' @param p total number of all X covariates that is used at least once in the 
#' external model, including the intercept (i.e. p=ncol(X)+1)
#' @param q total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
#' @param YInt Outcome vector
#' @param XInt X covariates that are used in the external models - Do not include 
#' intercept
#' @param BInt Newly added B covariates that are not included in the external models
#' @param gammaHatInt Internal parameter estimates of the full model using the 
#' internal data
#' @param betaHatExt_list a list of k items, each item is a vector of the external 
#' parameter estimates (beta). Vector name is required for each covariate, and has 
#' to be as consistent as the full model 
#' @param CovExt_list a list of k items, each item is the variance-covariance matrix 
#' of the external parameter estimates (beta) of the reduced model
#' @param rho a list of k items, each item is the sample size ratio, n/m (the 
#' internal sampel size n over the external sample size m)
#' @param ExUncertainty logic indicator, if TRUE then considering the external model 
#' uncertainty in the algorithm; if FALSE then ignoring the external model uncertaint
#' 
#' @examples 
#' # Full model: Y|X1, X2, B
#' # Reduced model 1: Y|X1 of sample size m1
#' # Reduced model 2: Y|X2 of sample size m2
#' # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
#' set.seed(2333)
#' n = 1000
#' data.n = data.frame(matrix(ncol = 4, nrow = n))
#' colnames(data.n) = c('Y', 'X1', 'X2', 'B')
#' data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
#' data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = m1 = m2 = 30000
#' data.m = data.frame(matrix(ncol = 4, nrow = m))
#' names(data.m) = c('Y', 'X1', 'X2', 'B')
#' data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
#' data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
#' fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
#' beta.E1 = coef(fit.E1)
#' beta.E2 = coef(fit.E2)
#' names(beta.E1) = c('int', 'X1')
#' names(beta.E2) = c('int', 'X2')
#' V.E1 = vcov(fit.E1)
#' V.E2 = vcov(fit.E2)
#' 
#' #Save all the external model information into lists for later use
#' betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
#' CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
#' rho = list(Ext1 = n/m1, Ext2 = n/m2)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
#' gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
#' gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
#' #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
#' gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
#' 
#' #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
#' asy.CML = asympVar_LogReg(k=2, 
#'                           p=2,
#'                           q=4, 
#'                           YInt=data.n$Y, 
#'                           XInt=data.n[,c('X1','X2')], #covariates that appeared in at least one external model
#'                           BInt=data.n$B,  #covariates that not used in any of the external models
#'                           gammaHatInt=gamma.I, 
#'                           betaHatExt_list=betaHatExt_list, 
#'                           CovExt_list=CovExt_list, 
#'                           rho=rho, 
#'                           ExUncertainty=TRUE)
#' asyV.I = asy.CML[['asyV.I']]                      #variance of gamma.I
#' asyV.CML1 = asy.CML[['asyV.CML']][[1]]            #variance of gamma.CML1
#' asyV.CML2 = asy.CML[['asyV.CML']][[2]]            #variance of gamma.CML2
#' asyCov.CML1.I = asy.CML[['asyCov.CML.I']][[1]]    #covariance of gamma.CML1 and gamma.I
#' asyCov.CML2.I = asy.CML[['asyCov.CML.I']][[2]]    #covariance of gamma.CML2 and gamma.I
#' asyCov.CML12 = asy.CML[['asyCov.CML']][['12']]    #covariance of gamma.CML1 and gamma.CML2
#' 
#' @return a list containing:
#' asyV.I: Variance of gamma_I (the direct regression parameter estimates using the internal data only)
#' asyV.CML: Variance of gamma_CML (the CML estiamtes [Chatterjee et al. 2016])
#' asyCov.CML: Covariance between two different CML estimates, gamma_CMLi and gamma_CMLj
#' asyCov.CML.I: Covariance between gamma_I and gamma_CML
#' ExtraTerm: the extra variance when ExUncertainty == TRUE (i.e. the external uncertainty is considered in the algorithm)
#'
#' @export
#'
asympVar_LogReg <- function(k, p, q, YInt, XInt, BInt, gammaHatInt, betaHatExt_list, CovExt_list, rho, ExUncertainty){
  n = length(YInt)
  if(is.null(names(XInt)) == TRUE){
    XiMatrix_all = as.matrix(cbind(int = 1, X = XInt), n, p)
  }else{ 
    XiMatrix_all = as.matrix(cbind(int = 1, XInt), n, p)
  }
  XBiMatrix = as.matrix(cbind(1,XInt,BInt), n, q)
  gammaHatInt = as.matrix(gammaHatInt, q, 1)
  
  ###########Calculate pExt = Pr(Y=1|Xs) for each external beta with different X's
  pExt_all = list()
  Xi_sub = list()
  for(i in 1:k){
    dim = length(betaHatExt_list[[i]])
    betaHatExt = matrix(betaHatExt_list[[i]], dim, 1)
    XiMatrix_Ext = XiMatrix_all[, colnames(XiMatrix_all) %in% names(betaHatExt_list[[i]])]
    pExt = expit(XiMatrix_Ext%*%betaHatExt)
    pExt_all[[i]] = pExt
    Xi_sub[[i]] = XiMatrix_Ext
  }
  
  ############calculate B, L1, L2, L12, C1, C2, C12, Q matrix
  B = 0
  C = vector('list', k) #C is a list of k vectors, one for each external beta
  Q = vector('list', k) #Q is a list of k vectors, one for each external beta, similar to C
  L = list() #L is similar to an upper diagnal matrix, with (1+k)*k/2 items
  for(i in 1:n) {
    XBi = XBiMatrix[i,]
    y = matrix(YInt[i],1,1)
    pInt = as.numeric(expit(t(XBi)%*%gammaHatInt))
    B = B + XBi %*% t(XBi) * pInt * (1-pInt) 
    #initialize the matrix of C & L when i=1
    if(i==1){
      for(a in 1:k){
        C[[a]] = XBi %*% t(Xi_sub[[a]][i,]) * pInt * (1-pInt) #C = C + XBi %*% t(Xi) * pInt * (1-pInt)
        Q[[a]] = -Xi_sub[[a]][i,] %*% t(Xi_sub[[a]][i,]) * pInt * (1-pInt) #Q = Q - Xi %*% t(Xi) * pInt * (1-pInt)
        for(b in 1:k){ if(a <= b){ #Because of symmetry, only calculate the upper diagnal matrix
          name = paste(a,b,sep='')
          L[[name]] = (pInt - pExt_all[[a]][i])*Xi_sub[[a]][i,] %*% t((pInt - pExt_all[[b]][i])*Xi_sub[[b]][i,]) #L = L + (pInt - pExt)*Xi %*% t((pInt - pExt)*Xi)
        }
        }
      }
    }
    else{ #add on the following iteration result
      for(a in 1:k){
        C[[a]] = C[[a]] + XBi %*% t(Xi_sub[[a]][i,]) * pInt * (1-pInt)
        Q[[a]] = Q[[a]] - Xi_sub[[a]][i,] %*% t(Xi_sub[[a]][i,]) * pInt * (1-pInt)
        for(b in 1:k){ if(a <= b){
          name = paste(a,b,sep='')
          L[[name]] = L[[name]] + (pInt - pExt_all[[a]][i])*Xi_sub[[a]][i,] %*% t((pInt - pExt_all[[b]][i])*Xi_sub[[b]][i,])
        }
        }
      }
    }
  }
  
  #############Use B, L1, L2, L12, C1, C2, C12 to get the Cov(CML)
  asyV.I = round(solve(B), 5)
  asyV.CML = asyCov.CML.I = ExtraTerm = vector('list', k)
  asyCov.CML = list() ######upper diagnal minus diagnal = (1+k)*k/2 - k
  for(a in 1:k){
    name = paste(a,a,sep='')
    if(ExUncertainty == TRUE){
      ###### Var.CML = 1/(B+C*L^-1*tC) * ( 1 + rho/n * (C/L)*(QVQ)*(C/L)/(B+C*C/L))
      tmp = solve(B+C[[a]]%*%round(solve(L[[name]]),5)%*%t(C[[a]]))
      QVQ = t(Q[[a]])%*%CovExt_list[[a]]%*%Q[[a]]
      asyV.CML[[a]] = round(tmp + (1/n)*rho[[a]]* tmp %*% C[[a]]%*%solve(L[[name]]) %*% QVQ %*% t(solve(L[[name]]))%*%t(C[[a]]) %*% tmp, 5)
      ExtraTerm[[a]] = round((1/n)*rho[[a]]* tmp %*% C[[a]]%*%solve(L[[name]])%*%QVQ%*%t(solve(L[[name]]))%*%t(C[[a]]) %*% tmp, 5)
      asyCov.CML.I[[a]] = round(tmp,5)
    }else if(ExUncertainty == FALSE){
      ###### Var.CML = 1/(B+C*L^-1*tC) 
      asyV.CML[[a]] = asyCov.CML.I[[a]] = round(solve(B+C[[a]]%*%round(solve(L[[name]]),5)%*%t(C[[a]])), 5)
      ExtraTerm[[a]] = matrix(0,q,q)
    }
    for(b in 1:k){
      if(a < b){
        name1 = paste(a,a,sep='')
        name2 = paste(b,b,sep='')
        name12 = paste(a,b,sep='')
        
        L1 = L[[name1]]
        L2 = L[[name2]]
        L12 = L[[name12]]
        C1 = C[[a]]
        C2 = C[[b]]
        
        #######Cov(CMLi,CMLj) = [B+C1*(solve(L1)*L12*solve(L2))*C2] / [(B+C1*C1/L1)*(B+C2*C2/L2)]
        asyCov.CML[[name12]] = round(solve((B+C1%*%solve(L1)%*%t(C1))) %*% ((B + C1%*% (solve(L1)%*%L12%*%solve(L2)) %*%t(C2))) %*% solve((B+C2%*%solve(L2)%*%t(C2))), 5)
      }
    }
  }
  
  #############Return following values
  # asyV.I: Variance of gamma_I (the direct regression parameter estimates using the internal data only)
  # asyV.CML: Variance of gamma_CML (the CML estiamtes [Chatterjee et al. 2016])
  # asyCov.CML: Covariance between two different CML estimates, gamma_CMLi and gamma_CMLj
  # asyCov.CML.I: Covariance between gamma_I and gamma_CML
  # ExtraTerm: the extra variance when ExUncertainty == TRUE (i.e. the external uncertainty is considered in the algorithm)
  return(list(asyV.I = asyV.I, asyV.CML = asyV.CML, asyCov.CML = asyCov.CML, asyCov.CML.I = asyCov.CML.I, ExtraTerm = ExtraTerm))
}

# ############### Example ##############
# # Full model: Y|X1, X2, B
# # Reduced model 1: Y|X1 of sample size m1
# # Reduced model 2: Y|X2 of sample size m2
# # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
# set.seed(2333)
# n = 1000
# data.n = data.frame(matrix(ncol = 4, nrow = n))
# colnames(data.n) = c('Y', 'X1', 'X2', 'B')
# data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
# data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = m1 = m2 = 30000
# data.m = data.frame(matrix(ncol = 4, nrow = m))
# names(data.m) = c('Y', 'X1', 'X2', 'B')
# data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
# data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
# fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
# beta.E1 = coef(fit.E1)
# beta.E2 = coef(fit.E2)
# names(beta.E1) = c('int', 'X1')
# names(beta.E2) = c('int', 'X2')
# V.E1 = vcov(fit.E1)
# V.E2 = vcov(fit.E2)
# 
# #Save all the external model information into lists for later use
# betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
# CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
# rho = list(Ext1 = n/m1, Ext2 = n/m2)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
# gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
# gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
# #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
# gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
# 
# #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
# asy.CML = asympVar_LogReg(k=2, 
#                           p=2,
#                           q=4, 
#                           YInt=data.n$Y, 
#                           XInt=data.n[,c('X1','X2')], #covariates that appeared in at least one external model
#                           BInt=data.n$B,  #covariates that not used in any of the external models
#                           gammaHatInt=gamma.I, 
#                           betaHatExt_list=betaHatExt_list, 
#                           CovExt_list=CovExt_list, 
#                           rho=rho, 
#                           ExUncertainty=TRUE)
# asyV.I = asy.CML[['asyV.I']]                      #variance of gamma.I
# asyV.CML1 = asy.CML[['asyV.CML']][[1]]            #variance of gamma.CML1
# asyV.CML2 = asy.CML[['asyV.CML']][[2]]            #variance of gamma.CML2
# asyCov.CML1.I = asy.CML[['asyCov.CML.I']][[1]]    #covariance of gamma.CML1 and gamma.I
# asyCov.CML2.I = asy.CML[['asyCov.CML.I']][[2]]    #covariance of gamma.CML2 and gamma.I
# asyCov.CML12 = asy.CML[['asyCov.CML']][['12']]    #covariance of gamma.CML1 and gamma.CML2
# ############### End of Example ##############






########################################################################################################################
########## Function 5: Asymptotic variance-covariance matrix for gamma_Int and gamma_CML for linear regression (continuous outcome Y)
########################################################################################################################
# Full Model: E(Y|X,B) = gamma_0 + gamma_1*X + gamma_2*B
# Reduced Model: E(Y|X) = beta_0 + beta_1*X
# gamma_I: Full model parameter estimates using the internal data only (MLE from direct regression)
# gamma_CML: Full model parameter estimates using the internal data and the external reduced model parameters (Chatterjee et al. 2016)
######################################################################################################################################
# Reference 1: Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum likelihood 
# estimation for model calibration using summary-level information from external big data sources. 
# Journal of the American Statistical Association 111, 107-117.
######################################################################################################################################
# Reference 2: Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction framework to integrate multiple regression 
# models into a current study. Manuscript in preparation.
######################################################################################################################################
# k: number of external models
# p: total number of X covariates including the intercept (i.e. p=ncol(X)+1)
# q: total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
# YInt: Outcome vector
# XInt: X covariates that are used in the external models - Do not include intercept
# BInt: Newly added B covariates that are not included in the external models
# gammaHatInt: Internal parameter estimates of the full model using the internal data
# betaHatExt_list: a list of k items, each item is a vector of the external parameter estimates (beta). Vector name is required for each 
#                  covariate, and has to be as consistent as the full model 
# CovExt_list: a list of k items, each item is the variance-covariance matrix of the external parameter estimates (beta) of the reduced model
# rho: a list of k items, each item is the sample size ratio, n/m (the internal sampel size n over the external sample size m)
# ExUncertainty: logic indicator, if TRUE then considering the external model uncertainty in the algorithm; if FALSE then ignoring the external model uncertainty

#' Asymptotic variance-covariance matrix for gamma_Int and gamma_CML for linear regression (continuous outcome Y)
#'
#' @references 
#' Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum 
#' likelihood estimation for model calibration using summary-level information from 
#' external big data sources. Journal of the American Statistical Association 111, 107-117.
#' 
#' Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction 
#' framework to integrate multiple regression models into a current study. 
#' Manuscript in preparation.
#'
#' @param k number of external models
#' @param p total number of X covariates including the intercept (i.e. p=ncol(X)+1)
#' @param q total number of covariates including the intercept (i.e. q=ncol(X)+ncol(B)+1)
#' @param YInt Outcome vector
#' @param XInt X covariates that are used in the external models - Do not include intercept
#' @param BInt Newly added B covariates that are not included in the external models
#' @param gammaHatInt Internal parameter estimates of the full model using the internal data
#' @param betaHatExt_list a list of k items, each item is a vector of the external 
#' parameter estimates (beta). Vector name is required for each covariate, and has to be as consistent as the full model 
#' @param CovExt_list a list of k items, each item is the variance-covariance matrix 
#' of the external parameter estimates (beta) of the reduced model
#' @param rho a list of k items, each item is the sample size ratio, n/m (the 
#' internal sampel size n over the external sample size m)
#' @param ExUncertainty logic indicator, if TRUE then considering the external 
#' model uncertainty in the algorithm; if FALSE then ignoring the external model uncertainty
#'
#' @return a list containing:
#' asyV.I: Variance of gamma_I (the direct regression parameter estimates using the internal data only)
#' asyV.CML: Variance of gamma_CML (the CML estiamtes [Chatterjee et al. 2016])
#' asyCov.CML: Covariance between two different CML estimates, gamma_CMLi and gamma_CMLj
#' asyCov.CML.I: Covariance between gamma_I and gamma_CML
#' ExtraTerm: the extra variance when ExUncertainty == TRUE (i.e. the external uncertainty is considered in the algorithm)
#'
#' @examples 
#' # Full model: Y|X1, X2, B
#' # Reduced model 1: Y|X1 of sample size m1
#' # Reduced model 2: Y|X2 of sample size m2
#' # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X1, X2, B follows N(-1-0.5*X1-0.5*X2+0.5*B, 1)
#' set.seed(2333)
#' n = 1000
#' data.n = data.frame(matrix(ncol = 4, nrow = n))
#' colnames(data.n) = c('Y', 'X1', 'X2', 'B')
#' data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
#' data.n$Y = rnorm(n, -1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B, 1)
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = m1 = m2 = 30000
#' data.m = data.frame(matrix(ncol = 4, nrow = m))
#' names(data.m) = c('Y', 'X1', 'X2', 'B')
#' data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
#' data.m$Y = rnorm(m, -1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B, 1)
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E1 = lm(Y ~ X1, data = data.m)
#' fit.E2 = lm(Y ~ X2, data = data.m)
#' beta.E1 = coef(fit.E1)
#' beta.E2 = coef(fit.E2)
#' names(beta.E1) = c('int', 'X1')
#' names(beta.E2) = c('int', 'X2')
#' V.E1 = vcov(fit.E1)
#' V.E2 = vcov(fit.E2)
#' 
#' #Save all the external model information into lists for later use
#' betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
#' CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
#' rho = list(Ext1 = n/m1, Ext2 = n/m2)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = lm(Y ~ X1 + X2 + B, data = data.n)
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
#' gamma.CML1 = fxnCC_LinReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
#' gamma.CML2 = fxnCC_LinReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
#' #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
#' gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
#' 
#' #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
#' asy.CML = asympVar_LinReg(k=2, 
#'                           p=2,
#'                           q=4, 
#'                           YInt=data.n$Y, 
#'                           XInt=data.n[,c('X1','X2')], #covariates that appeared in at least one external model
#'                           BInt=data.n$B,  #covariates that not used in any of the external models
#'                           gammaHatInt=gamma.I, 
#'                           betaHatExt_list=betaHatExt_list, 
#'                           CovExt_list=CovExt_list, 
#'                           rho=rho, 
#'                           ExUncertainty=TRUE)
#' asyV.I = asy.CML[['asyV.I']]                      #variance of gamma.I
#' asyV.CML1 = asy.CML[['asyV.CML']][[1]]            #variance of gamma.CML1
#' asyV.CML2 = asy.CML[['asyV.CML']][[2]]            #variance of gamma.CML2
#' asyCov.CML1.I = asy.CML[['asyCov.CML.I']][[1]]    #covariance of gamma.CML1 and gamma.I
#' asyCov.CML2.I = asy.CML[['asyCov.CML.I']][[2]]    #covariance of gamma.CML2 and gamma.I
#' asyCov.CML12 = asy.CML[['asyCov.CML']][['12']]    #covariance of gamma.CML1 and gamma.CML2
#' 
#'
asympVar_LinReg <- function(k, p, q, YInt, XInt, BInt, gammaHatInt, betaHatExt_list, CovExt_list, rho, ExUncertainty){
  if(is.null(names(XInt)) == TRUE){
    XiMatrix_all = as.matrix(cbind(int = 1, X = XInt), n, p)
  }else{ 
    XiMatrix_all = as.matrix(cbind(int = 1, XInt), n, p)
  }
  XBiMatrix = as.matrix(cbind(1,XInt,BInt), n, q)
  gammaHatInt = as.matrix(gammaHatInt, q, 1)
  yhatIntMatrix = XBiMatrix %*% gammaHatInt
  sigma2Int = 1/n*sum((YInt-yhatIntMatrix)^2)
  
  ############Calculate sigma2Ext = 1/n * sum(Y-X*betaExt)^2 for each external beta with different X's
  sigma2Ext_all = list()
  yhatExt_all = list()
  Xi_sub = list()
  for(a in 1:k){
    dim = length(betaHatExt_list[[a]])
    betaHatExt = matrix(betaHatExt_list[[a]], dim, 1)
    XiMatrix_Ext = XiMatrix_all[, colnames(XiMatrix_all) %in% names(betaHatExt_list[[a]])]
    yhatExt = XiMatrix_Ext%*%betaHatExt
    sigma2Ext = 1/n*sum((YInt-yhatExt)^2)
    Xi_sub[[a]] = XiMatrix_Ext
    sigma2Ext_all[[a]] = sigma2Ext
    yhatExt_all[[a]] = yhatExt
  }
  
  #############calculate B, L1, L2, L12, C1, C2, C12 matrix
  B = 0
  C = vector('list', k) #C is a list of k vectors, one for each external beta
  Q = vector('list', k) #Q is a list of k vectors, one for each external beta, similar to C
  L = list() #L is similar to an upper diagnal matrix, with (1+k)*k/2 items
  for(i in 1:n) {
    XBi = XBiMatrix[i,]
    y = matrix(YInt[i],1,1)
    yhatInt = yhatIntMatrix[i,]
    B = B + XBi %*% t(XBi) / sigma2Int #B = B + XBi %*% t(XBi) / sigma2Int
    #####initialize the matrix of C & L when i=1
    if(i==1){
      for(a in 1:k){
        yhatExt_a = yhatExt_all[[a]][i]
        Xi_a = Xi_sub[[a]][i,]
        sigma2Ext_a = sigma2Ext_all[[a]]
        C[[a]] = XBi %*% t(Xi_a) / sigma2Ext_a #C = XBi %*% t(Xi) / sigma2Ext
        Q[[a]] = - Xi_sub[[a]][i,] %*% t(Xi_sub[[a]][i,]) / sigma2Ext_a #Q = - Xi %*% t(Xi) / sigma2Ext
        ######Because of symmetry, only calculate the upper diagnal matrix
        for(b in 1:k){ 
          if(a <= b){
            name = paste(a,b,sep='')
            yhatExt_b = yhatExt_all[[b]][i]
            Xi_b = Xi_sub[[b]][i,]
            sigma2Ext_b = sigma2Ext_all[[b]]
            L[[name]] = round((yhatInt - yhatExt_a)*Xi_a %*% t((yhatInt - yhatExt_b)*Xi_b) / (sigma2Ext_a*sigma2Ext_b), 5) #L = L + (XB*gammaHatInt - X*betaExt)*Xi %*% t((XB*gammaHatInt - X*betaExt)*Xi) / (sigma2Ext)^2
          }
        }
      }
    }
    #####add on the following iteration result
    else{ 
      for(a in 1:k){
        yhatExt_a = yhatExt_all[[a]][i]
        Xi_a = Xi_sub[[a]][i,]
        sigma2Ext_a = sigma2Ext_all[[a]]
        C[[a]] = C[[a]] + XBi %*% t(Xi_a) / sigma2Ext_a #C = C + XBi %*% t(Xi) / sigma2Ext
        Q[[a]] = Q[[a]] - Xi_sub[[a]][i,] %*% t(Xi_sub[[a]][i,]) / sigma2Ext_a #Q = Q - Xi %*% t(Xi) / sigma2Ext
        for(b in 1:k){ 
          if(a <= b){
            name = paste(a,b,sep='')
            yhatExt_b = yhatExt_all[[b]][i]
            Xi_b = Xi_sub[[b]][i,]
            sigma2Ext_b = sigma2Ext_all[[b]]
            L[[name]] = L[[name]] + round((yhatInt - yhatExt_a)*(yhatInt - yhatExt_b)* Xi_a %*% t(Xi_b) / (sigma2Ext_a*sigma2Ext_b), 5)
          }
        }
      }
    }
  }

  #############Use B, L1, L2, L12, C1, C2, C12 to get the Cov(CML)
  asyV.I = round(solve(B/n) / n, 5)
  asyV.CML = asyCov.CML.I = ExtraTerm = vector('list', k)
  asyCov.CML = list() #upper diagnal minus diagnal = (1+k)*k/2 - k
  for(a in 1:k){
    name = paste(a,a,sep='')
    B = round(B, 5)
    if(ExUncertainty == TRUE){
      ######Var.CMLi = 1/(B+Ci*Ci/Li) * ( 1 + rho * (Ci/Li)*(QVQ)*(Ci/Li)/(B+Ci*Ci/Li))
      tmp = solve(B + C[[a]] %*% round(solve(L[[name]]),5) %*% t(C[[a]]))
      asyCov.CML.I[[a]] = round(tmp,5)
      QVQ = Q[[a]] %*% CovExt_list[[a]] %*% t(Q[[a]])
      asyV.CML[[a]] = round(tmp %*% (diag(1,q) + (1/n)*rho[[a]]*(C[[a]] %*% solve(L[[name]]) %*% QVQ %*% t(solve(L[[name]])) %*% t(C[[a]])) %*% tmp), 5)
      ExtraTerm[[a]] = round(tmp %*% ((1/n)*rho[[a]]*C[[a]] %*% solve(L[[name]]) %*% QVQ %*% t(solve(L[[name]])) %*% t(C[[a]]) %*% tmp), 5)
    }else if(ExUncertainty == FALSE){
      ######Var.CMLi = 1/(B+Ci*Ci/Li) 
      asyV.CML[[a]] = asyCov.CML.I[[a]] = round(solve(B + C[[a]] %*% round(solve(L[[name]]), 5) %*% t(C[[a]])), 5)
      ExtraTerm[[a]] = matrix(0, q, q)
    }
    for(b in 1:k){
      if(a < b){
        name1 = paste(a,a,sep='')
        name2 = paste(b,b,sep='')
        name12 = paste(a,b,sep='')
        
        L1 = L[[name1]]
        L2 = L[[name2]]
        L12 = L[[name12]]
        C1 = C[[a]]
        C2 = C[[b]]
        
        #######Cov(CMLi,CMLj) = [B+C1*(solve(L1)*L12*solve(2))*C2] / [(B+C1*C1/L1)*(B+C2*C2/L2)]
        asyCov.CML[[name12]] =  round(solve((B+C1%*%solve(L1)%*%t(C1))/n) %*% round(((B + C1%*% (round(solve(L1),5)%*%L12%*%round(solve(L2),5)) %*%t(C2))/n), 5) %*% round(solve((B+C2%*%solve(L2)%*%t(C2))/n) / n, 5), 5)
      }
    }
  }
  
  #############Return following values
  # asyV.I: Variance of gamma_I (the direct regression parameter estimates using the internal data only)
  # asyV.CML: Variance of gamma_CML (the CML estiamtes [Chatterjee et al. 2016])
  # asyCov.CML: Covariance between two different CML estimates, gamma_CMLi and gamma_CMLj
  # asyCov.CML.I: Covariance between gamma_I and gamma_CML
  # ExtraTerm: the extra variance when ExUncertainty == TRUE (i.e. the external uncertainty is considered in the algorithm)
  return(list(asyV.I = asyV.I, asyV.CML = asyV.CML, asyCov.CML = asyCov.CML, asyCov.CML.I = asyCov.CML.I, ExtraTerm = ExtraTerm))
}



# ############### Example ##############
# # Full model: Y|X1, X2, B
# # Reduced model 1: Y|X1 of sample size m1
# # Reduced model 2: Y|X2 of sample size m2
# # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X1, X2, B follows N(-1-0.5*X1-0.5*X2+0.5*B, 1)
# set.seed(2333)
# n = 1000
# data.n = data.frame(matrix(ncol = 4, nrow = n))
# colnames(data.n) = c('Y', 'X1', 'X2', 'B')
# data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
# data.n$Y = rnorm(n, -1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B, 1)
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = m1 = m2 = 30000
# data.m = data.frame(matrix(ncol = 4, nrow = m))
# names(data.m) = c('Y', 'X1', 'X2', 'B')
# data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
# data.m$Y = rnorm(m, -1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B, 1)
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E1 = lm(Y ~ X1, data = data.m)
# fit.E2 = lm(Y ~ X2, data = data.m)
# beta.E1 = coef(fit.E1)
# beta.E2 = coef(fit.E2)
# names(beta.E1) = c('int', 'X1')
# names(beta.E2) = c('int', 'X2')
# V.E1 = vcov(fit.E1)
# V.E2 = vcov(fit.E2)
# 
# #Save all the external model information into lists for later use
# betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
# CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
# rho = list(Ext1 = n/m1, Ext2 = n/m2)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = lm(Y ~ X1 + X2 + B, data = data.n)
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
# gamma.CML1 = fxnCC_LinReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
# gamma.CML2 = fxnCC_LinReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
# #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
# gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
# 
# #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
# asy.CML = asympVar_LinReg(k=2, 
#                           p=2,
#                           q=4, 
#                           YInt=data.n$Y, 
#                           XInt=data.n[,c('X1','X2')], #covariates that appeared in at least one external model
#                           BInt=data.n$B,  #covariates that not used in any of the external models
#                           gammaHatInt=gamma.I, 
#                           betaHatExt_list=betaHatExt_list, 
#                           CovExt_list=CovExt_list, 
#                           rho=rho, 
#                           ExUncertainty=TRUE)
# asyV.I = asy.CML[['asyV.I']]                      #variance of gamma.I
# asyV.CML1 = asy.CML[['asyV.CML']][[1]]            #variance of gamma.CML1
# asyV.CML2 = asy.CML[['asyV.CML']][[2]]            #variance of gamma.CML2
# asyCov.CML1.I = asy.CML[['asyCov.CML.I']][[1]]    #covariance of gamma.CML1 and gamma.I
# asyCov.CML2.I = asy.CML[['asyCov.CML.I']][[2]]    #covariance of gamma.CML2 and gamma.I
# asyCov.CML12 = asy.CML[['asyCov.CML']][['12']]    #covariance of gamma.CML1 and gamma.CML2
# ############### End of Example ##############








########################################################################################################################################
###### Function 6: Calculate the empirical Bayes (EB) estimates
########################################################################################################################################
# Full Model: E(Y|X,B) = gamma_0 + gamma_1*X + gamma_2*B
# gamma_I: Full model parameter estimates using the internal data only (MLE from direct regression)
# gamma_CML: Full model parameter estimates using the internal data and the external reduced model parameters (Chatterjee et al. 2016)
# gamma_EB: The empircal Bayes estimate of the full model (i.e. a weighted average of gamma_I and gamma_CML) (Estes et al. 2017)
########################################################################################################################################
# Reference 1: Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum likelihood estimation for model
# calibration using summary-level information from external big data sources. Journal of the American Statistical Association 111, 107-117.
########################################################################################################################################
# Reference 2: Estes, J.P., Mukherjee, B. and Taylor, J.M.G. (2017). Empirical bayes estimation and prediction using summary-level
# information from external big data sources adjusting for violations of transportability. Statistics in Biosciences 10, 568-586.
########################################################################################################################################
# gamma_I: Full model parameter estimates using the internal data only (MLE from direct regression)
# gamma_CML: Full model parameter estimates using the internal data and the external reduced model parameters (Chatterjee et al. 2016)
# asyV.I: Variance-covariance matrix of gamma_I from function asympVar_LinReg[['asyV.I']] or asympVar_LogReg[['asyV.I']]

#' Calculate the empirical Bayes (EB) estimates
#' 
#' @param gamma_I Full model parameter estimates using the internal data only (MLE from direct regression)
#' @param gamma_CML Full model parameter estimates using the internal data and the external reduced model parameters (Chatterjee et al. 2016)
#' @param asyV.I Variance-covariance matrix of gamma_I from function asympVar_LinReg[['asyV.I']] or asympVar_LogReg[['asyV.I']]
#'
#' @return a list with:
#' gamma_I: Full model parameter estimates using the internal data only (MLE from direct regression)
#' gamma_CML: Full model parameter estimates using the internal data and the external reduced model parameters (Chatterjee et al. 2016)
#' gamma_EB: The empircal Bayes estimate of the full model (i.e. a weighted average of gamma_I and gamma_CML) (Estes et al. 2017)
#' 
#' @references 
#' #' Chatterjee, N., Chen, Y.-H., P.Maas and Carroll, R. J. (2016). Constrained maximum 
#' likelihood estimation for model calibration using summary-level information from 
#' external big data sources. Journal of the American Statistical Association 111, 107-117.
#' 
#' Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction 
#' framework to integrate multiple regression models into a current study. 
#' Manuscript in preparation.
#'
#' @examples 
#' # Full model: Y|X1, X2, B
#' # Reduced model 1: Y|X1 of sample size m1
#' # Reduced model 2: Y|X2 of sample size m2
#' # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
#' set.seed(2333)
#' n = 1000
#' data.n = data.frame(matrix(ncol = 4, nrow = n))
#' colnames(data.n) = c('Y', 'X1', 'X2', 'B')
#' data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
#' data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = m1 = m2 = 30000
#' data.m = data.frame(matrix(ncol = 4, nrow = m))
#' names(data.m) = c('Y', 'X1', 'X2', 'B')
#' data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
#' data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
#' fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
#' beta.E1 = coef(fit.E1)
#' beta.E2 = coef(fit.E2)
#' names(beta.E1) = c('int', 'X1')
#' names(beta.E2) = c('int', 'X2')
#' V.E1 = vcov(fit.E1)
#' V.E2 = vcov(fit.E2)
#' 
#' #Save all the external model information into lists for later use
#' betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
#' CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
#' rho = list(Ext1 = n/m1, Ext2 = n/m2)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
#' gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
#' gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
#' #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
#' gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
#' 
#' #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
#' asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
#' asyV.I = asy.CML[['asyV.I']]                     
#' 
#' #Get the empirical Bayes (EB) estimates
#' gamma.EB1 = get_gamma_EB(gamma_I=gamma.I, gamma_CML=gamma.CML1, asyV.I=asyV.I)[['gamma.EB']]
#' gamma.EB2 = get_gamma_EB(gamma_I=gamma.I, gamma_CML=gamma.CML2, asyV.I=asyV.I)[['gamma.EB']]
#'
#'
#' @export 
#' 
get_gamma_EB <- function(gamma_I, gamma_CML, asyV.I){
  q = length(gamma_I)
  gammaDiff = gamma_I - gamma_CML
  A = gammaDiff %*% t(gammaDiff)
  Var_E = round(A, 5)
  Var_I = round(asyV.I, 5)
  
  #construct weights
  W = Var_E %*% solve(Var_E+Var_I,tol = 1e-1000)
  I_W = Var_I %*% solve(Var_E+Var_I ,tol = 1e-1000)
  
  gamma.EB = W %*% matrix(gamma_I, q, 1) + I_W %*% matrix(gamma_CML, q, 1)
  return(list('gamma.EB' = gamma.EB, 'weight' = W))
}

# ############### Example ####################################################################
# # Full model: Y|X1, X2, B
# # Reduced model 1: Y|X1 of sample size m1
# # Reduced model 2: Y|X2 of sample size m2
# # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
# set.seed(2333)
# n = 1000
# data.n = data.frame(matrix(ncol = 4, nrow = n))
# colnames(data.n) = c('Y', 'X1', 'X2', 'B')
# data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
# data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = m1 = m2 = 30000
# data.m = data.frame(matrix(ncol = 4, nrow = m))
# names(data.m) = c('Y', 'X1', 'X2', 'B')
# data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
# data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
# fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
# beta.E1 = coef(fit.E1)
# beta.E2 = coef(fit.E2)
# names(beta.E1) = c('int', 'X1')
# names(beta.E2) = c('int', 'X2')
# V.E1 = vcov(fit.E1)
# V.E2 = vcov(fit.E2)
# 
# #Save all the external model information into lists for later use
# betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
# CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
# rho = list(Ext1 = n/m1, Ext2 = n/m2)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
# gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
# gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
# #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
# gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
# 
# #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
# asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
# asyV.I = asy.CML[['asyV.I']]                     
# 
# #Get the empirical Bayes (EB) estimates
# gamma.EB1 = get_gamma_EB(gamma_I=gamma.I, gamma_CML=gamma.CML1, asyV.I=asyV.I)[['gamma.EB']]
# gamma.EB2 = get_gamma_EB(gamma_I=gamma.I, gamma_CML=gamma.CML2, asyV.I=asyV.I)[['gamma.EB']]
# ############### End of Example ##############################################################














######################################################################################################################################
###### Function 7: Using simulation to obtain the asymptotic variance-covariance matrix of gamma_EB, package corpcor and MASS are required
######################################################################################################################################
# Reference: Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction framework to integrate multiple regression 
# models into a current study. Manuscript in preparation.
######################################################################################################################################
# k: number of external models
# q: total number of covariates (X,B) including the intercept (i.e. q=ncol(X)+ncol(B)+1)
# gamma.CML: stack all k CML estimates in order, i.e. c(gamma.CML1,...,gamma.CMLk)
# asy.CML: a list of the estimated asymtotic variance-covariance matrix of c(gamma_CML, gamma_I) from the output of function asympVar_LinReg() or asympVar_LogReg()
# seed: specify seed for simulation
# nsim: number of simulation, default nsim=2,000

#' Using simulation to obtain the asymptotic variance-covariance matrix of gamma_EB, package corpcor and MASS are required
#'
#' @references 
#' Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction framework to integrate multiple regression 
#' models into a current study. Manuscript in preparation.
#'
#' @param k number of external models
#' @param q total number of covariates (X,B) including the intercept (i.e. q=ncol(X)+ncol(B)+1)
#' @param gamma.CML stack all k CML estimates in order, i.e. c(gamma.CML1,...,gamma.CMLk)
#' @param asy.CML a list of the estimated asymtotic variance-covariance matrix of 
#' c(gamma_CML, gamma_I) from the output of function asympVar_LinReg() or asympVar_LogReg()
#' @param seed specify seed for simulation
#' @param nsim number of simulation, default nsim=2,000
#' 
#' @return a list with: Var(gamma_EB), Cov(gamma_EB, gamma_I) and Cov(gamma_EBi, gamma_EBj)
#'
#' @examples 
#' # Full model: Y|X1, X2, B
#' # Reduced model 1: Y|X1 of sample size m1
#' # Reduced model 2: Y|X2 of sample size m2
#' # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
#' set.seed(2333)
#' n = 1000
#' data.n = data.frame(matrix(ncol = 4, nrow = n))
#' colnames(data.n) = c('Y', 'X1', 'X2', 'B')
#' data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
#' data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = m1 = m2 = 30000
#' data.m = data.frame(matrix(ncol = 4, nrow = m))
#' names(data.m) = c('Y', 'X1', 'X2', 'B')
#' data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
#' data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
#' fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
#' beta.E1 = coef(fit.E1)
#' beta.E2 = coef(fit.E2)
#' names(beta.E1) = c('int', 'X1')
#' names(beta.E2) = c('int', 'X2')
#' V.E1 = vcov(fit.E1)
#' V.E2 = vcov(fit.E2)
#' 
#' #Save all the external model information into lists for later use
#' betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
#' CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
#' rho = list(Ext1 = n/m1, Ext2 = n/m2)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
#' gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
#' gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
#' #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
#' gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
#' 
#' #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
#' asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
#' 
#' #Get the empirical Bayes (EB) estimates
#' gamma.EB1 = get_gamma_EB(gamma.I, gamma.CML1, asy.CML[['asyV.I']])[['gamma.EB']]
#' gamma.EB2 = get_gamma_EB(gamma.I, gamma.CML2, asy.CML[['asyV.I']])[['gamma.EB']]
#' 
#' #Get the asymptotic variance of the EB estimates
#' V.EB = get_var_EB(k=2, 
#'                   q=4, 
#'                   gamma.CML=c(gamma.CML1, gamma.CML2), 
#'                   asy.CML=asy.CML, 
#'                   seed=2333, 
#'                   nsim=2000)
#' asyV.EB1 = V.EB[['asyV.EB']][[1]]             #variance of gamma.EB1
#' asyV.EB2 = V.EB[['asyV.EB']][[2]]             #variance of gamma.EB2
#' asyCov.EB1.I = V.EB[['asyCov.EB.I']][[1]]     #covariance of gamma.EB1 and gamma.I
#' asyCov.EB2.I = V.EB[['asyCov.EB.I']][[2]]     #covariance of gamma.EB2 and gamma.I
#' asyCov.EB12 = V.EB[['asyCov.EB']][['12']]     #covariance of gamma.EB1 and gamma.EB2
#'
#' @export
#' 
get_var_EB <- function(k, q, gamma.CML, asy.CML, seed=2333, nsim=2000){
  library(corpcor)
  library(MASS)
  
  ######Define new variables from inputed list asy.CML
  asyV.I = asy.CML[['asyV.I']]              #variance of gamma.I
  asyV.CML = asy.CML[['asyV.CML']]          #variance of gamma.CML
  asyCov.CML.I = asy.CML[['asyCov.CML.I']]  #covariance of gamma.CML and gamma.I
  asyCov.CML = asy.CML[['asyCov.CML']]      #covariance of gamma.CMLi and gamma.CMLj
  ExtraTerm = asy.CML[['ExtraTerm']]        #the extra variance when considering the external model estimates uncertainty
  
  ######Calculate Variance-Covariance matrix of Z = gamma_I-gamma_CML
  Vz = list()
  for(a in 1:k){
    name = paste(a,a,sep='')
    Vz[[name]] = asyV.I + asyV.CML[[a]] - 2*asyCov.CML.I[[a]] #Var(Z)
    if(corpcor::is.positive.definite(Vz[[name]])==FALSE){Vz[[name]] = corpcor::make.positive.definite(Vz[[name]])}
    for(b in 1:k){
      if(a < b){
        name_ab = paste(a,b,sep=''); 
        Vz[[name_ab]] = asyV.I-asyCov.CML.I[[a]]-asyCov.CML.I[[b]]+asyCov.CML[[name_ab]]} #Cov(Zi,Zj)
    }
  }
  
  ########Calculate Cov(Z, gamma_CML)
  Cov.z.CML = matrix(0,k*q,k*q)
  for(a in 1:k){
    Cov.z.CML[(q*(a-1)+1):(q*a),(q*(a-1)+1):(q*a)] = -ExtraTerm[[a]] #Cov(Zi,CMLi)
    for(b in 1:k){
      if(a < b){
        name_ab = paste(a,b,sep=''); 
        Cov.z.CML[(q*(a-1)+1):(q*a),(q*(b-1)+1):(q*b)] = asyCov.CML.I[[b]] - asyCov.CML[[name_ab]]} #Cov(Zi,CMLj)
      if(a > b){
        name_ba = paste(b,a,sep=''); 
        Cov.z.CML[(q*(a-1)+1):(q*a),(q*(b-1)+1):(q*b)] = asyCov.CML.I[[b]] - asyCov.CML[[name_ba]]} #t(Cov(Zi,CMLj))
    }
  }
  
  ##################Start constructing Var-Cov matrix of c(Z, gamma_CML, gamma_I) ##################
  V_sim = matrix(0,(2*k+1)*q,(2*k+1)*q)
  ######Top-left kq*kq matrix
  for(a in 1:k){
    name = paste(a,a,sep='')
    V_sim[(q*(a-1)+1):(q*a),(q*(a-1)+1):(q*a)] = Vz[[name]] #diagnal terms Var(Z)
    V_sim[(q*(a-1)+1):(q*a),(2*k*q+1):(2*k*q+q)] = asyV.I - asyCov.CML.I[[a]] #Cov(Z,I)=Var(I)-Cov(CML,I)=Var(Z)+ExtraTerm
    for(b in 1:k){
      if(a < b){
        name_ab = paste(a,b,sep=''); 
        V_sim[(q*(a-1)+1):(q*(a-1)+q),(q*(b-1)+1):(q*b)] = Vz[[name_ab]]}
      if(a > b){
        name_ba = paste(b,a,sep=''); 
        V_sim[(q*(a-1)+1):(q*(a-1)+q),(q*(b-1)+1):(q*b)] = t(Vz[[name_ba]])}
    }
  }
  #######Top-right kq*kq matrix
  for(a in (k+1):(2*k)){
    #######diagnal terms
    V_sim[(q*(a-1)+1):(q*(a-1)+q),(q*(a-1)+1):(q*(a-1)+q)] = asyV.CML[[a-k]] #diagnal terms Var(CML)
    V_sim[(q*(a-1)+1):(q*(a-1)+q),(2*k*q+1):(2*k*q+q)] = asyCov.CML.I[[a-k]] #Cov(CML,I)
    for(b in (k+1):(2*k)){
      if(a < b){
        name_ab = paste(a-k,b-k,sep=''); 
        V_sim[(q*(a-1)+1):(q*(a-1)+q),(q*(b-1)+1):(q*b)] = asyCov.CML[[name_ab]]} #Cov(CMLi,CMLj)
      if(a > b){
        name_ba = paste(b-k,a-k,sep=''); 
        V_sim[(q*(a-1)+1):(q*(a-1)+q),(q*(b-1)+1):(q*b)] = t(as.matrix(asyCov.CML[[name_ab]],q,q))} #t(Cov(CMLi,CMLj))
    }
  }
  V_sim[1:(k*q),(k*q+1):(2*k*q)] = Cov.z.CML
  V_sim[(k*q+1):(2*k*q),1:(k*q)] = t(Cov.z.CML)
  V_sim[(2*k*q+1):(2*k*q+q),(2*k*q+1):(2*k*q+q)] = asyV.I
  
  ############ using derived the relationship to generate simulated gamma_EB of size nsim to obtain Var(gamma_EB)
  set.seed(seed)
  value_sim = MASS::mvrnorm(nsim, c(rep(0,k*q), gamma.CML, gamma.I), corpcor::make.positive.definite(V_sim))
  Z_sim = CML_sim = list()
  for(i in 1:k){
    Z_sim[[i]] = value_sim[,(1+(i-1)*q):(i*q)]
    CML_sim[[i]] = value_sim[,(1+(i+k-1)*q):((i+k)*q)]
  }
  I_sim = value_sim[,(2*k*q+1):(2*k*q+q)]
  
  asyV.EB = list()
  asyCov.EB.I = list()
  asyCov.EB = list()
  for(a in 1:k){
    A_Z_a = apply(Z_sim[[a]], 1, function(z){t(as.vector(z))%*%solve(asyV.I)%*%as.vector(z)/(1+t(as.vector(z))%*%solve(asyV.I)%*%as.vector(z))})
    f_Z_a = Z_sim[[a]] * matrix(rep(A_Z_a,q), nsim, q)
    f_Z_a_2 = Z_sim[[a]] * matrix(rep(1-A_Z_a,q), nsim, q)
    asyV.EB[[a]] = asyV.CML[[a]] + round(stats::var(f_Z_a), 5)
    asyCov.EB.I[[a]] = round(asyV.I - stats::cov(f_Z_a_2, I_sim), 5)
    for(b in 1:k){
      if(a<b){
        A_Z_b = apply(Z_sim[[b]], 1, function(z){t(as.vector(z))%*%solve(asyV.I)%*%as.vector(z)/(1+t(as.vector(z))%*%solve(asyV.I)%*%as.vector(z))})
        f_Z_b = Z_sim[[b]] * matrix(rep(A_Z_b,q), nsim, q)
        name = paste0(a,b)
        asyCov.EB[[name]] = asyCov.CML[[name]] + round(stats::cov(f_Z_a, CML_sim[[b]])+ stats::cov(f_Z_b, CML_sim[[a]]) + stats::cov(f_Z_a, f_Z_b), 5)
      }
    }
  }
  
  ############ return Var(gamma_EB), Cov(gamma_EB, gamma_I) and Cov(gamma_EBi, gamma_EBj)
  return(list(asyV.EB=asyV.EB, asyCov.EB.I=asyCov.EB.I, asyCov.EB=asyCov.EB))
}



# ############### Example ############################################
# # Full model: Y|X1, X2, B
# # Reduced model 1: Y|X1 of sample size m1
# # Reduced model 2: Y|X2 of sample size m2
# # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
# set.seed(2333)
# n = 1000
# data.n = data.frame(matrix(ncol = 4, nrow = n))
# colnames(data.n) = c('Y', 'X1', 'X2', 'B')
# data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
# data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = m1 = m2 = 30000
# data.m = data.frame(matrix(ncol = 4, nrow = m))
# names(data.m) = c('Y', 'X1', 'X2', 'B')
# data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
# data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
# fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
# beta.E1 = coef(fit.E1)
# beta.E2 = coef(fit.E2)
# names(beta.E1) = c('int', 'X1')
# names(beta.E2) = c('int', 'X2')
# V.E1 = vcov(fit.E1)
# V.E2 = vcov(fit.E2)
# 
# #Save all the external model information into lists for later use
# betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
# CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
# rho = list(Ext1 = n/m1, Ext2 = n/m2)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
# gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
# gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
# #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
# gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
# 
# #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
# asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
# 
# #Get the empirical Bayes (EB) estimates
# gamma.EB1 = get_gamma_EB(gamma.I, gamma.CML1, asy.CML[['asyV.I']])[['gamma.EB']]
# gamma.EB2 = get_gamma_EB(gamma.I, gamma.CML2, asy.CML[['asyV.I']])[['gamma.EB']]
# 
# #Get the asymptotic variance of the EB estimates
# V.EB = get_var_EB(k=2, 
#                   q=4, 
#                   gamma.CML=c(gamma.CML1, gamma.CML2), 
#                   asy.CML=asy.CML, 
#                   seed=2333, 
#                   nsim=2000)
# asyV.EB1 = V.EB[['asyV.EB']][[1]]             #variance of gamma.EB1
# asyV.EB2 = V.EB[['asyV.EB']][[2]]             #variance of gamma.EB2
# asyCov.EB1.I = V.EB[['asyCov.EB.I']][[1]]     #covariance of gamma.EB1 and gamma.I
# asyCov.EB2.I = V.EB[['asyCov.EB.I']][[2]]     #covariance of gamma.EB2 and gamma.I
# asyCov.EB12 = V.EB[['asyCov.EB']][['12']]     #covariance of gamma.EB1 and gamma.EB2
# ############### End of Example #####################################








######################################################################################################################################
###### Function 8: Obtain the proposed Optimal Covariate-Weighted (OCW) estimates, package Rsolnp is required
######################################################################################################################################
# Reference: Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction framework to integrate multiple regression 
# models into a current study. Manuscript in preparation.
######################################################################################################################################
# k: number of external models
# q: total number of covariates (X,B) including the intercept (i.e. q=ncol(X)+ncol(B)+1)
# data.XB: internal data (X,B)
# gamma.EB: stack all k EB estimates in order, i.e. c(gamma.EB1,...,gamma.EBk)
# V.EB: variance-covariance matrix obtained from function get_var_EB()

#' Obtain the proposed Optimal Covariate-Weighted (OCW) estimates
#'
#' @param k number of external models
#' @param q total number of covariates (X,B) including the intercept (i.e. q=ncol(X)+ncol(B)+1)
#' @param data.XB internal data (X,B)
#' @param gamma.EB stack all k EB estimates in order, i.e. c(gamma.EB1,...,gamma.EBk)
#' @param V.EB variance-covariance matrix obtained from function get_var_EB()
#' 
#' @return return weights of gamma.EB's, final estimates of OCW estimates and the 
#' corresponding variance-covariance matrix
#'
#' @references 
#' Reference: Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction framework to integrate multiple regression 
#' models into a current study. Manuscript in preparation.
#' 
#' @examples 
#' 
#' # Full model: Y|X1, X2, B
#' # Reduced model 1: Y|X1 of sample size m1
#' # Reduced model 2: Y|X2 of sample size m2
#' # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
#' set.seed(2333)
#' n = 1000
#' data.n = data.frame(matrix(ncol = 4, nrow = n))
#' colnames(data.n) = c('Y', 'X1', 'X2', 'B')
#' data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
#' data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = m1 = m2 = 30000
#' data.m = data.frame(matrix(ncol = 4, nrow = m))
#' names(data.m) = c('Y', 'X1', 'X2', 'B')
#' data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
#' data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
#' fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
#' beta.E1 = coef(fit.E1)
#' beta.E2 = coef(fit.E2)
#' names(beta.E1) = c('int', 'X1')
#' names(beta.E2) = c('int', 'X2')
#' V.E1 = vcov(fit.E1)
#' V.E2 = vcov(fit.E2)
#' 
#' #Save all the external model information into lists for later use
#' betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
#' CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
#' rho = list(Ext1 = n/m1, Ext2 = n/m2)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
#' gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
#' gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
#' #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
#' gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
#' 
#' #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
#' asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
#' 
#' #Get the empirical Bayes (EB) estimates
#' gamma.EB1 = get_gamma_EB(gamma.I, gamma.CML1, asy.CML[['asyV.I']])[['gamma.EB']]
#' gamma.EB2 = get_gamma_EB(gamma.I, gamma.CML2, asy.CML[['asyV.I']])[['gamma.EB']]
#' 
#' #Get the asymptotic variance of the EB estimates
#' V.EB = get_var_EB(k=2, q=4, gamma.CML=c(gamma.CML1, gamma.CML2), asy.CML=asy.CML, seed=2333, nsim=2000)
#' 
#' #Get the OCW estimates,  the corresponding variance-covariance matrix of the estimates and the weights of gamma.EB's
#' get_OCW(k=2, 
#'         q=4, 
#'         data.XB=data.n[,c('X1','X2','B')], 
#'         gamma.EB=c(gamma.EB1, gamma.EB2), 
#'         V.EB=V.EB)
#'
#' @export
#' 
get_OCW = function(k, q, data.XB, gamma.EB, V.EB){
  XBmatrix = data.matrix(cbind(1, data.XB))
  asyV.EB = V.EB[['asyV.EB']]
  asyCov.EB = V.EB[['asyCov.EB']]
  
  ###### create the constant coefficient matrix to multiply with each variance covariance matrix
  constantMatrix = matrix(0,q,q)
  for(a in 1:q){
    for(b in 1:q){
      if(a <= b){
        constantMatrix[a,b] = sum(XBmatrix[,a] * XBmatrix[,b]) #to get sum(Xi*Xj)
      }
    }
  }
  constantMatrix[lower.tri(constantMatrix)] = t(constantMatrix)[lower.tri(constantMatrix)]
  
  ###### construct Amatrix: e.g. when k=2, Amatrix = rbind(cbind(sum(constantMatrix*asyV.EB1), sum(constantMatrix*asyCov.EB12)),cbind(sum(constantMatrix*asyCov.EB12), sum(constantMatrix*asyV.EB2)))
  Amatrix = matrix(NA, k, k)
  diag(Amatrix) = as.numeric(lapply(asyV.EB, function(l) sum(constantMatrix*l)))
  for(a in 1:k){
    for(b in 1:k){
      if(a < b){
        name = paste0(a,b)
        Amatrix[a,b] = sum(constantMatrix*asyCov.EB[[name]])
      }
    }
  }
  Amatrix[lower.tri(Amatrix)] = t(Amatrix)[lower.tri(Amatrix)]
  
  ###### obtain weights of gamma.EB's for the OCW estimates 
  weights = Rsolnp::solnp(par = rep(1/k, k),
                          fun = function(w) matrix(w, 1, k) %*% Amatrix %*% matrix(w, k, 1),
                          eqfun = function(w){return(sum(w))},
                          eqB = 1,
                          LB = rep(0, k),
                          UB = rep(1, k),
                          control = list(trace = 0))
  weights.OCW = as.numeric(weights$pars)
  
  ###### obtain OCW estimates 
  gamma.OCW = matrix(gamma.EB, q, k) %*% matrix(weights.OCW, k, 1)
  
  ###### obtain the variance-covariance matrix of the OCW estimates 
  var.OCW = 0
  for(a in 1:k){
    var.OCW = var.OCW + weights.OCW[a]^2*asyV.EB[[a]]
    for(b in 1:k){
      if(a < b){
        ab = paste0(a, b)
        var.OCW = var.OCW + weights.OCW[a]*weights.OCW[b]*asyCov.EB[[ab]] + weights.OCW[a]*weights.OCW[b]*t(asyCov.EB[[ab]])
      }
    }
  }

  ##### return weights of gamma.EB's, final estimates of OCW estimates and teh corresponding variance-covariance matrix
  return(list('Weights' = weights.OCW, 'gamma.OCW'=gamma.OCW, 'var.OCW'=var.OCW))
}


# ################## Example ###################################################################
# # Full model: Y|X1, X2, B
# # Reduced model 1: Y|X1 of sample size m1
# # Reduced model 2: Y|X2 of sample size m2
# # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
# set.seed(2333)
# n = 1000
# data.n = data.frame(matrix(ncol = 4, nrow = n))
# colnames(data.n) = c('Y', 'X1', 'X2', 'B')
# data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
# data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = m1 = m2 = 30000
# data.m = data.frame(matrix(ncol = 4, nrow = m))
# names(data.m) = c('Y', 'X1', 'X2', 'B')
# data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
# data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
# fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
# beta.E1 = coef(fit.E1)
# beta.E2 = coef(fit.E2)
# names(beta.E1) = c('int', 'X1')
# names(beta.E2) = c('int', 'X2')
# V.E1 = vcov(fit.E1)
# V.E2 = vcov(fit.E2)
# 
# #Save all the external model information into lists for later use
# betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
# CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
# rho = list(Ext1 = n/m1, Ext2 = n/m2)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
# gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
# gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
# #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
# gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
# 
# #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
# asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
# 
# #Get the empirical Bayes (EB) estimates
# gamma.EB1 = get_gamma_EB(gamma.I, gamma.CML1, asy.CML[['asyV.I']])[['gamma.EB']]
# gamma.EB2 = get_gamma_EB(gamma.I, gamma.CML2, asy.CML[['asyV.I']])[['gamma.EB']]
# 
# #Get the asymptotic variance of the EB estimates
# V.EB = get_var_EB(k=2, q=4, gamma.CML=c(gamma.CML1, gamma.CML2), asy.CML=asy.CML, seed=2333, nsim=2000)
# 
# #Get the OCW estimates,  the corresponding variance-covariance matrix of the estimates and the weights of gamma.EB's
# get_OCW(k=2, 
#         q=4, 
#         data.XB=data.n[,c('X1','X2','B')], 
#         gamma.EB=c(gamma.EB1, gamma.EB2), 
#         V.EB=V.EB)
# ################## End of Example ##############################################################






######################################################################################################################################
###### Function 9: Obtain the proposed Selective Coefficient-Learner (SC-Learner) estimates
######################################################################################################################################
# Reference: Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble meta-prediction framework to integrate multiple regression 
# models into a current study. Manuscript in preparation.
######################################################################################################################################
# k: number of external models
# q: total number of covariates (X,B) including the intercept (i.e. q=ncol(X)+ncol(B)+1)
# predictor.matrix: a predictor matrix (q rows by k columns) that specifies the full model variables in the rows, and the external models on the columns.
#                   An entry of 0 means that the row variable is NOT used in the column external model; 1 represents that it is used.
# gamma.EB: bind all k EB estimates in order (q rows by k columns), i.e. cbind(gamma.EB1,...,gamma.EBk)
# V.EB: variance-covariance matrix obtained from function get_var_EB()

#' Obtain the proposed Selective Coefficient-Learner (SC-Learner) estimates
#' 
#' @param k number of external models
#' @param q total number of covariates (X,B) including the intercept (i.e. q=ncol(X)+ncol(B)+1)
#' @param predictor.matrix a predictor matrix (q rows by k columns) that specifies 
#' the full model variables in the rows, and the external models on the columns. 
#' An entry of 0 means that the row variable is NOT used in the column external model; 1 represents that it is used.
#' @param gamma.EB bind all k EB estimates in order (q rows by k columns), i.e. cbind(gamma.EB1,...,gamma.EBk)
#' @param V.EB variance-covariance matrix obtained from function get_var_EB()
#' 
#' @examples 
#' # Full model: Y|X1, X2, B
#' # Reduced model 1: Y|X1 of sample size m1
#' # Reduced model 2: Y|X2 of sample size m2
#' # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
#' # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
#' set.seed(2333)
#' n = 1000
#' data.n = data.frame(matrix(ncol = 4, nrow = n))
#' colnames(data.n) = c('Y', 'X1', 'X2', 'B')
#' data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
#' data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
#' 
#' # Generate the beta estimates from the external reduced model:
#' # generate a data of size m from the full model first, then fit the reduced regression 
#' # to obtain the beta estiamtes and the corresponsing estimated variance 
#' m = m1 = m2 = 30000
#' data.m = data.frame(matrix(ncol = 4, nrow = m))
#' names(data.m) = c('Y', 'X1', 'X2', 'B')
#' data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
#' data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
#' 
#' #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
#' fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
#' fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
#' beta.E1 = coef(fit.E1)
#' beta.E2 = coef(fit.E2)
#' names(beta.E1) = c('int', 'X1')
#' names(beta.E2) = c('int', 'X2')
#' V.E1 = vcov(fit.E1)
#' V.E2 = vcov(fit.E2)
#' 
#' #Save all the external model information into lists for later use
#' betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
#' CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
#' rho = list(Ext1 = n/m1, Ext2 = n/m2)
#' 
#' #get full model estimate from direct regression using the internal data only
#' fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
#' gamma.I = coef(fit.gamma.I)
#' 
#' #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
#' gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
#' gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
#' #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
#' gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
#' 
#' #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
#' asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
#' 
#' #Get the empirical Bayes (EB) estimates
#' gamma.EB1 = get_gamma_EB(gamma.I, gamma.CML1, asy.CML[['asyV.I']])[['gamma.EB']]
#' gamma.EB2 = get_gamma_EB(gamma.I, gamma.CML2, asy.CML[['asyV.I']])[['gamma.EB']]
#' 
#' #Get the asymptotic variance of the EB estimates
#' V.EB = get_var_EB(k=2, q=4, gamma.CML=c(gamma.CML1, gamma.CML2), asy.CML=asy.CML, seed=2333, nsim=2000)
#' 
#' #Get the SC-Learner estimates and the corresponding variance-covariance matrix
#' pred.matrix = matrix(c(1,1,1,0,
#'                        1,1,0,0), 4, 2)
#' rownames(pred.matrix) = c('int','X1','X2','B')
#' colnames(pred.matrix) = c('E1','E2')
#' 
#' get_SCLearner(k=2,
#'               q=4,
#'               pred.matrix=pred.matrix,
#'               gamma.EB=cbind(gamma.EB1, gamma.EB2),
#'               V.EB)
#' 
#' 
#' @return a list with gamma.SCLearner and var.SCLearner
#'
#' @export
#' 
get_SCLearner <- function(k, q, pred.matrix, gamma.EB, V.EB){
  asyV.EB = V.EB[['asyV.EB']]
  asyCov.EB = V.EB[['asyCov.EB']]
  var.EB.list = lapply(asyV.EB, diag)
  var.EB = var.EB.list[[1]]
  if(k>1){
    for(i in 2:k){
      var.EB = cbind(var.EB, var.EB.list[[i]])
    }
  }
  var.EB.to.use = var.EB*pred.matrix
  
  inv.var.EB = 1/var.EB.to.use
  inv.var.EB[is.infinite(inv.var.EB)] = 0
  weights = inv.var.EB / base::apply(inv.var.EB, 1, sum)
  weights[is.na(weights)] = 1/k

  #SC-Learner estimates
  gamma.SCLearner = rowSums(weights * gamma.EB)
  
  #variance-covariance matrix of the SC-Learner estimates
  var.SCLearner = 0
  for(a in 1:k){
    var.SCLearner = var.SCLearner + diag(weights[,a])%*%asyV.EB[[a]]%*%diag(weights[,a])
    for(b in 1:k){
      if(a < b){
        ab = paste0(a,b)
        var.SCLearner = var.SCLearner + diag(weights[,a])%*%asyCov.EB[[ab]]%*%diag(weights[,b])+t(diag(weights[,a])%*%asyCov.EB[[ab]]%*%diag(weights[,b]))
      }
    }
  }
  
  return(list('gamma.SCLearner'=gamma.SCLearner, 'var.SCLearner'=var.SCLearner))
}


# ################## Example ###################################################################
# # Full model: Y|X1, X2, B
# # Reduced model 1: Y|X1 of sample size m1
# # Reduced model 2: Y|X2 of sample size m2
# # (X1, X2, B) follows normal distribution with mean zero, variance one and correlation 0.3
# # Y|X1, X2, B follows Bernoulli[expit(-1-0.5*X1-0.5*X2+0.5*B)], where expit(x)=exp(x)/[1+exp(x)]
# set.seed(2333)
# n = 1000
# data.n = data.frame(matrix(ncol = 4, nrow = n))
# colnames(data.n) = c('Y', 'X1', 'X2', 'B')
# data.n[,c('X1', 'X2', 'B')] = MASS::mvrnorm(n, rep(0,3), diag(0.7,3)+0.3)
# data.n$Y = rbinom(n, 1, expit(-1 - 0.5*data.n$X1 - 0.5*data.n$X2 + 0.5*data.n$B))
# 
# # Generate the beta estimates from the external reduced model:
# # generate a data of size m from the full model first, then fit the reduced regression 
# # to obtain the beta estiamtes and the corresponsing estimated variance 
# m = m1 = m2 = 30000
# data.m = data.frame(matrix(ncol = 4, nrow = m))
# names(data.m) = c('Y', 'X1', 'X2', 'B')
# data.m[,c('X1', 'X2', 'B')] = MASS::mvrnorm(m, rep(0,3), diag(0.7,3)+0.3)
# data.m$Y = rbinom(m, 1, expit(-1 - 0.5*data.m$X1 - 0.5*data.m$X2 + 0.5*data.m$B))
# 
# #fit Y|X to obtain the external beta estimates, save the beta estiamtes and the corresponsing estimated variance 
# fit.E1 = glm(Y ~ X1, data = data.m, family = binomial(link='logit'))
# fit.E2 = glm(Y ~ X2, data = data.m, family = binomial(link='logit'))
# beta.E1 = coef(fit.E1)
# beta.E2 = coef(fit.E2)
# names(beta.E1) = c('int', 'X1')
# names(beta.E2) = c('int', 'X2')
# V.E1 = vcov(fit.E1)
# V.E2 = vcov(fit.E2)
# 
# #Save all the external model information into lists for later use
# betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2)
# CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2)
# rho = list(Ext1 = n/m1, Ext2 = n/m2)
# 
# #get full model estimate from direct regression using the internal data only
# fit.gamma.I = glm(Y ~ X1 + X2 + B, data = data.n, family = binomial(link='logit'))
# gamma.I = coef(fit.gamma.I)
# 
# #Get CML estimates using internal data and the beta estimates from the external model 1 and 2, respectively
# gamma.CML1 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X1, BInt=cbind(data.n$X2, data.n$B), betaHatExt=beta.E1, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400,factor=1)[["gammaHat"]]
# gamma.CML2 = fxnCC_LogReg(p=2, q=4, YInt=data.n$Y, XInt=data.n$X2, BInt=cbind(data.n$X1, data.n$B), betaHatExt=beta.E2, gammaHatInt=gamma.I, n=nrow(data.n), tol=1e-8, maxIter=400, factor=1)[["gammaHat"]]
# #It's important to reorder gamma.CML2 so that it follows the order (X1, X2, X3, B) as gamma.I and gamma.CML1
# gamma.CML2 = c(gamma.CML2[1], gamma.CML2[3], gamma.CML2[2], gamma.CML2[4])
# 
# #Get Variance-covariance matricx of c(gamma.I, gamma.CML1, gamma.CML2)
# asy.CML = asympVar_LogReg(k=2, p=2,q=4, YInt=data.n$Y, XInt=data.n[,c('X1','X2')], BInt=data.n$B,  gammaHatInt=gamma.I, betaHatExt_list=betaHatExt_list, CovExt_list=CovExt_list, rho=rho, ExUncertainty=TRUE)
# 
# #Get the empirical Bayes (EB) estimates
# gamma.EB1 = get_gamma_EB(gamma.I, gamma.CML1, asy.CML[['asyV.I']])[['gamma.EB']]
# gamma.EB2 = get_gamma_EB(gamma.I, gamma.CML2, asy.CML[['asyV.I']])[['gamma.EB']]
# 
# #Get the asymptotic variance of the EB estimates
# V.EB = get_var_EB(k=2, q=4, gamma.CML=c(gamma.CML1, gamma.CML2), asy.CML=asy.CML, seed=2333, nsim=2000)
# 
# #Get the SC-Learner estimates and the corresponding variance-covariance matrix
# pred.matrix = matrix(c(1,1,1,0,
#                        1,1,0,0), 4, 2)
# rownames(pred.matrix) = c('int','X1','X2','B')
# colnames(pred.matrix) = c('E1','E2')
# 
# get_SCLearner(k=2,
#               q=4,
#               pred.matrix=pred.matrix,
#               gamma.EB=cbind(gamma.EB1, gamma.EB2),
#               V.EB)
# ################## End of Example ##############################################################





