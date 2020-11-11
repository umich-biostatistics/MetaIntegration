
<!-- README.md is generated from README.Rmd. Please edit that file -->

# R package `MetaIntegration`

# Ensemble Meta-Prediction Framework to Integrate Multiple Regression Models

[![](https://img.shields.io/badge/devel%20version-0.1.0.9000-blue.svg)](https://github.com/umich-biostatistics/MetaIntegration)
[![](https://img.shields.io/github/languages/code-size/umich-biostatistics/MetaIntegration.svg)](https://github.com/umich-biostatistics/MetaIntegration)

## Overview

This `R` package is an ensemble meta-prediction framework to integrate
multiple regression models into a current study.

## Installation

If the devtools package is not yet installed, install it first:

``` r
install.packages('devtools')
```

``` r
# install the package from Github:
devtools::install_github('umich-biostatistics/MetaIntegration') 
```

Once installed, load the package:

``` r
library(MetaIntegration)
```

## Example Usage

For this example, we simulate data and test each of the functions.

## Simulate data

``` r
############## Generating 1 internal dataset of size n ###########################################
# Full model: Y|X1, X2, X3, X4, B
# (X1, X2, X3, X4, B) follows normal distribution with mean zero, variance one and correlation 0.3
# Y|X1, X2, X3, X4, B follows Bernoulli[expit(-1-0.5X+0.5B)], where expit(x)=exp(x)/[1+exp(x)]
##################################################################################################
set.seed(2333)
n = 200
data.n = data.frame(matrix(ncol = 5, nrow = n))
colnames(data.n) = c('Y', 'X1', 'X2', 'X3', 'B')
data.n[,c('X1', 'X2', 'X3', 'B')] = MASS::mvrnorm(n, rep(0,4), diag(0.7,4)+0.3)
############# Function 1 #########################################
data.n$Y = rbinom(n, 1, expit(-1 - 0.5*(data.n$X1 + data.n$X2 + data.n$X3) + 0.5*data.n$B))

############# Generate k=3 external models #########################################
# Full model: Y|X1, X2, X3, B
# Reduced external model 1: Y|X1, X2 with sample size m1
# Reduced external model 2: Y|X1, X3 with sample size m2
# Reduced external model 3: Y|X1, X2, X3 with sample size m3
####################################################################################
# generate data from the full model first, then fit the reduced logistic regression 
# on the data to obtain the beta estiamtes and the corresponsing estimated variance 
####################################################################################
m1 = 500
m2 = 2000
m3 = 30000
data.m1 = data.frame(matrix(ncol = 5, nrow = m1))
data.m2 = data.frame(matrix(ncol = 5, nrow = m2))
data.m3 = data.frame(matrix(ncol = 5, nrow = m3))
names(data.m1) = names(data.m2) = names(data.m3) = c('Y', 'X1', 'X2', 'X3', 'B')

data.m1[,c('X1', 'X2', 'X3', 'B')] = MASS::mvrnorm(m1, rep(0,4), diag(0.7,4)+0.3)
data.m2[,c('X1', 'X2', 'X3', 'B')] = MASS::mvrnorm(m2, rep(0,4), diag(0.7,4)+0.3)
data.m3[,c('X1', 'X2', 'X3', 'B')] = MASS::mvrnorm(m3, rep(0,4), diag(0.7,4)+0.3)
data.m1$Y = rbinom(m1, 1, expit(-1 - 0.5*(data.m1$X1 + data.m1$X2 + data.m1$X3) + 0.5*data.m1$B))
data.m2$Y = rbinom(m2, 1, expit(-1 - 0.5*(data.m2$X1 + data.m2$X2 + data.m2$X3) + 0.5*data.m2$B))
data.m3$Y = rbinom(m3, 1, expit(-1 - 0.5*(data.m3$X1 + data.m3$X2 + data.m3$X3) + 0.5*data.m3$B))

#fit Y|X using logistic regression to obtain the external beta estimates
fit.E1 = glm(Y ~ X1 + X2,      data = data.m1, family = binomial(link='logit'))
fit.E2 = glm(Y ~ X1 + X3,      data = data.m2, family = binomial(link='logit'))
fit.E3 = glm(Y ~ X1 + X2 + X3, data = data.m3, family = binomial(link='logit'))

#Save the beta estiamtes and the corresponsing estimated variance of the reduced external model 1
beta.E1 = coef(fit.E1)
names(beta.E1) = c('int', 'X1', 'X2')
V.E1 = vcov(fit.E1)

#Save the beta estiamtes and the corresponsing estimated variance of the reduced external model 2
beta.E2 = coef(fit.E2)
names(beta.E2) = c('int','X1','X3')
V.E2 = vcov(fit.E2)

#Save the beta estiamtes and the corresponsing estimated variance of the reduced external model 3
beta.E3 = coef(fit.E3)
names(beta.E3) = c('int','X1','X2','X3')
V.E3 = vcov(fit.E3)

#Save all the external model information into lists for later use
betaHatExt_list = list(Ext1 = beta.E1, Ext2 = beta.E2, Ext3 = beta.E3)
CovExt_list = list(Ext1 = V.E1, Ext2 = V.E2, Ext3 = V.E3)
rho = list(Ext1 = n/m1, Ext2 = n/m2, Ext3 = n/m3)
```

## fxnCC\_LogReg()

Constraint maximum likelihood (CML) method for logistic regression
(binary outcome Y).

``` r
fit.gamma.I = glm(Y ~ X1 + X2 + X3 + B, data = data.n, family = binomial(link='logit'))
gamma.I = coef(fit.gamma.I)
gamma.CML1 = fxnCC_LogReg(p=3, 
                          q=5, 
                          YInt=data.n$Y, 
                          XInt=cbind(data.n$X1,data.n$X2), 
                          BInt=cbind(data.n$X3,data.n$B), 
                          betaHatExt=beta.E1, 
                          gammaHatInt=gamma.I, 
                          n=n, 
                          tol=1e-8, 
                          maxIter=400,
                          factor=1)[["gammaHat"]]
gamma.CML2 = fxnCC_LogReg(p=3, 
                          q=5, 
                          YInt=data.n$Y, 
                          XInt=cbind(data.n$X1,data.n$X3), 
                          BInt=cbind(data.n$X2, data.n$B), 
                          betaHatExt=beta.E2, 
                          gammaHatInt=c(gamma.I[1:2],gamma.I[4],gamma.I[3],gamma.I[5]), 
                          n=n, 
                          tol=1e-8, 
                          maxIter=400,
                          factor=1)[["gammaHat"]]
gamma.CML2 = c(gamma.CML2[1:2], gamma.CML2[4], gamma.CML2[3], gamma.CML2[5])
gamma.CML3 = fxnCC_LogReg(p=4, 
                          q=5, 
                          YInt=data.n$Y, 
                          XInt=cbind(data.n$X1,data.n$X2,data.n$X3), 
                          BInt=cbind(data.n$B), 
                          betaHatExt=beta.E3, 
                          gammaHatInt=gamma.I, 
                          n=n, 
                          tol=1e-8, 
                          maxIter=400,
                          factor=1)[["gammaHat"]]
```

## asympVar\_LogReg()

Asymptotic variance-covariance matrix for gamma\_Int and gamma\_CML for
logistic regression (binary outcome Y).

``` r
asy.CML = asympVar_LogReg(k=3, 
                          p=4,
                          q=5, 
                          YInt=data.n$Y, 
                          XInt=data.n[,c('X1','X2','X3')], #covariates that appeared in at least one external model
                          BInt=data.n$B,  #covariates that not used in any of the external models
                          gammaHatInt=gamma.I, 
                          betaHatExt_list=betaHatExt_list, 
                          CovExt_list=CovExt_list, 
                          rho=rho, 
                          ExUncertainty=TRUE)
asyV.I = asy.CML[['asyV.I']]                      #variance of gamma.I
asyV.CML1 = asy.CML[['asyV.CML']][[1]]            #variance of gamma.CML1
asyV.CML2 = asy.CML[['asyV.CML']][[2]]            #variance of gamma.CML2
asyCov.CML1.I = asy.CML[['asyCov.CML.I']][[1]]    #covariance of gamma.CML1 and gamma.I
asyCov.CML2.I = asy.CML[['asyCov.CML.I']][[2]]    #covariance of gamma.CML2 and gamma.I
asyCov.CML12 = asy.CML[['asyCov.CML']][['12']]    #covariance of gamma.CML1 and gamma.CML2
```

## get\_gamma\_EB()

Calculate the empirical Bayes (EB) estimates.

``` r
gamma.EB1 = get_gamma_EB(gamma_I=gamma.I, gamma_CML=gamma.CML1, asyV.I=asyV.I)[['gamma.EB']]
gamma.EB2 = get_gamma_EB(gamma_I=gamma.I, gamma_CML=gamma.CML2, asyV.I=asyV.I)[['gamma.EB']]
gamma.EB3 = get_gamma_EB(gamma_I=gamma.I, gamma_CML=gamma.CML3, asyV.I=asyV.I)[['gamma.EB']]
```

## get\_var\_EB()

Using simulation to obtain the asymptotic variance-covariance matrix of
gamma\_EB, package corpcor and MASS are required.

``` r
#Get the asymptotic variance of the EB estimates
V.EB = get_var_EB(k=3, 
                  q=5, 
                  gamma.CML=c(gamma.CML1, gamma.CML2, gamma.CML3), 
                  gamma.I = gamma.I,
                  asy.CML=asy.CML, 
                  seed=2333, 
                  nsim=2000)
asyV.EB1 = V.EB[['asyV.EB']][[1]]             #variance of gamma.EB1
asyV.EB2 = V.EB[['asyV.EB']][[2]]             #variance of gamma.EB2
asyV.EB3 = V.EB[['asyV.EB']][[3]]             #variance of gamma.EB3
asyCov.EB1.I = V.EB[['asyCov.EB.I']][[1]]     #covariance of gamma.EB1 and gamma.I
asyCov.EB2.I = V.EB[['asyCov.EB.I']][[2]]     #covariance of gamma.EB2 and gamma.I
asyCov.EB3.I = V.EB[['asyCov.EB.I']][[3]]     #covariance of gamma.EB2 and gamma.I
asyCov.EB12 = V.EB[['asyCov.EB']][['12']]     #covariance of gamma.EB1 and gamma.EB2
asyCov.EB13 = V.EB[['asyCov.EB']][['13']]     #covariance of gamma.EB1 and gamma.EB3
asyCov.EB23 = V.EB[['asyCov.EB']][['23']]     #covariance of gamma.EB2 and gamma.EB3
```

# get\_OCW()

Obtain the proposed Optimal covariate-Weighted (OCW) estimates, package
Rsolnp is required.

``` r
get_OCW(k=3, 
        q=5, 
        data.XB=data.n[,c('X1','X2','X3','B')], 
        gamma.EB=c(gamma.EB1, gamma.EB2, gamma.EB3), 
        V.EB=V.EB)
```

# get\_SCLearner()

Obtain the proposed Selective Coefficient-Learner (SC-Learner)
estimates.

``` r
pred.matrix = matrix(c(1,1,1,0,0,
                       1,1,0,1,0,
                       1,1,1,1,0), 5, 3)
rownames(pred.matrix) = c('int','X1','X2','X3','B')
colnames(pred.matrix) = c('E1','E2','E3')

get_SCLearner(k=3,
              q=5,
              pred.matrix=pred.matrix,
              gamma.EB=cbind(gamma.EB1, gamma.EB2, gamma.EB3),
              V.EB)
```

### Current Suggested Citation

Gu, T., Taylor, J.M.G. and Mukherjee, B. (2020). An ensemble
meta-prediction framework to integrate multiple regression models into a
current study. Manuscript in preparation.
