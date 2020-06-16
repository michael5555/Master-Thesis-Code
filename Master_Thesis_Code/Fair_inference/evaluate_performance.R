# This script was written by Michaël Adriaensen.
# For a master's thesis at Universiteit Antwerpen.
# Department of Computer Science
# Data Science
# 2019 - 2020

# This script was adapted from the source code for Nabi and Shpitzer's experiments in the paper
# 'Fair Inference on Outcomes'. The script was adapted to perform fair inference on the data generated
# from the function generateWysiwygFIData() in the datagen.py script.
# The orginal code written by Razieh Nabi can be found in the following repository:
# https://github.com/raziehna/fair-inference-on-outcomes.git


sample_norm <- function(x, beta, cov_x, ndraws){
  # Sample ndraws number of samples from a gauss distribution.

  # Define the standard deviation as the matrix multiplication
  # of the  transpose of the coefficients of the model of the variable we sample from,
  # the covariance based on the matrix representation of the test data of the dependencies of the variable we want to sample
  # and a non transposed matrix of the same coefficients of the first factor.
  sd = t(beta)%*%cov_x%*%beta
  if (nrow(x) == 1){
    # define the mean as the linear regression model prediction of x using the fair/constrained model
    y = sum(x*beta)
    draws = rnorm(ndraws, y, sd)
  }else{
    x = as.matrix(x)
    y = x%*%beta
    draws = apply(y, 1, function(r) rnorm(ndraws, r, sd))
    draws = as.vector(t(as.matrix(draws)))
  }
  return(draws)
}

sample_opt <- function(x, y, beta, ndraws) {
  # this function is not used in the final experiments of the thesis.
  beta_x1 = beta$beta_x1
  beta_x2 = beta$beta_x2
  beta_x3 = beta$beta_x3
  beta_x4 = beta$beta_x4
  beta_x5 = beta$beta_x5
  beta_x6 = beta$beta_x6
  beta_y = beta$beta_y

  # X1 | A
  x_x1 = x[, c(1, 2, 9)]
  cov_x1 = cov(x_x1)
  cov_x2 = cov_x1

  x_x3 = x[, c(1, 2, 10)]
  cov_x3 = cov(x_x3)

  x_x4 = x[, c(1, 2, 11)]
  cov_x4 = cov(x_x4)

  x_x5 = x[, c(1, 2)]
  cov_x5 = cov(x_x5)
  cov_x6 = cov_x5

  X1c = c()
  X2c = c()
  X3c = c()
  X4c = c()
  X5c = c()
  X6c = c()
  for (i in 1:nrow(x)){

    yx1_s = sample_norm(x_x1[i, ], beta_x1, cov_x1, ndraws)
    yx2_s = sample_norm(x_x1[i, ], beta_x2, cov_x2, ndraws)
    yx3_s = sample_norm(x_x1[i, ], beta_x3, cov_x3, ndraws)
    yx4_s = sample_norm(x_x1[i, ], beta_x4, cov_x4, ndraws)
    yx5_s = sample_norm(x_x1[i, ], beta_x5, cov_x5, ndraws)
    yx6_s = sample_norm(x_x1[i, ], beta_x6, cov_x6, ndraws)

    yx1_sm = mean(yx1_s)
    yx2_sm = mean(yx2_s)
    yx3_sm = mean(yx3_s)
    yx4_sm = mean(yx4_s)
    yx5_sm = mean(yx5_s)
    yx6_sm = mean(yx6_s)

    X1c = c(X1c,yx1_sm)
    X2c = c(X2c,yx2_sm)
    X3c = c(X3c,yx3_sm)
    X4c = c(X4c,yx4_sm)
    X5c = c(X5c,yx5_sm)
    X6c = c(X6c,yx6_sm)

  }
  #xY = t(matrix(rep(as.numeric(x[i, ]), ndraws), nrow = ncol(x)))
  xY = data.frame(Y=y,A=x[ , 2],X1=X1c,X2=X2c,X3=X3c,X4=X4c,X5=X5c,X6=X6c)

  return(xY)
}

pred_constrain <- function(x, beta, ndraws){
  # predict the logistic regression outcome for the constrained model by converting
  # the test data to a constrained model/ fair representation.

  beta_x1 = beta$beta_x1
  beta_x2 = beta$beta_x2
  beta_x3 = beta$beta_x3
  beta_x4 = beta$beta_x4
  beta_x5 = beta$beta_x5
  beta_x6 = beta$beta_x6
  beta_y = beta$beta_y
  
  # X1 | A
  x_x1 = x[, c(1, 2, 9)]
  cov_x1 = cov(x_x1)
  cov_x2 = cov_x1

  x_x3 = x[, c(1, 2, 10)]
  cov_x3 = cov(x_x3)

  x_x4 = x[, c(1, 2, 11)]
  cov_x4 = cov(x_x4)

  x_x5 = x[, c(1, 2)]
  cov_x5 = cov(x_x5)
  cov_x6 = cov_x5

  
  ypred = c()
  for (i in 1:nrow(x)){

    # sample fair data for all the X variables
    yx1_s = sample_norm(x_x1[i, ], beta_x1, cov_x1, ndraws)
    yx2_s = sample_norm(x_x1[i, ], beta_x2, cov_x2, ndraws)
    yx3_s = sample_norm(x_x1[i, ], beta_x3, cov_x3, ndraws)
    yx4_s = sample_norm(x_x1[i, ], beta_x4, cov_x4, ndraws)
    yx5_s = sample_norm(x_x1[i, ], beta_x5, cov_x5, ndraws)
    yx6_s = sample_norm(x_x1[i, ], beta_x6, cov_x6, ndraws)

    xY = t(matrix(rep(as.numeric(x[i,c(1,3:8) ]), ndraws), nrow = ncol(x[,c(1,3:8)])))

    xY[, 2] = yx1_s
    xY[, 3] = yx2_s
    xY[, 4] = yx3_s
    xY[, 5] = yx4_s
    xY[, 6] = yx5_s
    xY[, 7] = yx6_s

    xY = as.matrix(xY)

    # Perform the logistic regression using the fair coefficients
    yY_hat = 1/(1+exp(-xY%*%beta_y))

    # return the mean of the ndraw number of prediction as the final prediction for this row.
    yhat = mean(yY_hat)
    ypred = c(ypred, yhat)
  }

  # return constrained model predictions for all instances
  return(ypred)
}

contingency_table <- function(x, y, beta, delta, ndraws, type){
  # Use tables to measure accuracy scores, DI and other metrics.
  if (type == "constrained"){
    ypred = pred_constrain(x, beta, ndraws)
  }else{
    #x = as.matrix(x)
    #ypred = 1/(1+exp(-x%*%beta))
    x_unc = x[c(1, 3:8)]

    x_unc = as.matrix(x_unc)
    ypred = 1/(1+exp(-x_unc%*%beta))

  }

  # If logistic regression prediction is above delta threshold:
  # predict Y outcome: 1
  # else predict Y outcome: 0
  idx = which(ypred > delta)
  if (length(idx) == length(y) | length(idx) == 0) return(0)
  ypred[idx] = 1
  ypred[-idx] = 0

  dit = table(A = x[, 2], Predictions = ypred)
  priv = dit[2, 2]/(dit[2, 1] + dit[2, 2])
  unpriv = dit[1, 2]/(dit[1, 2] + dit[1, 1])
  print("DI:")
  print(unpriv/priv)

  print(dit)
  t = table(Predictions = ypred, TrueLabels = y)
  tpr = t[2, 2]/(t[1, 2] + t[2, 2])
  fnr = t[1, 2]/(t[1, 2] + t[2, 2])
  fpr = t[2, 1]/(t[1, 1] + t[2, 1])
  tnr = t[1, 1]/(t[1, 1] + t[2, 1])
  print(t)

  ya = data.frame(A=x[, 2],Y=y,YPred=ypred)

  ya0 = ya[ya$A == 0, ]
  ya1 = ya[ya$A == 1, ]

  t0 = table(Predictions = ya0$YPred, TrueLabels = ya0$Y)
  t1 = table(Predictions = ya1$YPred, TrueLabels = ya1$Y)


  # compute accuracy per valye in the sensitive attribute domain. A = 0 or A = 1
  acc0 = ( t0[2, 2] + t0[1, 1])/sum(t0)
  acc1 = ( t1[2, 2] + t1[1, 1])/sum(t1)

  # return all computed metrics
  return(list(t = t, 
              tp = t[2, 2],
              fn = t[1, 2], 
              fp = t[2, 1], 
              tn = t[1, 1], 
              tpr = tpr, 
              fnr = fnr, 
              fpr = fpr, 
              tnr = tnr,
              priv = priv,
              unpriv = unpriv,
              acc0 = acc0,
              acc1 = acc1))
}

compute_accuracy <- function(perf){
  # compute model accuracy.
  accuracy = (perf$tp + perf$tn)/sum(perf$t)
  return(accuracy)
}

compute_di <- function(perf){
  # compute model DI.
  di = (perf$unpriv/perf$priv)
  return(di)
}





