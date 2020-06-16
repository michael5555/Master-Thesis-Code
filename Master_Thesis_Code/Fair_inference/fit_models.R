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



fit_logistic <- function(dat, fmla){
  # Fit a logistic regression model based on the dataset and formula given.
  # Return the coefficients of the model.
  beta = glm(fmla, dat, family = "binomial")$coefficients
  return(beta)
}

fit_regression <- function(dat, fmla){
  # Fit a linear regression model based on the dataset and formula given.
  # Return the coefficients of the model.
  beta = lm(fmla, dat)$coefficients
  return(beta)
}

fit_models <- function(dat,fmla_x1, fmla_x2,fmla_x3,fmla_x4,fmla_x5,fmla_x6, fmla_y){
  # Fit the X variables using linear regression and Y using logistic regression.
  beta_x1 = fit_regression(dat, fmla_x1)
  beta_x2 = fit_regression(dat, fmla_x2)
  beta_x3 = fit_regression(dat, fmla_x3)
  beta_x4 = fit_regression(dat, fmla_x4)
  beta_x5 = fit_regression(dat, fmla_x5)
  beta_x6 = fit_regression(dat, fmla_x6)
  beta_y = fit_logistic(dat, fmla_y)
  
  # Return the sets of coefficients.
  return(list(beta_x1=beta_x1, 
              beta_x2=beta_x2, 
              beta_x3=beta_x3, 
              beta_x4=beta_x4, 
              beta_x5=beta_x5,
              beta_x6=beta_x6,
              beta_y=beta_y))
}

