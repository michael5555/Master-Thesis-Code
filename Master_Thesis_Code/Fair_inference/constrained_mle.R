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


# -------------------------------------- 
# Optimize a constrained logistic model
# --------------------------------------
constrained_mle <- function(dat,fmla_x1,fmla_x2,fmla_x3,fmla_x4,fmla_x5,fmla_x6,fmla_y,func,tau_u,tau_l){
  
  # Define the negative log likelihood function.
  eval_f <- function(beta, dat, y, x1, x2, x3, x4, x5, x6, xy, x_x1, x_x2, x_x3, x_x4, x_x5, x_x6, func, tau_u, tau_l){
    
    beta_x1 = beta[1:ncol(x_x1)]
    beta_x2 = beta[(ncol(x_x1) + 1):(ncol(x_x1) + ncol(x_x2))]
    beta_x3 = beta[(ncol(x_x1) + ncol(x_x2) + 1):(ncol(x_x1) + ncol(x_x2) + ncol(x_x3))]
    beta_x4 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4))]
    beta_x5 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5))]
    beta_x6 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+ncol(x_x6))]
    beta_y = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+ncol(x_x6)+1):length(beta)]
    
    names(beta_x1) = colnames(x_x1)
    names(beta_x2) = colnames(x_x2)
    names(beta_x3) = colnames(x_x3)
    names(beta_x4) = colnames(x_x4)
    names(beta_x5) = colnames(x_x5)
    names(beta_x6) = colnames(x_x6)
    names(beta_y) = colnames(xy)
    
    x1 = as.matrix(x1)
    x2 = as.matrix(x2)
    x3 = as.matrix(x3)
    x4 = as.matrix(x4)
    x5 = as.matrix(x5)
    x6 = as.matrix(x6)
    y = as.matrix(y)
    
    f =  t(x1-x_x1%*%beta_x1)%*%(x1-x_x1%*%beta_x1)+t(x2-x_x2%*%beta_x2)%*%(x2-x_x2%*%beta_x2)+t(x3-x_x3%*%beta_x3)%*%(x3-x_x3%*%beta_x3)+t(x4-x_x4%*%beta_x4)%*%(x4-x_x4%*%beta_x4)+t(x5-x_x5%*%beta_x5)%*%(x5-x_x5%*%beta_x5)+t(x6-x_x6%*%beta_x6)%*%(x6-x_x6%*%beta_x6)+sum(y*log(1+exp(-xy%*%beta_y))+(1-y)*log(1+exp(xy%*%beta_y)))
    return(f/nrow(dat))
  }
  
  # Define the inequality constraint.
  eval_g_ineq  <- function(beta, dat, y, x1, x2, x3, x4, x5, x6, xy, x_x1, x_x2, x_x3, x_x4, x_x5, x_x6, func, tau_u, tau_l){
    beta_x1 = beta[1:ncol(x_x1)]
    beta_x2 = beta[(ncol(x_x1) + 1):(ncol(x_x1) + ncol(x_x2))]
    beta_x3 = beta[(ncol(x_x1) + ncol(x_x2) + 1):(ncol(x_x1) + ncol(x_x2) + ncol(x_x3))]
    beta_x4 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4))]
    beta_x5 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5))]
    beta_x6 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+ncol(x_x6))]
    beta_y = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+ncol(x_x6)+1):length(beta)]
    
    names(beta_x1) = colnames(x_x1)
    names(beta_x2) = colnames(x_x2)
    names(beta_x3) = colnames(x_x3)
    names(beta_x4) = colnames(x_x4)
    names(beta_x5) = colnames(x_x5)
    names(beta_x6) = colnames(x_x6)
    names(beta_y) = colnames(xy)
    pse = func(beta_x1,beta_x2,beta_x3,beta_x4,beta_x5,beta_x6,beta_y)

    eval_g =  c(pse - tau_u, tau_l - pse)
    return(eval_g)
  }
  
  # Prepare the data.
  x_x1 = as.matrix(model.matrix(fmla_x1, data=model.frame(dat)))
  x_x2 = as.matrix(model.matrix(fmla_x2, data=model.frame(dat)))
  x_x3 = as.matrix(model.matrix(fmla_x3, data=model.frame(dat)))
  x_x4 = as.matrix(model.matrix(fmla_x4, data=model.frame(dat)))
  x_x5 = as.matrix(model.matrix(fmla_x5, data=model.frame(dat)))
  x_x6 = as.matrix(model.matrix(fmla_x6, data=model.frame(dat)))
  xy = as.matrix(model.matrix(fmla_y, data=model.frame(dat)))

  x1 = model.frame(fmla_x1,dat)[, 1]
  x2 = model.frame(fmla_x2,dat)[, 1]
  x3 = model.frame(fmla_x3,dat)[, 1]
  x4 = model.frame(fmla_x4,dat)[, 1]
  x5 = model.frame(fmla_x5,dat)[, 1]
  x6 = model.frame(fmla_x6,dat)[, 1]
  y = model.frame(fmla_y,dat)[, 1]
  
  # Initialize parameters.
  beta_x1_0 = rep(0, ncol(x_x1))
  beta_x2_0 = rep(0, ncol(x_x2))
  beta_x3_0 = rep(0, ncol(x_x3))
  beta_x4_0 = rep(0, ncol(x_x4))
  beta_x5_0 = rep(0, ncol(x_x5))
  beta_x6_0 = rep(0, ncol(x_x6))
  beta_y_0 = rep(0, ncol(xy))
  
  beta_start = c(beta_x1_0, beta_x2_0, beta_x3_0, beta_x4_0, beta_x5_0, beta_x6_0, beta_y_0)
  
  # Solve the constrained optimization problem.
  # We attempt to find the most likely set of coefficients for the X and Y models
  # that hold to the inequality constraint using the COBYLA algorithm.
  mle = nloptr(x0=beta_start, 
               eval_f=eval_f, 
               eval_g_ineq=eval_g_ineq,
               opts = list("algorithm"="NLOPT_LN_COBYLA","xtol_rel"=1.0e-8, "maxeval"=5000),
               dat=dat, y=y, x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6,
               xy=xy, x_x1=x_x1, x_x2=x_x2, x_x3=x_x3, x_x4=x_x4, x_x5=x_x5, x_x6=x_x6,
               func=func, tau_u=tau_u, tau_l=tau_l)
  
  # Return the coefficients of the model.
  beta = mle$solution
  beta_x1 = beta[1:ncol(x_x1)]
  beta_x2 = beta[(ncol(x_x1) + 1):(ncol(x_x1) + ncol(x_x2))]
  beta_x3 = beta[(ncol(x_x1) + ncol(x_x2) + 1):(ncol(x_x1) + ncol(x_x2) + ncol(x_x3))]
  beta_x4 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4))]
  beta_x5 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5))]
  beta_x6 = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+1):(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+ncol(x_x6))]
  beta_y = beta[(ncol(x_x1)+ncol(x_x2)+ncol(x_x3)+ncol(x_x4)+ncol(x_x5)+ncol(x_x6)+1):length(beta)]
  
  names(beta_x1) = colnames(x_x1)
  names(beta_x2) = colnames(x_x2)
  names(beta_x3) = colnames(x_x3)
  names(beta_x4) = colnames(x_x4)
  names(beta_x5) = colnames(x_x5)
  names(beta_x6) = colnames(x_x6)
  names(beta_y) = colnames(xy)
  
  return(list(beta_x1=beta_x1, 
              beta_x2=beta_x2, 
              beta_x3=beta_x3, 
              beta_x4=beta_x4, 
              beta_x5=beta_x5,
              beta_x6=beta_x6,
              beta_y=beta_y))
}