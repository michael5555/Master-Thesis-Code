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



# These functions return R Formula objects that represent the causal dependencies for each X variable
# and Y.
# Example: X1 depends on A and D1, fmla_x1 returns "X1 ~ A + D1"
fmla_x1 <- function(dat){
    x_M = colnames(dat)[c(2, 9)] # use A
    y_M = "X1"
    fmla_M = as.formula(paste(paste(y_M, "~ "), paste(x_M, collapse="+")))
    return(fmla_M)
}
fmla_x2 <- function(dat){
    x_M = colnames(dat)[c(2, 9)] # use A
    y_M = "X2"
    fmla_M = as.formula(paste(paste(y_M, "~ "), paste(x_M, collapse="+")))
    return(fmla_M)
}
fmla_x3 <- function(dat){
    x_M = colnames(dat)[c(2, 10)] # use A
    y_M = "X3"
    fmla_M = as.formula(paste(paste(y_M, "~ "), paste(x_M, collapse="+")))
    return(fmla_M)
}
fmla_x4 <- function(dat){
    x_M = colnames(dat)[c(2, 11)] # use A
    y_M = "X4"
    fmla_M = as.formula(paste(paste(y_M, "~ "), paste(x_M, collapse="+")))
    return(fmla_M)
}
fmla_x5 <- function(dat){
    x_M = colnames(dat)[2] # use A
    y_M = "X5"
    fmla_M = as.formula(paste(paste(y_M, "~ "), paste(x_M, collapse="+")))
    return(fmla_M)
}
fmla_x6 <- function(dat){
    x_M = colnames(dat)[2] # use A
    y_M = "X6"
    fmla_M = as.formula(paste(paste(y_M, "~ "), paste(x_M, collapse="+")))
    return(fmla_M)
}
fmla_y <- function(dat){
    #x_M = colnames(dat)[-1] # use A
    x_M = colnames(dat)[c(3:8)]
    y_M = "Y"
    fmla_M = as.formula(paste(paste(y_M, "~ "), paste(x_M, collapse="+")))
    return(fmla_M)
}


