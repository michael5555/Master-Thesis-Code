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



# Compute the Path-Specific Effect with the given coefficients based on the equation
# by VanderWeele and Vansteelandt in "Odds ratios for mediation analysis for a dichotomous outcome" (2010)
compute_pse <- function(beta_x1,beta_x2,beta_x3,beta_x4,beta_x5,beta_x6,beta_y){
  
  x1_a = beta_x1[2]
  x2_a = beta_x2[2]
  x3_a = beta_x3[2]
  x4_a = beta_x4[2]
  x5_a = beta_x5[2]
  x6_a = beta_x6[2]

  y_x1 = beta_y[2]
  y_x2 = beta_y[3]
  y_x3 = beta_y[4]
  y_x4 = beta_y[5]
  y_x5 = beta_y[6]
  y_x6 = beta_y[7]
 
  pse_eff=exp(y_x1*x1_a + y_x2*x2_a + y_x3*x3_a + y_x4*x4_a + y_x5*x5_a + y_x6*x6_a)
  
  return(as.numeric(pse_eff))
}

