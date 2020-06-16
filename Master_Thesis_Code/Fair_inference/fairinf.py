# This script was written by Michaël Adriaensen.
# For a master's thesis at Universiteit Antwerpen.
# Department of Computer Science
# Data Science
# 2019 - 2020

# This python script was adapted from the source code for Nabi and Shpitzer's experiments in the paper
# 'Fair Inference on Outcomes'. The script was adapted to perform fair inference on the data generated
# from the function generateWysiwygFIData() in the datagen.py script.
# The orginal code written by Razieh Nabi can be found in the following repository:
# https://github.com/raziehna/fair-inference-on-outcomes.git


# This script functions as the main function where we combine the functions defined
# in the other fair inference script to perform the experiments.
# The functions in R are converted to python3 functions using rpy2.
# This was run using:
# * R version 3.6.3 (2020-02-29) 
# * Python 3.7.3 (default, Mar 27 2019, 22:13:19) [Clang 10.0.1 (clang-1001.0.46.3)] on darwin
# * rpy2 3.3.2

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover, LFR, OptimPreproc
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from collections import defaultdict
import copy
import argparse

r_source = robjects.r['source']
r_source("formulas.R")
r_source("fit_models.R")
r_source("compute_pse.R")
r_source("constrained_mle.R")
r_source("evaluate_performance.R")
robjects.r("library(nloptr)")
robjects.r("set.seed(0)")

def prepareWysiwygData(fname="../data/preFIData.csv",c_columns=["C1","C2","C3","C4","C5","C6"]):
    # import dataset and remove C columns.
    df = pd.read_csv(fname,index_col=0)
    df.drop(c_columns,axis=1,inplace=True)
    return df

def sampleWysiwygData(df,k=12000):
    ''' Use first k rows as training data and the rest of the rows as test data.'''
    dat = df[:k]
    test_dat = df[k:]

    return(dat,test_dat)

def getFormulae(r_dat):
    ''' Get the R Formula objects for the X and Y models.'''
    fmla_x1 = robjects.r['fmla_x1']
    fmla_x2 = robjects.r['fmla_x2']
    fmla_x3 = robjects.r['fmla_x3']
    fmla_x4 = robjects.r['fmla_x4']
    fmla_x5 = robjects.r['fmla_x5']
    fmla_x6 = robjects.r['fmla_x6']
    fmla_y = robjects.r['fmla_y']

    return((fmla_x1(r_dat),fmla_x2(r_dat),fmla_x3(r_dat),fmla_x4(r_dat),fmla_x5(r_dat),fmla_x6(r_dat), fmla_y(r_dat)))

def fitModels(dat,fmls):
    ''' fit the X(linear) and Y(logistic) regression models.'''
    fit_models = robjects.r['fit_models']
    return fit_models(dat,fmls[0],fmls[1],fmls[2],fmls[3],fmls[4],fmls[5],fmls[6])

def constrainedMLE(dat,fmls,tau_u,tau_l):
    ''' Perform the MLE Optimization for the constrained model.'''
    constrained_mle = robjects.r['constrained_mle']
    func = robjects.r['compute_pse']
    return constrained_mle(dat,fmls[0],fmls[1],fmls[2],fmls[3],fmls[4],fmls[5],fmls[6],func,tau_u,tau_l)

def computePseLm(beta_lm):
    ''' Compute the path-specific effect for the unconstrained model.'''
    robjects.globalenv['beta_lm'] = beta_lm
    pse_lm = robjects.r('''compute_pse(beta_lm$beta_x1,beta_lm$beta_x2,beta_lm$beta_x3,
                                    beta_lm$beta_x4,beta_lm$beta_x5,beta_lm$beta_x6,beta_lm$beta_y)''')
    return pse_lm

def computePseOpt(beta_opt):
    ''' Compute the path-specific effect for the constrained model.'''
    robjects.globalenv['beta_opt'] = beta_opt
    pse_opt = robjects.r('''compute_pse(beta_opt$beta_x1,beta_opt$beta_x2,beta_opt$beta_x3,
                                    beta_opt$beta_x4,beta_opt$beta_x5,beta_opt$beta_x6,beta_opt$beta_y)''')
    return pse_opt

def prepare_evaluation(r_test_dat):
    ''' separate the training attributes and the outcomes for the classification experiment.'''
    x = robjects.r('''data.frame("intercept" = 1, r_test_dat[, -1])''')
    y = robjects.r('''r_test_dat[, 1]''')

    return(x,y)

def evaluate_lm(x,y,beta_lm,delta_lm=0.47,ndraws=100):
    ''' Perform classification experiment for the unconstrained model.'''
    robjects.globalenv['delta_lm'] = delta_lm
    robjects.globalenv['beta_lm'] = beta_lm
    robjects.globalenv['ndraws'] = ndraws
    robjects.globalenv['x'] = x
    robjects.globalenv['y'] = y

    lm_eval = robjects.r('''
        # regular fit
        perf_lm = contingency_table(x, y, beta_lm$beta_y, delta_lm, ndraws, type="no-constrain")
        return(list(compute_accuracy(perf_lm),compute_di(perf_lm),perf_lm$acc0,perf_lm$acc1))
        ''')
    return lm_eval

def evaluate_opt(x,y,beta_opt,delta_opt=0.4722,ndraws=100):
    ''' Perform classification experiment for the constrained model.'''
    robjects.globalenv['delta_opt'] = delta_opt
    robjects.globalenv['beta_opt'] = beta_opt
    robjects.globalenv['ndraws'] = ndraws
    robjects.globalenv['x'] = x
    robjects.globalenv['y'] = y

    opt_eval = robjects.r('''
        # constrained fit
        perf_opt = contingency_table(x, y, beta_opt, delta_opt, ndraws, type="constrained")
        return(list(compute_accuracy(perf_opt),compute_di(perf_opt),perf_opt$acc0,perf_opt$acc1))
        ''')

    return opt_eval
def sample_opt(x,y,beta_opt,ndraws):
    ''' Function not used in final experiments.'''

    robjects.globalenv['beta_opt'] = beta_opt
    robjects.globalenv['ndraws'] = ndraws
    robjects.globalenv['x'] = x
    robjects.globalenv['y'] = y

    return robjects.r(''' sample_opt(x,y,beta_opt,ndraws) ''')

def drawDIGraphBar(lm_points,opt_points,pname="../plots/WysiwygFIPlotFIX.png"):
    ''' Draw graph of the results.'''

    fig, axs = plt.subplots(2)

    xlist = []
    ylist = []

    anno = ["PRE","POST"]
    annocolors = ['b','b']

    xlist.append(lm_points[0])
    ylist.append(lm_points[1])
    xlist.append(opt_points[0])
    ylist.append(opt_points[1])

    for i in range(0, len(xlist), 2):
        c = annocolors[i]
        dcode = c + 'o-'
        axs[0].plot(xlist[i:i+2], ylist[i:i+2], dcode)

    for i,txt in enumerate(anno):
        text = axs[0].annotate(txt,(xlist[i],ylist[i]))
        text.set_fontsize(6)

    axs[0].set(xlabel="CLASSIFIER ACCURACY")
    axs[0].set(ylabel="FAIRNESS DISPARATE IMPACT")
    axs[0].legend(["LR","SVM","RF"],fontsize=7,loc='best')
  
    n_groups = 2
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    xlistper0 = [lm_points[2],opt_points[2]]
    xlistper1 = [lm_points[3],opt_points[3]]

    rects1 = axs[1].bar(index, xlist, bar_width,
    alpha=opacity,
    color='c',
    label='ACC')

    rects2 = axs[1].bar(index + bar_width, xlistper0, bar_width,
    alpha=opacity,
    color='b',
    label='ACC, A=0')

    rects3 = axs[1].bar(index + 2*bar_width, xlistper1, bar_width,
    alpha=opacity,
    color='g',
    label='ACC, A=1')

    axs[1].set(xlabel='Method')
    axs[1].set(ylabel='Accuracy Scores')
    axs[1].set_title('Accuracy per intervention method per A value')
    axs[1].set_xticks(index)
    axs[1].set_xticklabels(('PRE', 'POST'))
    axs[1].legend(fontsize=6,loc='best')

    autolabel(rects1,axs[1])
    autolabel(rects2,axs[1])
    autolabel(rects3,axs[1])

    plt.tight_layout()
    plt.savefig(pname)
    plt.clf()

def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = "{:.2f}".format(rect.get_height())
        height = float(height)
        txt = ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        txt.set_fontsize(6)

if __name__ == "__main__":
    df = prepareWysiwygData(fname="../data/preFIData2.csv")
    t = sampleWysiwygData(df)
    dat = t[0]
    test_dat = t[1]

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_dat = robjects.conversion.py2rpy(dat)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_test_dat = robjects.conversion.py2rpy(test_dat)

    robjects.globalenv['r_dat'] = r_dat
    r_dat = robjects.r('''r_dat[, c("Y","A","X1","X2","X3","X4","X5","X6","D1","D2","D3")]''')
    robjects.globalenv['r_dat'] = r_dat

    robjects.globalenv['r_test_dat'] = r_test_dat
    r_test_dat = robjects.r('''r_test_dat[, c("Y","A","X1","X2","X3","X4","X5","X6","D1","D2","D3")]''')
    robjects.globalenv['r_test_dat'] = r_test_dat



    # set upper and lower bounds for the allowed PSE
    l_u = 1.05
    l_l = 0.95
    tau_u = min(l_u, 1/l_l)
    tau_l = max(l_l, 1/l_u)
    formulae = getFormulae(r_dat)

    # Get the coefficients for both unconstrained and constrained models.
    beta_lm = fitModels(r_dat,formulae)
    beta_opt = constrainedMLE(r_dat,formulae,tau_u,tau_l)

    # Show the PSE for both models to verify MLE optimization forces PSE between set bounds
    # for the constrained model..

    print(computePseLm(beta_lm))
    print(computePseOpt(beta_opt))

    ndraws = 100

    xy = prepare_evaluation(r_test_dat)
    x = xy[0]
    y = xy[1]

    lm_res = evaluate_lm(x,y,beta_lm)
    opt_res = evaluate_opt(x,y,beta_opt)

    lm_acc = np.asarray(lm_res[0])[0]
    lm_di = np.asarray(lm_res[1])[0]
    lm_acc0 = np.asarray(lm_res[2])[0]
    lm_acc1 = np.asarray(lm_res[3])[0]

    opt_acc = np.asarray(opt_res[0])[0]
    opt_di = np.asarray(opt_res[1])[0]
    opt_acc0 = np.asarray(opt_res[2])[0]
    opt_acc1 = np.asarray(opt_res[3])[0]

    lm_points = [lm_acc,lm_di,lm_acc0,lm_acc1]
    opt_points = [opt_acc,opt_di,opt_acc0,opt_acc1]

    # Results for Nabi's Adult dataset experiment first 16000 rows
    # training size 12000 rows
    # test size 4000
    lm_adult_points = [0.8115,0.1805441,0.9061288,0.7665068]
    lm_adult_pse = 3.69917890863797
    opt_adult_points = [0.721,1.021968,0.8347556,0.6669126]
    opt_adult_pse = 1.00705900862485

    drawDIGraphBar(lm_points,opt_points)
    drawDIGraphBar(lm_adult_points,opt_adult_points,pname="../plots/AdultFIPlot.png")
