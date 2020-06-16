# This script was written by Michaël Adriaensen.
# For a master's thesis at Universiteit Antwerpen.
# Department of Computer Science
# Data Science
# 2019 - 2020

# This script contains an experiment performed for the intervention analysis chapter of the thesis.
# The following scripts all contain one such experiment and have a very similar structure:
#   * reweighing.py
#   * disparateimpact.py
#   * lfr.py
# The fair inference experiment and its code is structured differently.
# In every one of these experiments we perform the following steps:
#   * Train a set of classification models. (Logistic Regression, SVM, Random Forest)
#   * Measure the accuracy and fairness according to a chosen fairness metric
#   * Perform a fairness intervention.
#   * Measure the accuracy and fairness again (using the same fairness metric)

# Fairness Intervention used in this script: 
#   LFR (Learning Fair Representations by Zemel et al., 2013)
# Fairness Metric used in this script: 
#   * Disparate Impact
#   * Consistency (as defined in "Learning Fair Representations")

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

from collections import defaultdict
import copy
import argparse

def prepareWysiwygData(fname="data/wysiwygdata4.csv",c_columns=["C1","C2","C3"]):
    ''' Read the dataset and transform into pandas dataframe, then remove unknown confounder data.'''
    df = pd.read_csv(fname,index_col=0)
    df.drop(c_columns,axis=1,inplace=True)
    return df

def prepareWysiwygDataForTraining(frame,x_columns,trainwithA=True):
    ''' split data into training and test sets, if we need the data separated on the domain of
    the sensitive attribute, also provide splits per A value. If intervention uses weighted instances,
    return those as wrll post intervention. '''

    Y = frame[['A','Y']]
    xa_columns = copy.deepcopy(x_columns)
    xa_columns.insert(0,"A")
    X = frame[xa_columns].copy()
    if 'weights' in frame.columns:
        X['weights'] = frame[['weights']]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0)

    Train_weights = None
    if 'weights' in frame.columns:
        Train_weights = X_train['weights']
    X_train = X_train[xa_columns]
    X_test = X_test[xa_columns]

    Y_train = Y_train[['Y']]
    Y_train = np.ravel(Y_train)

    X_testA0 = X_test[X_test.A == 0]
    X_testA1 = X_test[X_test.A == 1]
    Y_testA0 = Y_test[Y_test.A == 0]
    Y_testA1 = Y_test[Y_test.A == 1]
    Y_test = Y_test[['Y']]
    Y_test = np.ravel(Y_test)
    Y_testA0 = Y_testA0[['Y']]
    Y_testA0 = np.ravel(Y_testA0)
    Y_testA1 = Y_testA1[['Y']]
    Y_testA1 = np.ravel(Y_testA1)
    if not trainwithA:
        X_train = X_train[x_columns]
        X_test = X_test[x_columns]
        X_testA0 = X_testA0[x_columns]
        X_testA1 =X_testA1[x_columns]
        return (X_train,Y_train,X_test,Y_test,X_testA0,Y_testA0,X_testA1,Y_testA1,Train_weights)

    return (X_train,Y_train,X_test,Y_test,X_testA0,Y_testA0,X_testA1,Y_testA1,Train_weights)

def classifyFullFrame(frame,fittedmodel,x_columns,trainwithA=True):
    '''Classify the whole dataset (training + test) to get classification outcomes
    to compare for fairness comparisons. '''

    xa_columns = copy.deepcopy(x_columns)
    xa_columns.insert(0,"A")

    X = frame[xa_columns]
    if not trainwithA:
        X = X[x_columns]
    Ya = fittedmodel.predict(X)

    frame['Ya'] = Ya
    return frame

def trainWysiwygDataFromFrame(frame,model,x_columns,verbose=True,clffairness=True,trainwithA=True):
    ''' perform classication experiment and return accuracies needed.
    Whole test set or separated per A value.'''

    modelname = type(model).__name__
    clfdata = prepareWysiwygDataForTraining(frame,x_columns=x_columns,trainwithA=trainwithA)
    model.fit(clfdata[0], clfdata[1])
    Ya_test_data = model.predict(clfdata[2])
    Ya_test_dataA0 = model.predict(clfdata[4])
    Ya_test_dataA1 = model.predict(clfdata[6])

    accscore = accuracy_score(Ya_test_data, clfdata[3])
    accscoreA0 = accuracy_score(Ya_test_dataA0, clfdata[5])
    accscoreA1 = accuracy_score(Ya_test_dataA1, clfdata[7])

    clfframe = None
    if clffairness:
        clfframe = classifyFullFrame(frame,model,x_columns=x_columns,trainwithA=trainwithA)

    if verbose:
        print("\tACC: The accuracy of the {0} classifier is: {1}".format(modelname,accscore))
        print("\tACC: The accuracy of the {0} classifier for A == 0: {1}".format(modelname,accscoreA0))
        print("\tACC: The accuracy of the {0} classifier for A == 1: {1}".format(modelname,accscoreA1))

    return (modelname,accscore,accscoreA0,accscoreA1,clfframe)

def checkClassifierFairnessAndLFR(frame,dpoints,mname,x_columns,verbose=True,pre=True):
    ''' Measure fairness according to the metric using the value of A and the classification outcome.
    Results get added to a dictionary used to pass them to a function to generate graphs of the results.
    If we have not performed intervention, perform intervention and return post intervention data.'''

    xay_columns = copy.deepcopy(x_columns)
    xay_columns.extend(["A","Y"])

    ycols = copy.deepcopy(frame["Y"])
    tempframe = copy.deepcopy(frame)
    tempframe.drop(["Y"],axis=1,inplace=True)

    aifdf = BinaryLabelDataset(favorable_label=1.0,unfavorable_label=0.0, df=tempframe,label_names=['Ya'],
            protected_attribute_names=['A'])

    privileged_groups = [{'A': 1}]
    unprivileged_groups = [{'A': 0}]

    FairRep = LFR(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups,verbose=0,seed=0)

    metric_aifdf_train = BinaryLabelDatasetMetric(aifdf, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    if pre:
        if verbose:
            print("\n\tINTERVENTION: {}\n".format(type(FairRep).__name__))
            print("\t######### PRE {} ###########".format(type(FairRep).__name__))
            print("\tmean consistency between unprivileged and privileged groups = {}".format(metric_aifdf_train.consistency()))
            print("\tdisparate impact between unprivileged and privileged groups = {}\n".format(metric_aifdf_train.disparate_impact()))

        dpoints[mname]['PRE'][type(FairRep).__name__]['FAIRCONSIS'] = metric_aifdf_train.consistency()
        dpoints[mname]['PRE'][type(FairRep).__name__]['FAIRDI'] = metric_aifdf_train.disparate_impact()

        print("PRE CLASSIFICATION MATRIX")
        print("----------------")
        print("   |Y'=0  | Y'=1 |")
        print("----------------")
        print("A=0| {0} | {1} |".format(metric_aifdf_train.num_negatives(False),metric_aifdf_train.num_positives(False)))
        print("A=1| {0} | {1} |".format(metric_aifdf_train.num_negatives(True),metric_aifdf_train.num_positives(True)))
        print("----------------")

        dataset_transf_train = FairRep.fit_transform(aifdf)
        fairdf =  dataset_transf_train.convert_to_dataframe()[0]
        fairdf.drop(['Ya'],axis=1,inplace=True)
        ycols.reset_index(drop=True,inplace=True)
        fairdf.reset_index(drop=True,inplace=True)
        fairdf.insert(0,"Y",ycols)

        fairdf[xay_columns] = fairdf[xay_columns]
        fairdf[xay_columns] = fairdf[xay_columns].astype(int)
        return fairdf
    else:
        if verbose:
            print("\tMean consistency between unprivileged and privileged groups = {}\n".format(metric_aifdf_train.consistency()))
            print("\tdisparate impact between unprivileged and privileged groups = {}".format(metric_aifdf_train.disparate_impact()))
        dpoints[mname]['POST'][type(FairRep).__name__]['FAIRCONSIS'] = metric_aifdf_train.consistency()
        dpoints[mname]['POST'][type(FairRep).__name__]['FAIRDI'] = metric_aifdf_train.disparate_impact()

        print("POST CLASSIFICATION MATRIX")
        print("----------------")
        print("   |Y'=0 | Y'=1|")
        print("----------------")
        print("A=0| {0} | {1} |".format(metric_aifdf_train.num_negatives(False),metric_aifdf_train.num_positives(False)))
        print("A=1| {0} | {1} |".format(metric_aifdf_train.num_negatives(True),metric_aifdf_train.num_positives(True)))
        print("----------------")

        return frame

def runWysiwygLFRAnalysis(fname="wysiwygdata.csv",trainwithA=True):
    ''' run the full experiment, based on a parameters like train classifier with A (sensitive attribute)'''

    c_columns = ["C1","C2","C3","C4","C5","C6"]
    x_columns = ["X1","X2","X3","X4","X5","X6"]

    if fname == "wysiwygdata4.csv":
        c_columns = ["C1","C2","C3","C4","C5","C6"]
        x_columns = ["X1","X2","X3","X4"]

    wysiwygframe = prepareWysiwygData(fname=fname,c_columns=c_columns)
    privileged_groups = [{'A': 1}]
    unprivileged_groups = [{'A': 0}]


    LR = LogisticRegression(solver='liblinear',random_state=1)
    SVM = SVC(random_state=1,gamma='scale')
    RF = RandomForestClassifier(n_estimators=100,random_state=1)
    LFRmodels = [LR,SVM,RF]

    FairRep = LFR(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups,verbose=0,seed=0)
    lfrname = type(FairRep).__name__

    wysiwygpoints = {}
    for model in LFRmodels:
        modelname = type(model).__name__
        wysiwygpoints[modelname] = {}
        wysiwygpoints[modelname]['PRE'] = {}
        wysiwygpoints[modelname]['POST'] = {}
        wysiwygpoints[modelname]['PRE'][lfrname] = {}

        wysiwygpoints[modelname]['PRE'][lfrname]['ACC'] = 0.0
        wysiwygpoints[modelname]['PRE'][lfrname]['ACCA0'] = 0.0
        wysiwygpoints[modelname]['PRE'][lfrname]['ACCA1'] = 0.0

        wysiwygpoints[modelname]['PRE'][lfrname]['FAIRCONSIS'] = 0.0
        wysiwygpoints[modelname]['PRE'][lfrname]['FAIRDI'] = 0.0

        wysiwygpoints[modelname]['POST'][lfrname] = {}
        wysiwygpoints[modelname]['POST'][lfrname]['ACC'] = 0.0
        wysiwygpoints[modelname]['POST'][lfrname]['ACCA0'] = 0.0
        wysiwygpoints[modelname]['POST'][lfrname]['ACCA1'] = 0.0

        wysiwygpoints[modelname]['POST'][lfrname]['FAIRCONSIS'] = 0.0
        wysiwygpoints[modelname]['POST'][lfrname]['FAIRDI'] = 0.0

    for model in LFRmodels:
        modelname = type(model).__name__
        print("####### {} #########\n".format(modelname))
        preaccs = trainWysiwygDataFromFrame(wysiwygframe,model,x_columns=x_columns,trainwithA=trainwithA)
        wysiwygpoints[modelname]['PRE'][lfrname]['ACC'] = preaccs[1]
        wysiwygpoints[modelname]['PRE'][lfrname]['ACCA0'] = preaccs[2]
        wysiwygpoints[modelname]['PRE'][lfrname]['ACCA1'] = preaccs[3]

        lfrdata = checkClassifierFairnessAndLFR(preaccs[4],x_columns=x_columns,dpoints=wysiwygpoints,mname=modelname,verbose=True,pre=True)
        print("\t######### POST {} ###########".format(lfrname))

        postaccs = trainWysiwygDataFromFrame(lfrdata,model,x_columns=x_columns,trainwithA=trainwithA)
        wysiwygpoints[modelname]['POST'][lfrname]['ACC'] = postaccs[1]
        wysiwygpoints[modelname]['POST'][lfrname]['ACCA0'] = postaccs[2]
        wysiwygpoints[modelname]['POST'][lfrname]['ACCA1'] = postaccs[3]

        checkClassifierFairnessAndLFR(postaccs[4],x_columns=x_columns,dpoints=wysiwygpoints,mname=modelname,verbose=True,pre=False)

    return wysiwygpoints

def autolabel(rects,ax):
    """ pretty print the height (experiment result) of the bars in the bar chart part of the graph."""
    for rect in rects:
        height = "{:.2f}".format(rect.get_height())
        height = float(height)
        txt = ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        txt.set_fontsize(6)

def drawLFRGraphBar(wysiwygpoints,pname="plots/WysiwygLFRPlot.png",consis=True):
    ''' use matplotlib to draw graph of the results.
    Graph consists of two sub graphs, one graph plotting a the accuracy and fairness pre and post
    experiment on a 2 dimensional plane and then connecting the pre and post points per classifier.
    The second subgraph is a bar chart that depicts pre and post intervention accuracy per A value.
    This graph is the kind of graph used in the final thesis.'''

    fairlookupstr = 'FAIRCONSIS'
    gylabel = "FAIRNESS CONSISTENCY"
    if not consis:
        fairlookupstr = 'FAIRDI'
        gylabel = "FAIRNESS DISPARATE IMPACT"

    fig, axs = plt.subplots(2)

    xlist = []
    ylist = []

    anno = ["PRE","","PRE","","PRE",""]
    annocolors = ['b','b','r','r','g','g']

    xlistperA = []
    ylistperA = []

    annoperA = ["PRE","","PRE","","PRE","","PRE","","PRE","","PRE",""]
    annoperAcolors = ['b','b','r','r','g','g','y','y','m','m','k','k']


    lfrname = "LFR"
    lfrmodelnames = ["LogisticRegression","SVC","RandomForestClassifier"]

    for modelname in lfrmodelnames:
        xlist.append(wysiwygpoints[modelname]['PRE'][lfrname]['ACC'])
        ylist.append(wysiwygpoints[modelname]['PRE'][lfrname][fairlookupstr])
        xlist.append(wysiwygpoints[modelname]['POST'][lfrname]['ACC'])
        ylist.append(wysiwygpoints[modelname]['POST'][lfrname][fairlookupstr])

        xlistperA.append(wysiwygpoints[modelname]['PRE'][lfrname]['ACCA0'])
        ylistperA.append(wysiwygpoints[modelname]['PRE'][lfrname][fairlookupstr])
        xlistperA.append(wysiwygpoints[modelname]['POST'][lfrname]['ACCA0'])
        ylistperA.append(wysiwygpoints[modelname]['POST'][lfrname][fairlookupstr])
        xlistperA.append(wysiwygpoints[modelname]['PRE'][lfrname]['ACCA1'])
        ylistperA.append(wysiwygpoints[modelname]['PRE'][lfrname][fairlookupstr])
        xlistperA.append(wysiwygpoints[modelname]['POST'][lfrname]['ACCA1'])
        ylistperA.append(wysiwygpoints[modelname]['POST'][lfrname][fairlookupstr])
    for i in range(0, len(xlist), 2):
        c = annocolors[i]
        dcode = c + 'o-'
        axs[0].plot(xlist[i:i+2], ylist[i:i+2], dcode)

    for i,txt in enumerate(anno):
        text = axs[0].annotate(txt,(xlist[i],ylist[i]))
        text.set_fontsize(6)

    axs[0].set(xlabel="CLASSIFIER ACCURACY")
    axs[0].set(ylabel=gylabel)
    axs[0].legend(["LR","SVM","RF"],fontsize=7,loc='best')
  
    n_groups = 6
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    xlistper0 = xlistperA[0:2]
    xlistper1 = xlistperA[2:4]
    xlistper0.extend(xlistperA[4:6])
    xlistper1.extend(xlistperA[6:8])
    xlistper0.extend(xlistperA[8:10])
    xlistper1.extend(xlistperA[10:12])

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
    axs[1].set_xticklabels(('PRELR', 'POSTLR', 'PRESVM', 'POSTSVM','PRERF','POSTRF'))
    axs[1].legend(fontsize=6,loc='best')

    autolabel(rects1,axs[1])
    autolabel(rects2,axs[1])
    autolabel(rects3,axs[1])

    plt.tight_layout()
    plt.savefig(pname)
    plt.clf()
    
def drawLFRGraph(wysiwygpoints,pname="plots/WysiwygLFRPlot.png",consis=True):
    ''' use matplotlib to draw graphs of the results over the whole test set or the test set separated by A value.'''

    fairlookupstr = 'FAIRCONSIS'
    gylabel = "FAIRNESS CONSISTENCY"
    if not consis:
        fairlookupstr = 'FAIRDI'
        gylabel = "FAIRNESS DISPARATE IMPACT"
       
    fig, axs = plt.subplots(2)

    xlist = []
    ylist = []

    anno = ["PRE","","PRE","","PRE",""]
    annocolors = ['b','b','r','r','g','g']

    xlistperA = []
    ylistperA = []

    annoperA = ["PRE","","PRE","","PRE","","PRE","","PRE","","PRE",""]
    annoperAcolors = ['b','b','r','r','g','g','y','y','m','m','k','k']

    lfrname = "LFR"
    lfrmodelnames = ["LogisticRegression","SVC","RandomForestClassifier"]

    for modelname in lfrmodelnames:
        xlist.append(wysiwygpoints[modelname]['PRE'][lfrname]['ACC'])
        ylist.append(wysiwygpoints[modelname]['PRE'][lfrname][fairlookupstr])
        xlist.append(wysiwygpoints[modelname]['POST'][lfrname]['ACC'])
        ylist.append(wysiwygpoints[modelname]['POST'][lfrname][fairlookupstr])

        xlistperA.append(wysiwygpoints[modelname]['PRE'][lfrname]['ACCA0'])
        ylistperA.append(wysiwygpoints[modelname]['PRE'][lfrname][fairlookupstr])
        xlistperA.append(wysiwygpoints[modelname]['POST'][lfrname]['ACCA0'])
        ylistperA.append(wysiwygpoints[modelname]['POST'][lfrname][fairlookupstr])
        xlistperA.append(wysiwygpoints[modelname]['PRE'][lfrname]['ACCA1'])
        ylistperA.append(wysiwygpoints[modelname]['PRE'][lfrname][fairlookupstr])
        xlistperA.append(wysiwygpoints[modelname]['POST'][lfrname]['ACCA1'])
        ylistperA.append(wysiwygpoints[modelname]['POST'][lfrname][fairlookupstr])

    for i in range(0, len(xlist), 2):
        c = annocolors[i]
        dcode = c + 'o-'
        axs[0].plot(xlist[i:i+2], ylist[i:i+2], dcode)

    for i,txt in enumerate(anno):
        text = axs[0].annotate(txt,(xlist[i],ylist[i]))
        text.set_fontsize(6)

    axs[0].set(xlabel="CLASSIFIER ACCURACY")
    axs[0].set(ylabel=gylabel)
    axs[0].legend(["LR","SVM","RF"],fontsize=7,loc='best')
  
    for i in range(0, len(xlistperA), 2):
        c = annoperAcolors[i]
        dcode = c + 'o-'
        axs[1].plot(xlistperA[i:i+2], ylistperA[i:i+2], dcode)

        
    for i,txt in enumerate(annoperA):
        text = axs[1].annotate(txt,(xlistperA[i],ylistperA[i]))
        text.set_fontsize(6)

    axs[1].set(xlabel="CLASSIFIER ACCURACY")
    axs[1].set(ylabel=gylabel)
    axs[1].legend(["LR,A=0","LR,A=1","SVM,A=0","SVM,A=1","RF,A=0","RF,A=1"],fontsize=7,loc='best')

    plt.tight_layout()
    plt.savefig(pname)
    plt.clf()

if __name__ == "__main__":
    wysiwygpoints = runWysiwygLFRAnalysis(fname="data/wysiwygdata4.csv",trainwithA=True)
    wysiwygpoints2 = runWysiwygLFRAnalysis(fname="data/wysiwygdata4.csv",trainwithA=False)
    drawLFRGraphBar(wysiwygpoints,"plots/WysiwygLFRPlot.png")
    drawLFRGraphBar(wysiwygpoints2,"plots/WysiwygLFRWithoutAPlot.png")
    drawLFRGraphBar(wysiwygpoints,"plots/WysiwygLFRDIPlot.png",consis=False)
    drawLFRGraphBar(wysiwygpoints2,"plots/WysiwygLFRDIWithoutAPlot.png",consis=False)

