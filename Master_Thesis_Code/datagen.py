# This script was written by Michaël Adriaensen.
# For a master's thesis at Universiteit Antwerpen.
# Department of Computer Science
# Data Science
# 2019 - 2020

# The functions in this script are used to generate data for experiments on fairness interventions.
# The data is sampled from a distribution defined in the code.
# The distribution is specified using bayesian networks
# The library pgmpy is used for defining the bayesian networks and sampling from them
# The networks generally consider discrete variables. Our distributions feature continous variables as well.
# These continous variable distributions are generated after the discrete distributions.

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling,HamiltonianMC as HMC, LeapFrog, GradLogPDFGaussian
from pgmpy.factors.continuous import ContinuousFactor,RoundingDiscretizer
from pgmpy.factors.distributions import GaussianDistribution as JGD

from scipy.stats import norm, multivariate_normal
import numpy as np
import pandas as pd

def samplecontinuous(AY,samplesize=2000,contatt="C2",meana0=0,meana1=1,covy0=[1],covy1=[0.75]):
	'''' Sample the continous variables. 4 distributions are generated depending on discrete variables A and Y.
	The continous variables are causally influenced by A and Y, both of these discrete variables have two possible outcomes,
	so we need to generate 4 gaussian distributions per continous variable'''

	samplera0y0 = HMC(model=gengaussdist(meana0,covy0,"contatta0y0"), grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
	samplesa0y0 = samplera0y0.sample(initial_pos=np.array([meana0]), num_samples = samplesize,trajectory_length=2, stepsize=None)

	samplera1y0 = HMC(model=gengaussdist(meana1,covy0,"contatta1y0"), grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
	samplesa1y0 = samplera1y0.sample(initial_pos=np.array([meana1]), num_samples = samplesize,trajectory_length=2, stepsize=None)

	samplera0y1 = HMC(model=gengaussdist(meana0,covy1,"contatta0y1"), grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
	samplesa0y1 = samplera0y1.sample(initial_pos=np.array([meana0]), num_samples = samplesize,trajectory_length=2, stepsize=None)

	samplera1y1 = HMC(model=gengaussdist(meana1,covy1,"contatta1y1"), grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
	samplesa1y1 = samplera1y1.sample(initial_pos=np.array([meana1]), num_samples = samplesize,trajectory_length=2, stepsize=None)

	df = pd.concat([AY,samplesa0y0,samplesa1y0,samplesa0y1,samplesa1y1],axis=1)
	df[contatt] = 0.0

	# Select which of the 4 distribution we need the sample from
	for i, row in df.iterrows():
		if row["A"] == 0 and row["Y"] == 0:
			df.at[i,contatt] = row["contatta0y0"]
		elif row["A"] == 1 and row["Y"] == 0:
			df.at[i,contatt] = row["contatta1y0"]
		elif row["A"] == 0 and row["Y"] == 1:
			df.at[i,contatt] = row["contatta0y1"]
		else:
			df.at[i,contatt] = row["contatta1y1"]

	df.drop(['contatta0y0','contatta1y0','contatta0y1','contatta1y1','A','Y'],axis=1,inplace=True)
	return df
def gengaussdist(mean=0,cov=[1],attname="c2a0y0"):
	''' Generate a gaussian distribution based on the variables mean and cov. '''
	mean = np.array([mean])
	covariance = np.array([cov])
	return JGD([attname],mean,covariance)

def generateWysiwygData(samplesize=4000,filename="data/wysiwygdata4.csv"):
	''' We define a bayesian model based on the WYSIWYG model from the thesis.
		There are 6 C variables and 6 X variables. For both C and X the first four are discrete variables,
		the other two continous. The variable C1 is causally influencing Y to assure a certain level of 
		group unfairness in the data.'''

	wysiwygmodel = BayesianModel([('A', 'C1'), 
        ('A', 'C2'),('A', 'C3'),('A', 'C4'),('C1', 'Y'),
        ('Y', 'C2'),('Y', 'C3'),('Y', 'C4'),('A', 'X1'),('A', 'X2'),
        ('A', 'X3'),('A', 'X4'),('Y', 'X1'),('Y', 'X2'),('Y', 'X3'),('Y', 'X4')])

	cpd_a = TabularCPD(variable='A', variable_card=2,
		      values=[[0.5], [0.5]])

	cpd_y = TabularCPD(variable='Y', variable_card=2,
		values=[[0.65], [0.4],
				[0.35], [0.6]],
		evidence=['C1'],
		evidence_card=[2])

	cpd_c1 = TabularCPD(variable='C1', variable_card=2,
		values=[[0.85, 0.2],
				[0.15, 0.8]],
		evidence=['A'],
		evidence_card=[2])

	cpd_c2 = TabularCPD(variable='C2', variable_card=4,
		values=[[0.23, 0.27,0.25,0.20],
				[0.35, 0.23,0.24,0.15],
				[0.22, 0.27,0.25,0.25],
				[0.20, 0.23,0.26,0.40]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_c3 = TabularCPD(variable='C3', variable_card=2,
			values=[[0.52, 0.49,0.5,0.45],
					[0.48, 0.51,0.5,0.55]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_c4 = TabularCPD(variable='C4', variable_card=4,
			values=[[0.22, 0.25,0.25,0.37],
					[0.23, 0.25,0.26,0.21],
					[0.23, 0.25,0.25,0.22],
					[0.32, 0.25,0.24,0.20]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x1 = TabularCPD(variable='X1', variable_card=2,
			values=[[0.57, 0.48,0.52,0.38],
					[0.43, 0.52,0.48,0.62]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x2 = TabularCPD(variable='X2', variable_card=4,
			values=[[0.24, 0.28,0.26,0.19],
					[0.38, 0.22,0.24,0.15],
					[0.20, 0.28,0.26,0.23],
					[0.18, 0.22,0.24,0.43]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x3 = TabularCPD(variable='X3', variable_card=2,
			values=[[0.54, 0.48,0.52,0.4],
					[0.46, 0.52,0.48,0.6]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x4 = TabularCPD(variable='X4', variable_card=4,
			values=[[0.20, 0.25,0.24,0.40],
					[0.21, 0.25,0.28,0.21],
					[0.21, 0.25,0.24,0.21],
					[0.38, 0.25,0.24,0.18]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	wysiwygmodel.add_cpds(cpd_a, cpd_c1, cpd_c2,cpd_c3,cpd_c4,cpd_x1,cpd_x2,cpd_x3,cpd_x4,cpd_y)
	datasamples = BayesianModelSampling(wysiwygmodel)
	discframe = datasamples.forward_sample(samplesize)
	AY = discframe[["A","Y"]]

	C5 = samplecontinuous(AY,samplesize=samplesize,contatt="C5",meana0=1,meana1=1.2,covy0=[1],covy1=[0.9])
	C6 = samplecontinuous(AY,samplesize=samplesize,contatt="C6",meana0=2,meana1=1.8,covy0=[1],covy1=[0.95])

	X5 = samplecontinuous(AY,samplesize=samplesize,contatt="X5",meana0=1.1,meana1=1.4,covy0=[1.1],covy1=[0.95])
	X6 = samplecontinuous(AY,samplesize=samplesize,contatt="X6",meana0=1.9,meana1=1.5,covy0=[1],covy1=[1.1])

	discframe = pd.concat([discframe,C5,C6,X5,X6],axis=1)
	discframe.to_csv(path_or_buf=filename)

def generateWysiwygDataDI(samplesize=4000):
	''' same principle as generateWysiwygData(), but we have 3 continous variables and 3 discrete C and X variables.
	This distribution was used for the DI removal experiment, because IBM AIF360's DI removal only impacts continous variables. '''
	wysiwygmodel = BayesianModel([('A', 'C1'), 
        ('A', 'C2'),('A', 'C3'),('C1', 'Y'),
        ('Y', 'C2'),('Y', 'C3'),('A', 'X1'),('A', 'X2'),
        ('A', 'X3'),('Y', 'X1'),('Y', 'X2'),('Y', 'X3')])

	cpd_a = TabularCPD(variable='A', variable_card=2,
		      values=[[0.5], [0.5]])

	cpd_y = TabularCPD(variable='Y', variable_card=2,
		values=[[0.7], [0.35],
				[0.3], [0.65]],
		evidence=['C1'],
		evidence_card=[2])

	cpd_c1 = TabularCPD(variable='C1', variable_card=2,
		values=[[0.65, 0.3],
				[0.35, 0.7]],
		evidence=['A'],
		evidence_card=[2])

	cpd_c2 = TabularCPD(variable='C2', variable_card=4,
		values=[[0.24, 0.27,0.25,0.24],
				[0.28, 0.23,0.24,0.22],
				[0.24, 0.27,0.25,0.26],
				[0.24, 0.23,0.26,0.28]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_c3 = TabularCPD(variable='C3', variable_card=4,
			values=[[0.22, 0.25,0.25,0.37],
					[0.23, 0.25,0.26,0.21],
					[0.23, 0.25,0.25,0.22],
					[0.32, 0.25,0.24,0.20]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x1 = TabularCPD(variable='X1', variable_card=2,
			values=[[0.54, 0.48,0.52,0.45],
					[0.46, 0.52,0.48,0.55]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x2 = TabularCPD(variable='X2', variable_card=4,
			values=[[0.25, 0.27,0.26,0.23],
					[0.30, 0.23,0.24,0.23],
					[0.23, 0.27,0.26,0.23],
					[0.22, 0.23,0.24,0.31]],
		evidence=['A','Y'],
		evidence_card=[2,2])


	cpd_x3 = TabularCPD(variable='X3', variable_card=4,
			values=[[0.22, 0.25,0.25,0.30],
					[0.23, 0.25,0.26,0.24],
					[0.23, 0.25,0.24,0.24],
					[0.32, 0.25,0.25,0.22]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	wysiwygmodel.add_cpds(cpd_a, cpd_c1, cpd_c2,cpd_c3,cpd_x1,cpd_x2,cpd_x3,cpd_y)
	datasamples = BayesianModelSampling(wysiwygmodel)
	discframe = datasamples.forward_sample(samplesize)
	AY = discframe[["A","Y"]]

	C4 = samplecontinuous(AY,samplesize=samplesize,contatt="C4",meana0=5,meana1=6,covy0=[1],covy1=[1.8])
	C5 = samplecontinuous(AY,samplesize=samplesize,contatt="C5",meana0=1,meana1=2,covy0=[1],covy1=[0.9])
	C6 = samplecontinuous(AY,samplesize=samplesize,contatt="C6",meana0=4,meana1=5.3,covy0=[1],covy1=[0.95])

	X4 = samplecontinuous(AY,samplesize=samplesize,contatt="X4",meana0=5.5,meana1=6,covy0=[1.2],covy1=[1.4])
	X5 = samplecontinuous(AY,samplesize=samplesize,contatt="X5",meana0=1.1,meana1=1.7,covy0=[1.1],covy1=[1.0])
	X6 = samplecontinuous(AY,samplesize=samplesize,contatt="X6",meana0=4.5,meana1=5.1,covy0=[1],covy1=[1.1])

	discframe = pd.concat([discframe,C4,C5,C6,X4,X5,X6],axis=1)
	discframe.to_csv(path_or_buf="data/wysiwygdata5.csv")

def generateWysiwygFIDataOld(samplesize=4000,filename="data/preFIData.csv"):
	''' old version of the bayesian model for the Fair Inference experiment.
	Here Y still influences X to make modelling Y simpler.
	This is not suitable for FI.
	This model is unused in the experiments in the final thesis.
	'''
	wysiwygmodel = BayesianModel([('A', 'C1'), 
        ('A', 'C2'),('A', 'C3'),('A', 'C4'),('C1', 'Y'),
        ('Y', 'C2'),('Y', 'C3'),('Y', 'C4'),('A', 'X1'),('A', 'X2'),
        ('A', 'X3'),('A', 'X4'),('Y', 'X1'),('Y', 'X2'),('Y', 'X3'),('Y', 'X4'),('D1','X1'),('D1','X2'),('D2','X3'),('D3','X4')])

	cpd_a = TabularCPD(variable='A', variable_card=2,
		      values=[[0.5], [0.5]])

	cpd_d1 = TabularCPD(variable='D1', variable_card=2,
		      values=[[0.45], [0.55]])

	cpd_d2 = TabularCPD(variable='D2', variable_card=4,
		      values=[[0.22], [0.24],[0.28], [0.26]])
	cpd_d3 = TabularCPD(variable='D3', variable_card=2,
		      values=[[0.54], [0.46]])

	cpd_y = TabularCPD(variable='Y', variable_card=2,
		values=[[0.7], [0.3],
				[0.3], [0.7]],
		evidence=['C1'],
		evidence_card=[2])

	cpd_c1 = TabularCPD(variable='C1', variable_card=2,
		values=[[0.85, 0.2],
				[0.15, 0.8]],
		evidence=['A'],
		evidence_card=[2])

	cpd_c2 = TabularCPD(variable='C2', variable_card=4,
		values=[[0.23, 0.27,0.25,0.20],
				[0.35, 0.23,0.24,0.15],
				[0.22, 0.27,0.25,0.25],
				[0.20, 0.23,0.26,0.40]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_c3 = TabularCPD(variable='C3', variable_card=2,
			values=[[0.52, 0.49,0.5,0.45],
					[0.48, 0.51,0.5,0.55]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_c4 = TabularCPD(variable='C4', variable_card=4,
			values=[[0.22, 0.25,0.25,0.37],
					[0.23, 0.25,0.26,0.21],
					[0.23, 0.25,0.25,0.22],
					[0.32, 0.25,0.24,0.20]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x1 = TabularCPD(variable='X1', variable_card=2,
			values=[[0.38,0.40,0.42,0.44, 0.57,0.59,0.60,0.62], #GOOD
					[0.62,0.60,0.58,0.56, 0.43,0.41,0.40,0.38]],
		evidence=['A','Y','D1'],
		evidence_card=[2,2,2])

	cpd_x2 = TabularCPD(variable='X2', variable_card=4,
			values=[[0.30,0.28,0.27,0.25, 0.17,0.16,0.15,0.14],
					[0.24,0.26,0.26,0.27, 0.29,0.31,0.30,0.32], #GOOD 2
					[0.16,0.18,0.20,0.22, 0.35,0.37,0.38,0.40], #GOOD 1
					[0.30,0.28,0.27,0.26, 0.19,0.16,0.17,0.14]],
		evidence=['A','Y','D1'],
		evidence_card=[2,2,2])

	cpd_x3 = TabularCPD(variable='X3', variable_card=2,
			values=[[0.64,0.62,0.62,0.63, 0.60,0.58,0.58,0.59, 0.40,0.39,0.39,0.38, 0.38,0.35,0.35,0.37],
					[0.36,0.38,0.38,0.37, 0.40,0.42,0.42,0.41, 0.60,0.61,0.61,0.62, 0.62,0.65,0.65,0.63]], #GOOD
		evidence=['A','Y','D2'],
		evidence_card=[2,2,4])

	cpd_x4 = TabularCPD(variable='X4', variable_card=4,
			values=[[0.25,0.27,0.21,0.23, 0.10,0.12,0.07,0.09],
					[0.36,0.34,0.42,0.40, 0.60,0.58,0.64,0.62],#GOOD1
					[0.25,0.27,0.21,0.23, 0.10,0.12,0.07,0.09],
					[0.14,0.12,0.16,0.14, 0.20,0.18,0.22,0.20]],#GOOD2
		evidence=['A','Y','D3'],
		evidence_card=[2,2,2])


	wysiwygmodel.add_cpds(cpd_a, cpd_c1, cpd_c2,cpd_c3,cpd_c4,cpd_x1,cpd_x2,cpd_x3,cpd_x4,cpd_y,cpd_d1,cpd_d2,cpd_d3)
	datasamples = BayesianModelSampling(wysiwygmodel)
	discframe = datasamples.forward_sample(samplesize)
	AY = discframe[["A","Y"]]

	C5 = samplecontinuous(AY,samplesize=samplesize,contatt="C5",meana0=1,meana1=1.2,covy0=[1],covy1=[0.9])
	C6 = samplecontinuous(AY,samplesize=samplesize,contatt="C6",meana0=2,meana1=1.8,covy0=[1],covy1=[0.95])

	X5 = samplecontinuous(AY,samplesize=samplesize,contatt="X5",meana0=1.1,meana1=1.4,covy0=[1.1],covy1=[0.95])
	X6 = samplecontinuous(AY,samplesize=samplesize,contatt="X6",meana0=1.9,meana1=1.5,covy0=[1],covy1=[1.1])

	discframe = pd.concat([discframe,C5,C6,X5,X6],axis=1)
	ndf = discframe.reindex(axis=1,labels=['A','Y','C1','C2','C3','C4','C5','C6','X1','X2','X3','X4','X5','X6','D1','D2','D3'])
	ndf.to_csv(path_or_buf=filename)

def generateWysiwygFIData(samplesize=4000,filename="data/preFIData.csv"):
	''' The bayesian network that was used in the FI experiment.
	The edges between X and Y are flipped from the previous models,
	so X causally influences Y. The D variables are added to more closely approximate 
	the experiments from the 'Fair Inference on Outcomes' paper.  '''
	wysiwygmodel = BayesianModel([('A', 'C1'), 
        ('A', 'C2'),('A', 'C3'),('A', 'C4'),
        ('Y', 'C2'),('Y', 'C3'),('Y', 'C4'),('A', 'X1'),('A', 'X2'),
        ('A', 'X3'),('A', 'X4'),('X1', 'Y'),('X2', 'Y'),('X3', 'Y'),('X4', 'Y'),('D1','X1'),('D1','X2'),('D2','X3'),('D3','X4')])

	cpd_a = TabularCPD(variable='A', variable_card=2,
		      values=[[0.5], [0.5]])

	cpd_d1 = TabularCPD(variable='D1', variable_card=2,
		      values=[[0.45], [0.55]])

	cpd_d2 = TabularCPD(variable='D2', variable_card=4,
		      values=[[0.22], [0.24],[0.28], [0.26]])
	cpd_d3 = TabularCPD(variable='D3', variable_card=2,
		      values=[[0.54], [0.46]])

	ydists = computeYDist()

	cpd_y = TabularCPD(variable='Y', variable_card=2,
		values=[ydists[0],ydists[1]],
		evidence=['X1','X3','X2','X4'],
		evidence_card=[2,2,4,4])

	cpd_c1 = TabularCPD(variable='C1', variable_card=2,
		values=[[0.85, 0.2],
				[0.15, 0.8]],
		evidence=['A'],
		evidence_card=[2])

	cpd_c2 = TabularCPD(variable='C2', variable_card=4,
		values=[[0.23, 0.27,0.25,0.20],
				[0.35, 0.23,0.24,0.15],
				[0.22, 0.27,0.25,0.25],
				[0.20, 0.23,0.26,0.40]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_c3 = TabularCPD(variable='C3', variable_card=2,
			values=[[0.52, 0.49,0.5,0.45],
					[0.48, 0.51,0.5,0.55]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_c4 = TabularCPD(variable='C4', variable_card=4,
			values=[[0.22, 0.25,0.25,0.37],
					[0.23, 0.25,0.26,0.21],
					[0.23, 0.25,0.25,0.22],
					[0.32, 0.25,0.24,0.20]],
		evidence=['A','Y'],
		evidence_card=[2,2])

	cpd_x1 = TabularCPD(variable='X1', variable_card=2,
			values=[[0.38,0.40,0.60,0.62], #GOOD
					[0.62,0.60,0.40,0.38]],
		evidence=['A','D1'],
		evidence_card=[2,2])

	cpd_x2 = TabularCPD(variable='X2', variable_card=4,
			values=[[0.30,0.28, 0.15,0.14],
					[0.24,0.26, 0.30,0.32], #GOOD 2
					[0.16,0.18, 0.38,0.40], #GOOD 1
					[0.30,0.28, 0.17,0.14]],
		evidence=['A','D1'],
		evidence_card=[2,2])

	cpd_x3 = TabularCPD(variable='X3', variable_card=2,
			values=[[0.64,0.62,0.62,0.63, 0.38,0.35,0.35,0.37],
					[0.36,0.38,0.38,0.37, 0.62,0.65,0.65,0.63]], #GOOD
		evidence=['A', 'D2'],
		evidence_card=[2,4])

	cpd_x4 = TabularCPD(variable='X4', variable_card=4,
			values=[[0.25,0.27, 0.07,0.09],
					[0.36,0.34, 0.64,0.62],#GOOD1
					[0.25,0.27, 0.07,0.09],
					[0.14,0.12, 0.22,0.20]],#GOOD2
		evidence=['A','D3'],
		evidence_card=[2,2])


	wysiwygmodel.add_cpds(cpd_a, cpd_c1, cpd_c2,cpd_c3,cpd_c4,cpd_x1,cpd_x2,cpd_x3,cpd_x4,cpd_y,cpd_d1,cpd_d2,cpd_d3)
	datasamples = BayesianModelSampling(wysiwygmodel)
	discframe = datasamples.forward_sample(samplesize)
	AY = discframe[["A","Y"]]

	C5 = samplecontinuous(AY,samplesize=samplesize,contatt="C5",meana0=1,meana1=1.2,covy0=[1],covy1=[0.9])
	C6 = samplecontinuous(AY,samplesize=samplesize,contatt="C6",meana0=2,meana1=1.8,covy0=[1],covy1=[0.95])

	X5 = samplecontinuous(AY,samplesize=samplesize,contatt="X5",meana0=1.1,meana1=1.4,covy0=[1.1],covy1=[0.95])
	X6 = samplecontinuous(AY,samplesize=samplesize,contatt="X6",meana0=1.9,meana1=1.5,covy0=[1],covy1=[1.1])

	discframe = pd.concat([discframe,C5,C6,X5,X6],axis=1)
	ndf = discframe.reindex(axis=1,labels=['A','Y','C1','C2','C3','C4','C5','C6','X1','X2','X3','X4','X5','X6','D1','D2','D3'])
	ndf.to_csv(path_or_buf=filename)

def computeYDist():
	''' compute the distribution of Y in  generateWysiwygFIData2()'''
	y0dist = [0.0] * 64
	y1dist = [1.0] * 64

	for i in range(len(y0dist)):
		if i < len(y0dist)/2:
			y0dist[i] = 0.35
		else:
			y0dist[i] = 0.65
	for i in range(len(y0dist)):
		if i in list(range(16)):
			y0dist[i] -= 0.02
		elif i in list(range(16,32)):
			y0dist[i] += 0.03
		elif i in list(range(32,48)):
			y0dist[i] -= 0.02
		elif i in list(range(48,64)):
			y0dist[i] += 0.03
	yx2dist0 = list(range(4)) + list(range(16,20)) + list(range(32,36)) + list(range(48,52))
	yx2dist1 = list(range(4,8)) + list(range(20,24)) + list(range(36,40)) + list(range(52,56))
	yx2dist2 = list(range(8,12)) + list(range(24,28)) + list(range(40,44)) + list(range(56,60))
	yx2dist3 = list(range(12,16)) + list(range(28,32)) + list(range(44,48)) + list(range(60,64))
	for i in range(len(y0dist)):
		if i in yx2dist0:
			y0dist[i] -= 0.01
		elif i in yx2dist1:
			y0dist[i] += 0.01
		elif i in yx2dist2:
			y0dist[i] += 0.03
		elif i in yx2dist3:
			y0dist[i] -= 0.02

	for i in range(0,61,4):
		y0dist[i] -= 0.02
	for i in range(1,62,4):
		y0dist[i] += 0.02
	for i in range(2,63,4):
		y0dist[i] -= 0.02
	for i in range(3,64,4):
		y0dist[i] += 0.01

	for i in range(len(y0dist)):
		y1dist[i] -= y0dist[i]

	return(y0dist,y1dist)
if __name__ == "__main__":

	generateWysiwygFIData(samplesize=16000,filename="data/preFIData2.csv")