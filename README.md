# Master Thesis Code
This repository holds all the code written for the experiments in Michaël Adriaensen's Master Thesis 'Choosing Suitable Fairness Metrics And Interventions'.

## How to Use
You should not change the directory structure.

All of the scripts that perform the experiments were written in python 3.

The script datagen.py holds the functions used to generate the datasets used in the experiments from the thesis.
These datasets get added in the data/ directory.
You can run datagen.py, but this will change the results that the intervention scripts will show when you run those.

These are the scripts used to perform the experiments in the thesis:

  * reweighing.py (reweighing experiment)
  * di.py (Disparate impact experiment)
  * lfr.py (Learning Fair Representations experiment)
  * Fair_Inference/fairinf.py (Fair Inference experiment)
  
 Each of these scripts will output the results to the console and write a matplotlib based graph to the plots/ directory.
 You will find that the plots/ directory already contains the graphs that were shown in the thesis.
 
 The fairinf script contains several functions written in R that are contained in the Fair_inference/ directory.
 The fairinf script runs for a lot longer than the other ones. On my laptop approximately 35 minutes to 45 minutes to run to completion.
 
 The directories are hardcoded in the experiment scripts, so you cannot run the scripts from any directory.
 
 This is how you run the scripts in this repo:
 
 example 1: (currently in Master_Thesis_Code directory)
  * python3 reweighing.py
  * python3 di.py
  * python3 lfr.py
  
 example 2: (currently in Fair_inference directory)
  * python3 fairinf.py
  
  If you want to run the experiments that Nabi and Shpitzer ran for fair inference, you can find them in the following repo:
  https://github.com/raziehna/fair-inference-on-outcomes.git
  
  I ran the adult sexism experiment using Rscript.

