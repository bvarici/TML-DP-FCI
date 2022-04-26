# TML-DP-FCI
ECSE 6962 - Trustworthy Machine Learning Spring'22 Project - Differentially Private FCI algorithm


Requirements:
Beyond basic libraries (e.g., numpy, scipy), you should have networkx installed.

functions.py: contains the main algorithms, FCI and PrivFCI.

run_simulations.py: runs FCI and PrivFCI with input dataset and arguments. Some parameters are not too important and it is safe to use default values.
important parameters are:

iter: number of times algorithms are run for taking average results.

q: subsampling ratio. Depends on the dataset size, but should not be too small, e.g., not less than 0.05

alpha: significance level for conditional indepence test. 0.1 is a common choice.

epslogmin: minimum epsilon value to run (on log10 base)

epslogmax: maximum epsilon value to run (on log10 base)

epsnum: the number of mid eps values for [epslogmin,epslogmax] range. Set at least 10 to observe the effect of various epsilon values.

Available datasets are: ['asia', 'cancer', 'earthquake', 'survey']

Sample command to run experiments:

python run_simulations.py --dataset cancer --iter 10 --epslogmin -1.0 --epslogmax 0.5 --epsnum 10


The results are saved to './results/estimated' directory.

Use plot_results.py to create plots for the chosen dataset.

