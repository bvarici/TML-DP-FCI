Requirements:
Beyond basic libraries (e.g., numpy, scipy), you should have networkx installed on python.

functions.py: contains the main algorithms, FCI and PrivFCI.
run_simulations.py: runs FCI and PrivFCI with input dataset and arguments as follows:

parser.add_argument('--dataset',default='cancer')
parser.add_argument('--iter', type=int, default=20)
parser.add_argument('--delta', type=float, default=1e-3)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--q', type=float, default=0.1)
parser.add_argument('--bias', type=float, default=0.02)
parser.add_argument('--epslogmin', type=float, default=0)
parser.add_argument('--epslogmax', type=float, default=0)
parser.add_argument('--epsnum', type=int, default=1)

Some parameters are not too important and it is safe to use default values.
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

