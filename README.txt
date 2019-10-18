* The codes are implemented and tested with Python 3.6.1.
	- 'numpy' and 'scipy' are required for simulation, and 'matplotlib' is required for plotting.

1. How to run simulations
(1) Basic description
	- In 'configs' directory, there are '*.config' files specifying the simulation settings.
		- 'bern_2arms.config' is for Bernoulli MAB with two arms.
		- 'bern_10arms.config' is for Bernoulli MAB with ten arms.
		- 'gauss_2arms.config' is for Gaussian MAB with two arms.
		- 'gauss_10arms.config' is for Gaussian MAB with ten arms.
		- 'gauss_5arms_asym.config' is for Gaussian MAB with five arms with heteroscedastic noise.

	- Run "python simul.py config=bern_2arms" to simulate and measure the performance of MAB algorithms. The results will be saved in 'results/result_bern_2arms' directory for each algorithm.

	- Run "python simul.py config=bern_2arms dual" to compute the dual bounds produced by IRS algorithms. The results will be saved in 'results/result_bern_2arms' directory for each algorithm.

	- Run "python publish_plots.py bern_2arms" to create the plots based on the files in 'results/result_bern_2arms' directory. The results will be saved in 'results/figures' and 'results/tables' directories.

(2) Running simulations in parallel
	- One can specify an optional argument 'pid' to run simulations with a different random number seed.
	- For example, "python simul.py config=bern_2arms pid=1" and "python simul.py config=bern_2arms pid=2" will run independently, and the results will be saved into different files.
	- "python publish_plots.py bern_2arms" will collect all results automatically.

(3) Description of configuration file
	- One can run simulations with different options by modifying '*.config' file.
	- A configuration file basically defines variables that will be used in 'simul.py'. Each value will be evaulated via python 'eval()' function, so any python expression can be used.
	- In common,
		- the variable 'problem' specifies the type of MAB problem: 'bern' or 'gauss' is available now.
		- the variable 'S' specifies the number of simulations to be conducted.
		- the variable 'Ts' specifies an array of time horizons.
		- the variable 'priors' specifies an array of prior distributions whose length equals to the number of arms.
		- the variable(s) 'algorithm' specifies the algorithms to be simulated.

2. Source code description
(1) Common modules
	- 'common.py' implements the skeleton codes of MAB algorithm. 
	- 'simul.py' implements the codes for simulations.
	- 'publish_plots.py' implements the codes for creating plots.
(2) Algorithms
	- 'irs_fh.py', 'irs_vzero.py', 'irs_vemax.py', and 'irs_index.py' implement IRS.FH, IRS.V-ZERO, IRS.V-EMAX, and IRS.INDEX algorithms.
	- 'ts.py' implements Thompson sampling algorithm.
	- 'ucb_bayes.py' implements UCB-Bayes algorithm.
	- 'ids.py' implements Information-directed sampling algorithm.
	- 'ogi.py' implements Optimistic Gittins-index algorithms.

