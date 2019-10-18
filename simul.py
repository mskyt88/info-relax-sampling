import numpy as np
import scipy.stats

from matplotlib import pyplot

import sys
import os
import datetime

from common import *


class Result:
	"""
	Class for a collection of simulation results of an algorithm.
	Each measure is stored in the form of dictionary whose key is time horizon and value is an array of the simulation outcomes.
	"""

	def __init__(self):
		self.Ts = []			# time horizons
		self.rewards = {}		# total rewards obtained by the algorithm
		self.bounds = {}		# the upper bounds computed by the algorithm (it could be None)
		self.eps = {}			# the length of time that the algorithm has used
		self.max_rewards = {}	# conventional bench (independent of algorithm)
		self.exp_rewards = {}	# denoised total rewards obtained by the algorithm

	def accumulate(self, T, actual_reward, exp_reward, max_reward, dual_bound, elapsed_time):
		if T not in self.Ts:
			self.Ts.append( T )
			self.Ts.sort()

			self.rewards[T] = []
			self.bounds[T] = []
			self.eps[T] = []
			self.max_rewards[T] = []
			self.exp_rewards[T] = []

		if actual_reward is not None:
			self.rewards[T].append( actual_reward )
		if exp_reward is not None:
			self.exp_rewards[T].append( exp_reward )
		if max_reward is not None:
			self.max_rewards[T].append( max_reward )
		if dual_bound is not None:
			self.bounds[T].append( dual_bound )
		if elapsed_time is not None:
			self.eps[T].append( elapsed_time )


	def truncate(self, n_samples):
		for T in self.Ts:
			self.rewards[T] = self.rewards[T][:n_samples]
			self.bounds[T] 		= self.bounds[T][:n_samples]
			self.eps[T] 		= self.eps[T][:n_samples]
			self.max_rewards[T] = self.max_rewards[T][:n_samples]
			self.exp_rewards[T] = self.exp_rewards[T][:n_samples]

	def merge(self, another):
		for T in another.Ts:
			if T not in self.Ts:
				self.Ts.append( T )
				self.Ts.sort()

				self.rewards[T] = []
				self.bounds[T] = []
				self.eps[T] = []
				self.max_rewards[T] = []
				self.exp_rewards[T] = []

			self.rewards[T].extend( another.rewards[T] )
			self.bounds[T].extend( another.bounds[T] )
			self.eps[T].extend( another.eps[T] )
			self.max_rewards[T].extend( another.max_rewards[T] )
			self.exp_rewards[T].extend( another.exp_rewards[T] )

	def clear_bounds(self):
		for T in self.Ts:
			self.bounds[T] = []

	def counts(self):
		return [ len(self.rewards[T]) for T in self.Ts ]

	def avg_rewards(self):
		return np.array( [ np.mean(self.rewards[T]) for T in self.Ts ] )

	def avg_regrets(self):
		return np.array( [ np.mean(self.max_rewards[T]) - np.mean(self.exp_rewards[T]) for T in self.Ts ] )

	def avg_max_rewards(self):
		return np.array( [ np.mean(self.max_rewards[T]) for T in self.Ts ] )

	def std_regrets(self):
		return np.array( [ np.std( np.array(self.max_rewards[T]) - np.array(self.exp_rewards[T]) ) for T in self.Ts ] )

	def stderr_regrets(self):
		return self.std_regrets() / np.sqrt( self.counts() )

	def avg_eps(self):
		return np.array( [ np.mean(self.eps[T]) for T in self.Ts ] )

	def avg_dual_bounds(self):
		if len(self.Ts) == 0 or len(self.bounds[self.Ts[0]]) == 0:
			return None
		return np.array( [ np.mean(self.bounds[T]) for T in self.Ts ] )

	def std_dual_bounds(self):
		if len(self.Ts) == 0 or len(self.bounds[self.Ts[0]]) == 0:
			return None
		return np.array( [ np.std(self.bounds[T]) for T in self.Ts ] )

	def stderr_dual_bounds(self):
		if len(self.Ts) == 0 or len(self.bounds[self.Ts[0]]) == 0:
			return None
		return np.array( [ np.std(self.bounds[T]) / np.sqrt(len(self.bounds[T])) for T in self.Ts ] )

	def avg_regret_bounds(self):
		if len(self.Ts) == 0 or len(self.bounds[self.Ts[0]]) == 0:
			return None
		return np.array( [ np.mean(self.max_rewards[T]) - np.mean(self.bounds[T]) for T in self.Ts ] )

	def avg_regret_ratios(self):
		return np.array( [ 1 - np.mean(self.exp_rewards[T])/np.mean(self.max_rewards[T])  for T in self.Ts ] )

	def save(self, fn):
		fd = open( fn, "w" )
		fd.write( "Ts=%s\n" % str(self.Ts) )
		for T in self.Ts:
			fd.write( "rewards[%d]=%s\n" % (T, self.rewards[T]) )
			fd.write( "bounds[%d]=%s\n" % (T, self.bounds[T]) )
			fd.write( "eps[%d]=%s\n" % (T, self.eps[T]) )
			fd.write( "max_rewards[%d]=%s\n" % (T, self.max_rewards[T]) )
			fd.write( "exp_rewards[%d]=%s\n" % (T, self.exp_rewards[T]) )
		fd.close()

	def load(self, fn):
		parse_t = lambda line: int( line[ line.find("[")+1: line.find("]") ] )
		parse_v = lambda line: eval( line[ line.find("=")+1 : ].strip() )

		fd = open( fn )
		for line in fd:
			if line.startswith("Ts="):
				self.Ts = eval( line[ line.find("=")+1: ].strip() )

			elif line.startswith("rewards"):
				T = parse_t( line )
				self.rewards[T] = parse_v( line )

			elif line.startswith("bounds"):
				T = parse_t( line )
				self.bounds[T] = parse_v( line )

			elif line.startswith("eps"):
				T = parse_t( line )
				self.eps[T] = parse_v( line )

			elif line.startswith("max_rewards"):
				T = parse_t( line )
				self.max_rewards[T] = parse_v( line )

			elif line.startswith("exp_rewards"):
				T = parse_t( line )
				self.exp_rewards[T] = parse_v( line )
		fd.close()



class Simulator:
	"""
	Class for simulating MAB algorithms specified in the configuration file
	- generate() generates the random reward realizations that will be shared across MAB algorithms
	- simulate() simulates MAB algorithms
	- dual_simulate() computes the dual bounds produced by IRS algorithms
	"""

	def __init__(self, config):
		assert( config.problem in ["bern", "gauss", "bern_single"] )

		self.config = config

		if config.problem == "bern":
			self.priors = np.array( config.priors )
			self.K = len(self.priors)

		elif config.problem in "gauss":
			self.priors = np.array( config.priors, dtype=np.float )
			self.K = len(self.priors)

		elif config.problem == "bern_single":
			self.outside = config.outside
			self.prior = config.prior
			self.K = 2

		self.Ts = config.Ts

	def generate(self, pid, sid, problem, T_max):
		seeds = [ (pid*10000 + sid)*self.K + k for k in range(self.K) ]
		np.random.seed( seed = seeds[0] )

		if problem == "bern":
			means = scipy.stats.beta.rvs( self.priors[:,0], self.priors[:,1], size=self.K )
			rss = np.zeros( (self.K, T_max), dtype=np.int )

			for k in range(self.K):
				np.random.seed( seed = seeds[k] )
				rss[k] = scipy.stats.bernoulli.rvs( means[k], size=T_max )

		elif problem == "gauss":
			means = scipy.stats.norm.rvs( self.priors[:,0], self.priors[:,1], size=self.K )
			rss = np.zeros( (self.K, T_max), dtype=np.float )

			for k in range(self.K):
				np.random.seed( seed = seeds[k] )
				rss[k] = scipy.stats.norm.rvs( means[k], self.config.noise_sigmas[k], size=T_max )

		return means, rss


	def simulate(self, algs, pid, callback, debug=False):
		T_max = max(self.Ts)
		PROBLEM = self.config.problem

		last_report_timestamp = datetime.datetime.now()

		S = config.S	# number of simulations to be performed

		# 1. prepare algorithms
		for alg in algs:
			alg.prepare( T_max )

		# 2. simulate MAB algorithms
		results = [ Result() for _ in algs ]

		pid = int(pid)
		for s in range(S):

			# 2-1. generate random reward realizations
			means, rss = self.generate( pid, s, PROBLEM, T_max )

			max_rewards = np.array( [ means.max() * T for T in self.Ts ] )

			# 2-2. simulate MAB algorithms
			for ai,alg in enumerate(algs):
				for ti,T in enumerate(self.Ts):
					if alg.anytime() == True:
						if ti < len(self.Ts)-1:
							continue
						else:
							alg.reset( None )
					else:
						alg.reset( T )

					act_reward, exp_reward, dual_bound = 0.0, 0.0, None
					cnts = [0] * self.K

					time_st = datetime.datetime.now()

					for t in range(T):
						a,v = alg.action(t)

						if t == 0 and v is not None:
							dual_bound = v

						r = rss[a][ cnts[a] ]
						alg.feedback( t, a, r )		# algorithm observes the realized outcome
						cnts[a] += 1

						act_reward += r				# total actual reward collected by algorithm
						exp_reward += means[a]		# total mean reward collected by algorithm

						if alg.anytime() == True and t+1 in self.Ts:
							_time_en, _T, _ti = datetime.datetime.now(), t+1, self.Ts.index(t+1)
							results[ ai ].accumulate( _T,   act_reward, exp_reward, max_rewards[_ti], 	dual_bound, (_time_en-time_st).total_seconds() )

					if alg.anytime() == False:
						time_en = datetime.datetime.now()
						results[ ai ].accumulate( T,   act_reward, exp_reward, max_rewards[ti], 	dual_bound, (time_en-time_st).total_seconds() )

			# 2-3. print out progress of simulations
			if s%10 == 0 or debug == True:
				print()
				#regretss = [ full_info - results[ai].avg_rewards() for ai in range(len(algs)) ]
				regretss = [ results[ai].avg_regrets() for ai in range(len(algs)) ]

				avgs = [ (regretss[ai].mean(),ai) for ai in range(len(algs)) ]
				avgs.sort()
				avgs.reverse()
				for _,ai in avgs:
					print( "%4d, %16s, %s" % (s, algs[ai].name, str(regretss[ai]).replace("\n", "") ) )
					
			# 2-4. archive the result (at least every five minutes)
			if (datetime.datetime.now() - last_report_timestamp).seconds > 5*60 or debug == True:
				callback( results )
				last_report_timestamp = datetime.datetime.now()

		return results

	def dual_simulate(self, algs, pid, callback, debug=False):
		T_max = max(self.Ts)
		PROBLEM = self.config.problem

		last_report_timestamp = datetime.datetime.now()

		S = config.S * 5
		support_duals = [True] * len(algs)

		# 1. prepare algorithms that can compute upper bounds
		for ai,alg in enumerate(algs):
			try:
				alg.bound
				alg.prepare( T_max )
			except:
				support_duals[ai] = False

		# 2. simulate
		results = [ Result() for _ in algs ]

		pid = int(pid)
		for s in range(S):
			means, rss = self.generate( pid, s, PROBLEM, T_max )

			for ai,alg in enumerate(algs):
				if support_duals[ai] == False:
					continue

				for T in self.Ts:
					alg.reset( T )
					dual_bound = alg.bound( alg.future_beliefs( (means, rss[:,:T]) ) )
					results[ ai ].accumulate( T,   None, None, None, 	dual_bound, None )
			
			if s%100 == 0 or debug == True:
				print( "DUAL:")
				duals = [ results[ai].avg_dual_bounds() for ai in range(len(algs)) ]

				avgs = [ (duals[ai].mean(),ai) for ai in range(len(algs)) if duals[ai] is not None ]
				avgs.sort()
				avgs.reverse()
				for _,ai in avgs:
					print( "%4d, %16s, %s" % (s, algs[ai].name, str(duals[ai]).replace("\n", "") ) )

			if (datetime.datetime.now() - last_report_timestamp).seconds > 5*60 or debug == True:
				callback( results )
				last_report_timestamp = datetime.datetime.now()

		return results


class Config:
	"""
	Class for parsing configuration file
	"""

	def __init__(self, fn):
		self.options = { "algorithms":[] }

		fd = open(fn)
		category = "DEFAULT"
		for line in fd:
			line = line.strip()
			if line.startswith("#"):
				continue
			elif line.endswith(":"):
				category = line[:-1].strip()
			elif line.startswith("algorithm"):
				self.options[ "algorithms" ].append( ( category, line.split("=")[1].strip() ) )
			elif line.strip() == "":
				continue
			else:
				line = line.split("=")
				self.options[ line[0].strip() ] = eval( line[1] )
		fd.close()

	def __getattr__(self, key):
		if key in self.options:
			return self.options[key]
		return None


def find_option(key, default=None):
	for a in sys.argv:
		if a.startswith(key+"="):
			return a[ len(key)+1 : ]
	return default




if __name__ == "__main__":

	# 1. load configuration file
	config_name = find_option("config")
	if config_name.endswith(".config"):
		config_name = config_name[ : -len(".config") ]

	config = Config( "configs/" + config_name + ".config" )
	res_dir = "result_" + config_name

	if res_dir not in os.listdir( "results" ):
		os.mkdir( "results/"+res_dir )

	# 2. setup algorithms
	algs = []
	for cat, cmd in config.algorithms:
		exec( "import " + cmd[ :cmd.find(".") ] )
		module = cmd[:-1] + ", config )"
		alg = eval( module )

		only = find_option( "only", None )
		if only is not None and alg.name not in only.split(","):
			continue

		category = find_option( "category", "default" )
		if "publish" not in sys.argv and category != "all" and cat.lower() != category.lower():
			continue

		print( module )
		algs.append( alg )

	if len(algs) == 0:
		raise Exception( "no algorithm is configured" )

	sim = Simulator( config )

	# 3. simulate
	pid = find_option( "pid", "0" )
	debug = ("debug" in sys.argv)
	dual_mode = ("dual" in sys.argv)

	if dual_mode:
		def archive( results ):
			for ai,alg in enumerate(algs):
				if results[ai].avg_dual_bounds() is not None:
					results[ai].save( "results/"+res_dir+"/DUAL_"+alg.name +"_"+pid+".result" )
			print( "archived" )
		
		results = sim.dual_simulate( algs, pid, archive, debug )

	else:
		def archive( results ):
			for ai,alg in enumerate(algs):
				results[ai].save( "results/"+res_dir+"/"+alg.name +"_"+pid+".result" )
			print( "archived" )

		results = sim.simulate( algs, pid, archive, debug )

	archive( results )
