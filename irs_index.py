#### implementation of IRS.INDEX algorithm

from common import *

import random
import scipy.stats


def worth_trying_1( T, outside, means, mean_cumsum, emaxs ):		# standard version
	mean_cumsum = mean_cumsum[:T]
	emax_cumsum = np.cumsum( emaxs )[:T]
	emax_cummin = np.minimum.accumulate( emaxs )[1:]
	ds = T - np.arange(1,T+1)

	vals = np.zeros( T+1 )
	vals[0] = outside*T
	vals[1:] = ds * (outside - emax_cummin) + mean_cumsum + T*emaxs[0] - emax_cumsum

	opt_n = np.argmax(vals)
	if opt_n == T:
		return True
	elif opt_n == 0 or np.argmin(emaxs[:opt_n+1]) == 0:
		return False
	return True

def worth_trying_2( T, outside, means, mean_cumsum, emaxs ):		# variation
	shifted = emaxs[1:]
	return np.cumsum( means[:T] - outside - (shifted - emaxs[0]) ).max() > 0


def worth_trying_3( T, outside, means, mean_cumsum, emaxs ):
	mean_cumsum = mean_cumsum[:T]
	emax_cumsum = np.cumsum( emaxs )[:T]
	emax_cummin = np.minimum.accumulate( emaxs )[1:]
	ds = T - np.arange(1,T+1)

	vals = np.zeros( T+1 )
	vals[0] = outside*T
	vals[1:] = ds * (outside - emax_cummin) + mean_cumsum + T*emaxs[0] - emax_cumsum

	return (np.argmax(vals) > 0)


def worth_trying_4( T, outside, means, mean_cumsum, emaxs ):
	ss = np.arange( T ) + 1
	shifted = emaxs[1:]
	return np.cumsum( means[:T] - outside - (T-ss) * (shifted - emaxs[:T]) ).max() > 0



def index_bern_emax( T, alphas, betas, old_val = None, scheme="1" ):
	mus = np.array( alphas, dtype=np.float ) / ( alphas + betas )
	mean_cumsum = np.cumsum( mus )

	if scheme == "1":
		worth_trying = lambda outside: worth_trying_1( T, outside, mus, mean_cumsum,
			outside * scipy.stats.beta.cdf(outside,alphas,betas) + mus * (1-scipy.stats.beta.cdf(outside,alphas+1,betas)) )
	elif scheme == "2":
		worth_trying = lambda outside: worth_trying_2( T, outside, mus, mean_cumsum,
			outside * scipy.stats.beta.cdf(outside,alphas,betas) + mus * (1-scipy.stats.beta.cdf(outside,alphas+1,betas)) )
	elif scheme == "3":
		worth_trying = lambda outside: worth_trying_3( T, outside, mus, mean_cumsum,
			outside * scipy.stats.beta.cdf(outside,alphas,betas) + mus * (1-scipy.stats.beta.cdf(outside,alphas+1,betas)) )
	elif scheme == "4":
		worth_trying = lambda outside: worth_trying_4( T, outside, mus, mean_cumsum,
			outside * scipy.stats.beta.cdf(outside,alphas,betas) + mus * (1-scipy.stats.beta.cdf(outside,alphas+1,betas)) )
	else:
		raise Exception( "Scheme must be specified for IRS.INDEX algorithm" )

	lower, upper = 0.0, 1.0
	
	conf_itv = 1.0/np.sqrt(alphas[0]+betas[0])

	if worth_trying( old_val - conf_itv ) == True:
		lower = old_val - conf_itv
	
	if worth_trying( old_val + conf_itv ) == False:
		upper = old_val + conf_itv

	while upper > lower + 1e-4:
		mid = (upper + lower)/2
		if worth_trying( mid ):
			lower = mid
		else:
			upper = mid

	return (upper + lower)/2


class IRS_INDEX_bern(Alg_bern):
	def reset(self, T):
		Alg_bern.reset( self, T )
		self.index = self.estimate_ps()
		self.scheme = self.name[-1]

	def action(self, t):
		K, D = self.K, self.T-t

		if D == 1:
			return self.estimate_ps().argmax(), None

		ps, rss, yss, muss = self.future_beliefs( self.sample_outcome(D) )

		for k in range(K):
			self.index[k] = index_bern_emax( D, yss[k,:,0], yss[k,:,1], self.index[k], self.scheme )

		return self.index.argmax(), None



def index_gauss_emax( T, means, sigmas, old_val = None, scheme = "1" ):
	squared_sigmas = np.square( sigmas )
	mean_cumsum = np.cumsum( means )

	if scheme == "1":
		worth_trying = lambda outside: worth_trying_1( T, outside, means, mean_cumsum,
			means + (outside - means) * scipy.stats.norm.cdf(outside,means,sigmas) + squared_sigmas * scipy.stats.norm.pdf(outside,means,sigmas) )
	elif scheme == "2":
		worth_trying = lambda outside: worth_trying_2( T, outside, means, mean_cumsum,
			means + (outside - means) * scipy.stats.norm.cdf(outside,means,sigmas) + squared_sigmas * scipy.stats.norm.pdf(outside,means,sigmas) )
	elif scheme == "3":
		worth_trying = lambda outside: worth_trying_3( T, outside, means, mean_cumsum,
			means + (outside - means) * scipy.stats.norm.cdf(outside,means,sigmas) + squared_sigmas * scipy.stats.norm.pdf(outside,means,sigmas) )
	elif scheme == "4":
		worth_trying = lambda outside: worth_trying_4( T, outside, means, mean_cumsum,
			means + (outside - means) * scipy.stats.norm.cdf(outside,means,sigmas) + squared_sigmas * scipy.stats.norm.pdf(outside,means,sigmas) )
	else:
		raise Exception( "Scheme must be specified for IRS.INDEX algorithm" )

	conf_itv = sigmas[0]

	lower, upper = -40, 40

	if worth_trying( old_val - conf_itv ) == True:
		lower = old_val - conf_itv
	
	if worth_trying( old_val + conf_itv ) == False:
		upper = old_val + conf_itv

	while upper > lower + 1e-4:
		mid = (upper + lower)/2

		if worth_trying( mid ):
			lower = mid
		else:
			upper = mid

	return (upper + lower)/2


class IRS_INDEX_gauss(Alg_gauss):
	def reset(self, T):
		Alg_gauss.reset( self, T )
		self.index = self.estimate_ms()
		self.scheme = self.name[-1]

	def action(self, t):
		K, D = self.K, self.T-t

		if D == 1:
			return self.estimate_ms().argmax(), None

		ps, rss, yss, muss = self.future_beliefs( self.sample_outcome(D) )

		for k in range(K):
			self.index[k] = index_gauss_emax( D, yss[k,:,0], yss[k,:,1], self.index[k], self.scheme )

		return self.index.argmax(), None


class IRS_INDEX_AVG_gauss(Alg_gauss):
	def reset(self, T):
		Alg_gauss.reset( self, T )
		self.index = self.estimate_ms()
		self.scheme = self.name[-1]

	def action(self, t):
		K, D = self.K, self.T-t

		if D == 1:
			return self.estimate_ms().argmax(), None

		nsamples = 5
		sum_index = np.zeros(K)
		for _ in range(nsamples):
			ps, rss, yss, muss = self.future_beliefs( self.sample_outcome(D) )
			for k in range(K):
				sum_index[k] += index_gauss_emax( D, yss[k,:,0], yss[k,:,1], self.index[k], "1" )

		self.index = sum_index / nsamples
		return self.index.argmax(), None