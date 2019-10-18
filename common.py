import numpy as np
import scipy.stats


class Alg:
	"""
	Abstract class for general MAB algorithms
	"""

	def __init__(self, _name, config):
		self.name = _name
		self.config = config

	def anytime(self):
		return False
		
	def prepare(self, T_max):
		pass

	def reset(self, T):
		pass

	def action(self, t):
		return 0

	def feedback(self, t, a, r):
		pass


class Alg_bern(Alg):
	"""
	Parent class for Bernoulli MAB algorithms
	"""

	def reset(self, T):
		self.T = T
		self.K = len(self.config.priors)
		self.alphas = np.array( self.config.priors[:,0] )
		self.betas = np.array( self.config.priors[:,1] )

	def feedback(self, t, a, r):
		self.alphas[a] += r
		self.betas[a] += 1-r

	def estimate_ps(self):
		return np.array( self.alphas, dtype=np.float ) / (self.alphas + self.betas)

	def sample_ps(self):
		return scipy.stats.beta.rvs( self.alphas, self.betas )

	def sample_outcome(self, D):
		ps = scipy.stats.beta.rvs( self.alphas, self.betas )
		rss = np.zeros( (self.K, D), dtype=np.int )
		for k in range(self.K):
			rss[k] = scipy.stats.bernoulli.rvs( ps[k], size=D )
		
		return ps, rss

	def future_beliefs(self, outcome):
		ps, rss = outcome
		D = rss.shape[1]
		yss = np.zeros( (self.K, D+1, 2), dtype=np.int )
		muss = np.zeros( (self.K, D), dtype=np.float )

		for k in range(self.K):
			yss[k][0] = self.alphas[k], self.betas[k]
			yss[k,1:,0] = self.alphas[k] + np.cumsum( rss[k] )
			yss[k,1:,1] = self.betas[k] + np.cumsum( 1-rss[k] )
			muss[k] = np.array( yss[k,:-1,0], dtype=np.float ) / (yss[k,:-1,0] + yss[k,:-1,1])

		return ps, rss, yss, muss


class Alg_gauss(Alg):
	"""
	Parent class for Gaussian MAB algorithms
	"""

	def reset(self, T):
		self.T = T
		self.K = len(self.config.priors)
		self.means = np.array( self.config.priors[:,0], dtype=np.float )
		self.sigmas = np.array( self.config.priors[:,1], dtype=np.float )
		self.precs = 1.0 / np.square( self.sigmas )
		self.noise_sigmas = np.array( self.config.noise_sigmas )
		self.noise_precs = 1.0 / np.square( self.noise_sigmas )

	def feedback(self, t, a, r):
		self.precs[a] += self.noise_precs[a]
		self.means[a] += (r - self.means[a]) * self.noise_precs[a] / self.precs[a]
		self.sigmas[a] = 1.0 / np.sqrt( self.precs[a] )

	def estimate_ms(self):
		return np.array( self.means )

	def sample_ms(self):
		return scipy.stats.norm.rvs( self.means, self.sigmas, size=self.K )

	def sample_outcome(self, D):
		ms = scipy.stats.norm.rvs( self.means, self.sigmas, size=self.K )
		rss = np.zeros( (self.K, D) )
		for k in range(self.K):
			rss[k] = scipy.stats.norm.rvs( ms[k], self.noise_sigmas[k], size=D )
		return ms, rss

	def future_beliefs(self, outcome):
		ms, rss = outcome
		D = rss.shape[1]
		yss = np.zeros( (self.K, D+1, 2) )
		muss = np.zeros( (self.K, D) )
		for k in range(self.K):
			precss = self.precs[k] + self.noise_precs[k] * np.arange(D+1)
			yss[k,0,0] = self.means[k]
			yss[k,1:,0] = ( self.means[k] * self.precs[k] + np.cumsum( rss[k] ) * self.noise_precs[k] ) / precss[1:]
			yss[k,:,1] = 1.0/np.sqrt(precss)
			muss[k] = yss[k,:-1,0]
		return ms, rss, yss, muss





class Alg_bern_single(Alg):
	def __init__(self, _name, config):
		Alg.__init__(self, _name, config )
		self.outside = self.config.outside
		self.alpha = self.config.prior[0]
		self.beta = self.config.prior[1]

	def reset(self, T):
		self.T = T
		self.alpha = self.config.prior[0]
		self.beta = self.config.prior[1]

	def feedback(self, t, a, r):
		if a == 1:
			self.alpha += r
			self.beta += 1-r

	def sample_ps(self):
		return np.array( [self.outside, scipy.stats.beta.rvs( self.alpha, self.beta ) ] )

	def estimate_ps(self):
		return np.array( [self.outside, float(self.alpha)/(self.alpha + self.beta) ] )
