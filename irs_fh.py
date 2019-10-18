#### implementation of IRS.FH algorithm


from common import *

import random

class IRS_FH_bern(Alg_bern):
	def action(self, t):
		K, D = self.K, self.T-t
		sampled_ps = self.sample_ps()
		best_estimates = np.zeros( K )
		for k in range(K):
			rsum = scipy.stats.binom.rvs( D-1, sampled_ps[k] )
			best_estimates[k] = float(self.alphas[k] + rsum) / (self.alphas[k] + self.betas[k] + D-1)
		return best_estimates.argmax(), best_estimates.max() * D

	def bound(self, future):
		ps, rss, yss, muss = future
		return muss[:, self.T-1 ].max() * self.T


class IRS_FH_gauss(Alg_gauss):
	def action(self, t):
		K, D = self.K, self.T-t
		sampled_ms = self.sample_ms()
		best_estimates = np.zeros( K )
		for k in range(K):
			rsum = scipy.stats.norm.rvs( (D-1) * sampled_ms[k], np.sqrt(D-1) * self.noise_sigmas[k] )
			best_estimates[k] = ( self.precs[k] * self.means[k] + self.noise_precs[k] * rsum) / ( self.precs[k] + (D-1) * self.noise_precs[k] )
		return best_estimates.argmax(), best_estimates.max() * D

	def bound(self, future):
		ms, rss, yss, muss = future
		return muss[:, self.T-1 ].max() * self.T

class IRS_FH_bern_single(Alg_bern_single):
	def action(self, t):
		K, D = self.K, self.T-t
		sampled_ps = self.sample_ps()
		best_estimates = np.zeros(2)
		best_estimates[0] = self.outside

		rsum1 = scipy.stats.binom.rvs( D-1, sampled_ps[1] )
		best_estimates[1] = float(self.alpha + rsum1) / (self.alpha + self.beta + D-1)

		return best_estimates.argmax(), best_estimates.max() * D

