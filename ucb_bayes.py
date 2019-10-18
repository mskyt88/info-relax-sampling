#### implementation of UCB-Bayes algorithm 
# reference: Kaufmann, Cappe and Garivier (2012)

import numpy as np
import scipy.stats

from common import *

class UCB_Bayes_bern(Alg_bern):
	def anytime(self):
		return True

	def action(self, t):
		T, K = self.T, self.K
		quantile = 1 - 1.0 / (t+1)
		qs = scipy.stats.beta.ppf( quantile, self.alphas, self.betas )
		return qs.argmax(), None

class UCB_Bayes_gauss(Alg_gauss):
	def anytime(self):
		return True
	
	def action(self, t):
		T, K = self.T, self.K
		quantile = 1 - 1.0 / (t+1)
		qs = scipy.stats.norm.ppf( quantile, self.means, self.sigmas )
		return qs.argmax(), None

