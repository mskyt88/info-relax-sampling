#### implementation of Bayesian optimal algorithm for Bernoulli MAB

from common import *

import random

class OPT_bern(Alg_bern):
	def prepare(self, T_max):
		assert( len(self.config.priors) == 2 )
		a1_0, b1_0 = self.config.priors[0]
		a2_0, b2_0 = self.config.priors[1]

		self.Vss = np.zeros( (T_max+1, T_max+a1_0+2, T_max+b1_0+2, T_max+a2_0+2, T_max+b2_0+2) )
		self.path = np.zeros( (T_max+1, T_max+a1_0+2, T_max+b1_0+2, T_max+a2_0+2, T_max+b2_0+2), dtype=np.int )

		print( "preparing OPT ..." )
		for d in range(1,T_max+1):
			for a1 in range(1,T_max-d+a1_0+2):
				for b1 in range(1,T_max-d+b1_0+2):
					for a2 in range(1,T_max-d+a2_0+2):
						for b2 in range(1,T_max-d+b2_0+2):

							v1 = float(a1)/(a1+b1) * (1 + self.Vss[d-1][a1+1][b1][a2][b2]) + float(b1)/(a1+b1) * self.Vss[d-1][a1][b1+1][a2][b2]
							v2 = float(a2)/(a2+b2) * (1 + self.Vss[d-1][a1][b1][a2+1][b2]) + float(b2)/(a2+b2) * self.Vss[d-1][a1][b1][a2][b2+1]

							self.Vss[d][a1][b1][a2][b2] = max(v1,v2)
							self.path[d][a1][b1][a2][b2] = 0 if v1 >= v2 else 1
		print( "done." )

	def action(self, t):
		D = self.T-t
		a1, b1 = self.alphas[0], self.betas[0]
		a2, b2 = self.alphas[1], self.betas[1]
		return self.path[D][a1][b1][a2][b2], self.Vss[D][a1][b1][a2][b2]


class OPT_bern_single(Alg_bern_single):
	def prepare(self, T_max):
		a0, b0 = self.alpha, self.beta
		self.Vss = np.zeros( (T_max+1, T_max+a0+2, T_max+b0+2) )
		self.path = np.zeros( (T_max+1, T_max+a0+2, T_max+b0+2), dtype=np.int )

		print( "preparing OPT ..")
		for d in range(1, T_max+1):
			for a in range(a0, a0+T_max-d+1):
				for b in range(b0, b0+T_max-d+1):
					v0 = self.outside + self.Vss[d-1][a][b]
					v1 = float(a)/(a+b) * (1+self.Vss[d-1][a+1][b]) + float(b)/(a+b) * self.Vss[d-1][a][b+1]

					self.Vss[d][a][b] = max(v0,v1)
					self.path[d][a][b] = 1 if v1 >= v0 else 0

		print( "done." )

	def action(self, t):
		D = self.T-t
		return self.path[ D ][ self.alpha ][ self.beta ], self.Vss[ D ][ self.alpha ][ self.beta ]


