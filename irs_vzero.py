#### implementation of IRS.V-ZERO algorithm

from common import *

import random

def IRS_separable_inner( K, T, muss ):
	Sss = np.zeros( (K,T+1) )
	for k in range(K):
		Sss[k] = np.append( 0, np.cumsum(muss[k]) )

	Vss = np.zeros( (K+1, T+1) )		# V[k][t] := maximal value when allocating t opportunities to arms 0 ... k
	path = np.zeros( (K+1, T+1), dtype=np.int )

	Vss[-1,1:] = -np.inf

	for t in range(1,T+1):
		for k in range(K):
			vs = Vss[k-1][ t : : -1 ] + Sss[k][ : t+1 ]
			Vss[k][t] = vs.max()
			path[k][t] = vs.argmax()
	
	# calculation of solution
	opt_sol = np.zeros( K, dtype=np.int )
	rem = T
	for k in range(K-1,-1,-1):
		opt_sol[k] = path[k][rem]
		rem -= opt_sol[k]

	return Vss[K-1][T], opt_sol



class IRS_VZERO_bern(Alg_bern):
	def action(self, t):
		K, D = self.K, self.T-t
		ps, rss, yss, muss = self.future_beliefs( self.sample_outcome(D) )
		v, sol = IRS_separable_inner( K, D, muss )
		return sol.argmax(), v

	def bound(self, future):
		ps, rss, yss, muss = future
		v, sol = IRS_separable_inner( self.K, self.T, muss )
		return v


class IRS_VZERO_gauss(Alg_gauss):
	def action(self, t):
		K, D = self.K, self.T-t
		ms, rss, yss, muss = self.future_beliefs( self.sample_outcome(D) )
		v, sol = IRS_separable_inner( K, D, muss )
		return sol.argmax(), v

	def bound(self, future):
		ms, rss, yss, muss = future
		v, sol = IRS_separable_inner( self.K, self.T, muss )
		return v


