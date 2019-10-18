#### implementation of Optimistic Gittins Index algorithm 
# reference: Gutin and Farias (2016)

from common import *

import scipy.stats
import scipy.special

def ogi_value_bern(a, b, gamma, old_val = None):
	g = lambda lam: float(a)/(a+b) * ( 1 - gamma * scipy.stats.beta.cdf(lam, a+1, b) ) + gamma * lam * scipy.stats.beta.cdf(lam,a,b) - lam
	dg = lambda lam: gamma * scipy.stats.beta.cdf(lam, a, b) - 1

	old_v = float(a)/(a+b) if old_val is None else old_val

	for i in range(128):
		old_v = max( 1e-6, min(old_v, 1-1e-6) )
		g_at_v = g(old_v)
		new_v = old_v - g_at_v / dg(old_v)
		if abs(g_at_v) < 1e-4:
			break
		old_v = new_v
	else:
		print( a, b, gamma, old_val )
		raise Exception("Newton didn't converge in OGI algorithm")
		
	return new_v


def ogi_value_gauss(m, nu, gamma, old_val = None):
	g = lambda lam: m + gamma * ( (lam-m)*scipy.stats.norm.cdf((lam-m)*nu) + 1.0/nu*scipy.stats.norm.pdf((lam-m)*nu) ) - lam
	dg = lambda lam: gamma * scipy.stats.norm.cdf((lam-m)*nu) - 1

	old_v = m if old_val is None else old_val

	for i in range(128):
		g_at_v = g(old_v)
		new_v = old_v - g_at_v / dg(old_v)
		if abs(g_at_v) < 1e-3:
			break
		old_v = new_v
	else:
		print( m, nu, gamma, old_val )
		raise Exception("Newton didn't converge in OGI algorithm")

	return new_v



class OGI_bern(Alg_bern):
	def anytime(self):
		return (self.name.endswith("-T") == False)

	def reset(self, T):
		Alg_bern.reset(self, T)
		self.T_known = True if self.name.endswith("-T") else False
		self.indices = np.array( self.config.priors[:,0], dtype=np.float ) / (self.config.priors[:,0] + self.config.priors[:,1])

	def action(self, t):
		gamma = 1 - 1.0/(self.T-t) if self.T_known else 1 - 1.0/(t+1)

		for k in range(self.K):
			self.indices[k] = ogi_value_bern( self.alphas[k], self.betas[k], gamma, self.indices[k] ) 
		return self.indices.argmax(), None


class OGI_gauss(Alg_gauss):
	def anytime(self):
		return (self.name.endswith("-T") == False)
		
	def reset(self, T):
		Alg_gauss.reset(self, T)
		self.T_known = True if self.name.endswith("-T") else False
		self.indices = np.zeros( self.K )

	def action(self, t):
		gamma = 1 - 1.0/(self.T-t) if self.T_known else 1 - 1.0/(t+1)

		for k in range(self.K):
			self.indices[k] = ogi_value_gauss( self.means[k], 1.0/self.sigmas[k], gamma, self.indices[k] ) 
		return self.indices.argmax(), None


class OGI_bern_single(Alg_bern_single):
	def reset(self, T):
		Alg_bern_single.reset(self, T)
		self.T_known = True if self.name.endswith("-T") else False
		self.index = float(self.alpha) / (self.alpha+self.beta)

	def action(self, t):
		gamma = 1 - 1.0/(self.T-t) if self.T_known else 1 - 1.0/(t+1)
		self.index = ogi_value_bern( self.alpha, self.beta, gamma, self.index ) 

		if self.index >= self.outside:
			return 1, None
		return 0, None
