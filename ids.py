#### implementation of Information-Directed Sampling
# reference: Russo and Van Roy (2014)
# reference2: https://github.com/gutin/FastGittins/blob/master/cpp/policy.hpp

import numpy as np
import scipy.stats

from common import *


def IDS_optimize( deltas, gs ):
	K = len(deltas)
	best_pair = (np.inf, None, None, None)
	#print( np.square(deltas[gs.argmax()]) / gs.max() )
	for i in range(K-1):
		for j in range(i+1,K):
			di,dj, gi,gj = deltas[i], deltas[j], gs[i], gs[j]
			v_func = lambda q: (q*di + (1-q)*dj)**2 / (q*gi + (1-q)*gj)

			if di == dj:
				candidates = []
			elif gi == gj:
				candidates = [dj/(-di+dj)]
			else:
				candidates = [dj/(-di+dj), (-dj*gi + 2*di*gj - dj*gj)/((-di+dj)*(gi-gj)) ]
			candidates = [ q for q in candidates if 0 <= q <= 1 ]
			
			if abs(gi) > 1e-16:
				candidates.append( 1 )
			if abs(gj) > 1e-16:
				candidates.append( 0 )
			if len( candidates ) == 0:
				continue
			candidates = np.array( candidates )

			vs = v_func( candidates )
			v = vs.min()
			q = candidates[ vs.argmin() ]

			if max(vs) > 1e100:
				print( di, dj, gi, gj, candidates, vs )
				input()

			if best_pair[0] > v:
				best_pair = ( v, i, j, q )

	return best_pair



def IDS_IR_bern(alphas, betas, xs, cdfss, pdfss, Qss ):
	K, H = len(alphas), len(xs)
	dx = 1.0/H #xs[1] - xs[0]
	ms = np.array( alphas, dtype=np.float ) / (alphas + betas)

	F = np.prod( cdfss, axis = 0 )
	
	prob_opts = np.zeros(K)
	for a in range(K):
		prods = np.prod( [ cdfss[a_else] for a_else in range(K) if a_else != a ], axis=0 )
		prob_opts[a] = np.dot( pdfss[a], prods ) * dx

	Maa = np.zeros( (K,K) )
	for i in range(K):
		for j in range(K):
			if i == j:
				Maa[i][j] = np.sum( xs * pdfss[i] * F / np.maximum(cdfss[i], 1e-7) ) * dx / prob_opts[i]
			else:
				Maa[i][j] = np.sum( pdfss[i] * F * Qss[j] / np.maximum(cdfss[i]*cdfss[j], 1e-7) ) * dx / prob_opts[i]

	rho_star = np.dot( prob_opts, np.diag(Maa) )
	deltas = rho_star - ms

	def KL(p1, p2):
		return p1 * np.log( p1/p2 ) + (1-p1) * np.log( (1-p1)/(1-p2) )

	gs = np.zeros(K)
	for i in range(K):
		for j in range(K):
			gs[i] += prob_opts[j] * KL(Maa[j][i], ms[i])

	return deltas, gs


class IDS_bern(Alg_bern):
	def anytime(self):
		return True

	def prepare(self, T_max):
		self.H = self.config.H_IDS
		self.xs = np.linspace( 0,1,self.H )

	def reset(self, T):
		Alg_bern.reset(self, T)
		self.cdfss = np.zeros( (self.K, self.H) )
		self.pdfss = np.zeros( (self.K, self.H) )
		self.Qss = np.zeros( (self.K, self.H) )
		
		for a in range(self.K):
			self.cdfss[a] = scipy.stats.beta.cdf( self.xs, self.alphas[a], self.betas[a] )
			self.pdfss[a] = scipy.stats.beta.pdf( self.xs, self.alphas[a], self.betas[a] )
			self.Qss[a] = float(self.alphas[a])/(self.alphas[a]+self.betas[a]) * scipy.stats.beta.cdf( self.xs, self.alphas[a]+1, self.betas[a] )

	def action(self, t):
		deltas, gs = IDS_IR_bern(self.alphas, self.betas, self.xs, self.cdfss, self.pdfss, self.Qss )
		best_pair = IDS_optimize( deltas, gs )

		b = scipy.stats.bernoulli.rvs( best_pair[3] )
		return best_pair[1] if b == 1 else best_pair[2], None

	def feedback(self, t, a, r):
		Alg_bern.feedback(self, t,a,r)
		self.cdfss[a] = scipy.stats.beta.cdf( self.xs, self.alphas[a], self.betas[a] )
		self.pdfss[a] = scipy.stats.beta.pdf( self.xs, self.alphas[a], self.betas[a] )
		self.Qss[a] = float(self.alphas[a])/(self.alphas[a]+self.betas[a]) * scipy.stats.beta.cdf( self.xs, self.alphas[a]+1, self.betas[a] )



RANGE = 40.0


def IDS_IR_gauss( mus, sigmas, xs, cdfss, pdfss ):
	K, H = len(mus), len(xs)
	dx = RANGE / H

	F = np.prod( cdfss, axis = 0 )
	
	prob_opts = np.zeros(K)
	for a in range(K):
		prods = np.prod( [ cdfss[a_else] for a_else in range(K) if a_else != a ], axis=0 )
		prob_opts[a] = np.dot( pdfss[a], prods ) * dx

	#print( prob_opts.sum(), prob_opts )
	#if abs(prob_opts.sum()-1.0) >= 1e-2:
	#	print( prob_opts.sum(), prob_opts )
	#	asdf
	
	Maa = np.zeros( (K,K) )
	for i in range(K):
		for j in range(K):
			if i == j:
				Maa[i][j] = np.sum( xs * pdfss[i] * F / np.maximum(cdfss[i], 1e-7) ) * dx / np.maximum( prob_opts[i], 1e-7 )
			else:
				ss = np.sum( pdfss[i] * F * pdfss[j] / np.maximum(cdfss[i]*cdfss[j], 1e-7) ) * dx / np.maximum( prob_opts[i], 1e-7 )
				Maa[i][j] = mus[j] - ss * sigmas[j]**2

	rho_star = np.dot( prob_opts, np.diag(Maa) )
	deltas = rho_star - mus

	gs = np.zeros(K)
	for i in range(K):
		for j in range(K):
			gs[i] += prob_opts[j] * (Maa[j][i] - mus[i])**2

	return deltas, gs


class IDS_gauss(Alg_gauss):
	def anytime(self):
		return True

	def prepare(self, T_max):
		self.H = self.config.H_IDS
		self.xs = np.linspace( -0.5*RANGE, 0.5*RANGE, self.H )

	def reset(self, T):
		Alg_gauss.reset(self, T)
		self.cdfss = np.zeros( (self.K, self.H) )
		self.pdfss = np.zeros( (self.K, self.H) )
		
		for a in range(self.K):
			self.cdfss[a] = scipy.stats.norm.cdf( self.xs, self.means[a], self.sigmas[a] )
			self.pdfss[a] = scipy.stats.norm.pdf( self.xs, self.means[a], self.sigmas[a] )

	def action(self, t):
		deltas, gs = IDS_IR_gauss(self.means, self.sigmas, self.xs, self.cdfss, self.pdfss )
		best_pair = IDS_optimize( deltas, gs )
		#print( t, deltas, gs, best_pair )
		
		b = scipy.stats.bernoulli.rvs( best_pair[3] )
		return best_pair[1] if b == 1 else best_pair[2], None

	def feedback(self, t, a, r):
		Alg_gauss.feedback(self, t,a,r)
		self.cdfss[a] = scipy.stats.norm.cdf( self.xs, self.means[a], self.sigmas[a] )
		self.pdfss[a] = scipy.stats.norm.pdf( self.xs, self.means[a], self.sigmas[a] )


