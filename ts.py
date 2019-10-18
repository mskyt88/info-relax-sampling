#### implementation of Thompson sampling algorithm

import scipy.stats

from common import *


class TS_bern(Alg_bern):
	def action(self, t):
		sampled_ps = self.sample_ps()
		return sampled_ps.argmax(), sampled_ps.max() * (self.T-t)

	def bound(self, future):
		ps, rss, yss, muss = future
		return ps.max() * self.T

class TS_gauss(Alg_gauss):
	def action(self, t):
		sampled_ms = self.sample_ms()
		return sampled_ms.argmax(), sampled_ms.max() * (self.T-t)

	def bound(self, future):
		ms, rss, yss, muss = future
		return ms.max() * self.T

class TS_bern_single(Alg_bern_single):
	def action(self, t):
		sampled_ps = self.sample_ps()
		return sampled_ps.argmax(), sampled_ps.max() * (self.T-t)
		
