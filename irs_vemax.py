#### implementation of IRS.V-EMAX algorithm

from common import *



def IRS_coupled_inner_TWO( D, rss ):		# rss[ 0 or 1 ][ s0 ][ s1 ]
	Vss = np.zeros( (D+1,D+1), dtype=np.float )
	path = np.zeros( (D+1,D+1), dtype=np.int )

	# calc value funcs
	Vss[0,1:] = np.cumsum( rss[1][0,:-1] )
	Vss[1:,0] = np.cumsum( rss[0][:-1,0] )
	path[0][1:] = 1

	for s0 in range(1,D+1):
		Vss[s0] = Vss[s0-1] + rss[0][s0-1]
		for s1 in range(1,D+1-s0):
			v1 = Vss[s0][s1-1] + rss[1][s0][s1-1]
			if Vss[s0][s1] < v1:
				Vss[s0][s1] = v1
				path[s0][s1] = 1

	# retrieve optimal soln
	cur_s0, cur_s1 = D, 0
	for s0 in range(D+1):
		s1 = D-s0
		if Vss[ cur_s0 ][ cur_s1 ] < Vss[s0][s1]:
			cur_s0, cur_s1 = s0, s1
	opt_v = Vss[ cur_s0 ][ cur_s1 ]

	opt_sol = []
	for d in range(D):
		opt_sol.append( path[cur_s0][cur_s1] )
		if opt_sol[-1] == 0:
			cur_s0 -= 1
		else:
			cur_s1 -= 1
	opt_sol.reverse()

	return opt_v, opt_sol



class IRS_VEMAX_bern(Alg_bern):
	def prepare(self, T_max):
		self.H = self.config.H_IRS
		xs = np.linspace( 0,1,self.H )
		max_a = self.config.priors.max()

		self.cdf_dic = np.zeros( (T_max+max_a+1, T_max+max_a+1, self.H) )
		for a in range(1,T_max+max_a+1):
			for b in range(1,T_max+max_a+1):
				self.cdf_dic[a][b] = scipy.stats.beta.cdf( xs, a, b )

	def action(self, t):
		K, D = self.K, self.T-t
		assert( K == 2 )

		ps, rss, yss, muss = self.future_beliefs( self.sample_outcome(D) )

		emaxs = np.zeros( (D+1,D+1) )
		for s0 in range(D):
			a0,b0 = yss[0][s0]
			for s1 in range(D-s0):
				a1,b1 = yss[1][s1]
				emaxs[s0][s1] = np.sum( 1.0 - self.cdf_dic[a0][b0] * self.cdf_dic[a1][b1] ) / self.H

		rss = np.zeros( (2,D+1,D+1) )
		for s0 in range(D):
			for s1 in range(D-s0):
				d = D-s0-s1
				rss[0][s0][s1] = muss[0][s0] + ( (emaxs[0][0] - emaxs[s0+1][s1]) if d > 1 else 0 )
				rss[1][s0][s1] = muss[1][s1] + ( (emaxs[0][0] - emaxs[s0][s1+1]) if d > 1 else 0 )

		v, sol = IRS_coupled_inner_TWO( D, rss )

		return sol[0], v

	def bound(self, future):
		K, D = self.K, self.T
		assert( K == 2 )

		ps, rss, yss, muss = future

		emaxs = np.zeros( (D+1,D+1) )
		for s0 in range(D):
			a0,b0 = yss[0][s0]
			for s1 in range(D-s0):
				a1,b1 = yss[1][s1]
				emaxs[s0][s1] = np.sum( 1.0 - self.cdf_dic[a0][b0] * self.cdf_dic[a1][b1] ) / self.H

		rss = np.zeros( (2,D+1,D+1) )
		for s0 in range(D):
			for s1 in range(D-s0):
				d = D-s0-s1
				rss[0][s0][s1] = muss[0][s0] + ( (emaxs[0][0] - emaxs[s0+1][s1]) if d > 1 else 0 )
				rss[1][s0][s1] = muss[1][s1] + ( (emaxs[0][0] - emaxs[s0][s1+1]) if d > 1 else 0 )

		v, sol = IRS_coupled_inner_TWO( D, rss )

		return v



def eabs_gauss( m, s ):
	etrunc = m * scipy.stats.norm.cdf(m/s) + s * scipy.stats.norm.pdf(m/s)
	return 2 * etrunc - m

def emax2_gauss( m0, s0, m1s, s1s ):
	return 0.5 * ( m0 + m1s + eabs_gauss( m0 - m1s, np.sqrt( s0**2 + np.square(s1s) ) ) )


class IRS_VEMAX_gauss(Alg_gauss):
	def action(self, t):
		K, D = self.K, self.T-t
		assert( K == 2 )

		ms, rss, yss, muss = self.future_beliefs( self.sample_outcome(D) )

		emaxs = np.zeros( (D+1,D+1) )
		sr0, sr1 = self.noise_sigmas[0], self.noise_sigmas[1]

		for s0 in range(D):
			m0, std0 = yss[0][s0]
			m1s, std1s = yss[1,:D-s0,0], yss[1,:D-s0,1]
			emaxs[s0][:D-s0] = emax2_gauss( m0, std0,  m1s, std1s )

		rss = np.zeros( (2,D+1,D+1) )
		for s0 in range(D):
			for s1 in range(D-s0):
				d = D-s0-s1

				rss[0][s0][s1] = muss[0][s0] + ( (emaxs[0][0] - emaxs[s0+1][s1]) if d > 1 else 0 )
				rss[1][s0][s1] = muss[1][s1] + ( (emaxs[0][0] - emaxs[s0][s1+1]) if d > 1 else 0 )

		v, sol = IRS_coupled_inner_TWO( D, rss )

		return sol[0], v


	def bound(self, future):
		K, D = self.K, self.T
		assert( K == 2 )

		ms, rss, yss, muss = future

		emaxs = np.zeros( (D+1,D+1) )
		sr0, sr1 = self.noise_sigmas[0], self.noise_sigmas[1]

		for s0 in range(D):
			m0, std0 = yss[0][s0]
			m1s, std1s = yss[1,:D-s0,0], yss[1,:D-s0,1]
			emaxs[s0][:D-s0] = emax2_gauss( m0, std0,  m1s, std1s )

		rss = np.zeros( (2,D+1,D+1) )
		for s0 in range(D):
			for s1 in range(D-s0):
				d = D-s0-s1

				rss[0][s0][s1] = muss[0][s0] + ( (emaxs[0][0] - emaxs[s0+1][s1]) if d > 1 else 0 )
				rss[1][s0][s1] = muss[1][s1] + ( (emaxs[0][0] - emaxs[s0][s1+1]) if d > 1 else 0 )

		v, sol = IRS_coupled_inner_TWO( D, rss )

		return v






