import numpy as np
import scipy.stats


from matplotlib import pyplot
from matplotlib.font_manager import FontProperties

import sys
import os
import datetime

from simul import *
from common import *


def change_algname( algname ):
	return algname.replace( "UCB.Bayes", "Bayes-UCB" ).replace( "IRS.INDEX1", "IRS.INDEX" ).replace( "IRS.INDEX2", "IRS.INDEX*")

def format_algname( algname ):		# for latex format
	return algname.replace("IRS", "Irs").replace( "V-ZERO", "V-Zero" ).replace( "V-EMAX", "V-EMax" ).replace( "INDEX", "Index" )

def format_timespan(secs):
	if secs < 1e-3:
		return "0 ms"
	elif secs < 1:
		return "%.f ms" % (secs*1000)
	elif secs < 60:
		return "%.1f sec" % secs
	elif secs < 60*60:
		return "%.1f min" % (secs/60)
	else:
		return "%.1f hr" % (secs/3600)

def adjust_label_positions( ys, min_gap ):
	index = np.zeros( len(ys), dtype=np.int )
	index[ np.argsort( ys ) ] = np.arange( len(ys) )

	ys = np.sort( np.array( ys, dtype=np.float ) )
	adj = min_gap / 10.0

	flag = True
	while flag:
		flag = False
		for i in range( len(ys)-1 ):
			if ys[i] > ys[i+1]-min_gap:
				ys[i] -= adj
				flag = True
		for i in range( 1, len(ys) ):
			if ys[i] < ys[i-1]+min_gap:
				ys[i] += adj
				flag = True

	return ys[ index ]




def publish_regret_plot( config_name, filename, title, figsize, options = {} ):
	print( "="*20, filename, "="*20 )

	config = Config( "configs/"+config_name + ".config" )
	res_dir = "results/result_" + config_name

	def get_option( key, default=None ):
		if key in options:
			return options[ key ]
		return default

	# 1. setup algorithms
	algs = []
	for cat, cmd in config.algorithms:
		exec( "import " + cmd[ :cmd.find(".") ] )
		module = cmd[:-1] + ", config )"
		alg = eval( module )

		an = change_algname( alg.name )
		if get_option( "only", None ) is None or an in get_option( "only", [] ):
			if an not in get_option( "except", [] ):
				algs.append( alg )

	# 2. parse simulation results
	results = []
	for alg in algs:
		result = Result()
		
		print( alg.name, end=" : " )
		for fn in os.listdir( res_dir ):
			if fn.endswith(".result") == False:
				continue
			parsed = fn.split("_")

			if parsed[0] == alg.name:
				print( fn, end=", " )
				r = Result()
				r.load( res_dir + "/" + fn )
				r.clear_bounds()
				result.merge( r )
				#break

			elif parsed[0] == "DUAL" and parsed[1] == alg.name: # and parsed[2] == "1.result":
				print( fn, end=", ")
				r = Result()
				r.load( res_dir + "/" + fn )
				r.truncate( 2000 )
				result.merge( r )

				if alg.name == "TS":
					emax = result.avg_dual_bounds()[0] / r.Ts[0]
		print()

		results.append( result )

	# 3. sort algorithms in order of elapsed time
	elapsed_times = []

	Ss = []
	for ai in range(len(algs)):
		if len(results[ai].Ts) > 0:
			Ss.extend( results[ai].counts() )
			elapsed_times.append( ( results[ai].avg_eps()[-1], ai ) )

	elapsed_times.sort()


	# 4. create a regret plot

	# pre-defined color table
	COLORS = { "TS":"#1f77b4", "IRS.FH":"#ff7f0e", "IRS.V-ZERO":"#2ca02c", "IRS.V-EMAX":"#8c564b", "IRS.INDEX":"#9467bd", "OPT":"#d62728" }
	CMAP = [ "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#131313" ]

	def get_color( algname ):
		if algname in COLORS:
			return COLORS[ algname ]
		return CMAP.pop( 0 )

	pyplot.figure( figsize=figsize )

	side_labels = []

	lowest_avg_regrets = [np.inf]
	highest_dual_bounds = [-np.inf]

	for time, ai in elapsed_times:
		if len(results[ai].Ts) > 0:
			alg = algs[ai]
			Ts = np.append( [0], results[ai].Ts )
			avg_regrets = np.append( [0], results[ai].avg_regrets() )
			dual_bounds = results[ai].avg_regret_bounds()
			if dual_bounds is not None:
				dual_bounds = np.append( [0], dual_bounds )
			time = format_timespan( results[ai].avg_eps()[-1] )

			if get_option( "Tmax", None ) is not None:	# truncate time horizon
				Tmax = get_option( "Tmax", -1 )
				avg_regrets = avg_regrets[ Ts <= Tmax ]
				if dual_bounds is not None:
					dual_bounds = dual_bounds[ Ts <= Tmax ]
				Ts = Ts[ Ts <= Tmax ]

			if get_option( "hide_dual", False ) == True or alg.name in get_option( "hide_dual", [] ):
				dual_bounds = None

			alg_name = change_algname( alg.name )
			alg_label = alg_name
			if get_option( "hide_time", False ) == False:
				alg_label += " (%s)" % time

			line_width = 2
			if get_option( "emphasize", False ) == True:	# highlight regret of IRS algorithms
				if alg_name.startswith("IRS"):
					line_width = 2.5
				else:
					line_width = 1
			if alg_name in get_option( "highlight", [] ):
				line_width = 2.5

			p = pyplot.plot( Ts, avg_regrets, "-", lw=line_width, label=alg_label, c=get_color(alg_name) )
			side_labels.append( (Ts[-1], avg_regrets[-1], alg_name) )
			if lowest_avg_regrets[-1] > avg_regrets[-1]:
				lowest_avg_regrets = avg_regrets

			if dual_bounds is not None:
				pyplot.plot( Ts, dual_bounds, "--", lw=line_width, c=get_color(alg_name) )
				side_labels.append( (Ts[-1], dual_bounds[-1], alg_name) )
				if highest_dual_bounds[-1] < dual_bounds[-1]:
					highest_dual_bounds = dual_bounds

	if get_option( "opt_regrets", None ) is not None:
		opt_regrets = get_option("opt_regrets",None)
		p = pyplot.plot( np.arange(len(opt_regrets)), opt_regrets, "-", lw=2, label="OPT", c=COLORS["OPT"] )
		side_labels.append( (len(opt_regrets), opt_regrets[-1], "OPT") )


	if get_option( "hide_side_label", False ) == False:
		min_gap = (pyplot.ylim()[1] - pyplot.ylim()[0])/40.

		new_ys = adjust_label_positions( [ y for x,y,l in side_labels ], min_gap )

		for i in range( len(side_labels) ):
			x,y,l = side_labels[i]
			
			if get_option( "emphasize", False ) and l.startswith("IRS"):
				pyplot.text( x, new_ys[i], " "+l, fontweight="bold" )
			elif l in get_option( "highlight", [] ):
				pyplot.text( x, new_ys[i], " "+l, fontweight="bold" )
			else:
				pyplot.text( x, new_ys[i], " "+l )

	if get_option( "xlim", False ):
		xlim = pyplot.xlim()
		pyplot.xlim( (xlim[0] + get_option("xlim")[0], xlim[1] + get_option("xlim")[1]) )

	if get_option( "fill_between", False ):
		pyplot.fill_between( Ts, highest_dual_bounds, lowest_avg_regrets, hatch='///', facecolor="none", edgecolor='gray', alpha=0.5 )

	pyplot.title( title )
	
	pyplot.xlabel( r"Time horizon $T$" )
	pyplot.ylabel( r"Bayesian regret $W^{TS}(T,y)-V(\pi,T,y)$" )

	if get_option("hide_legend", False) == False:
		pyplot.legend( loc='upper left' )
	pyplot.savefig( "results/figures/" + filename + ".pdf", formoat="pdf", bbox_inches='tight' )
	pyplot.close()


	# 5. create a table in latex format for paper

	desired_order = ["TS", "IRS.FH", "IRS.V-ZERO", "IRS.V-EMAX", "IRS.INDEX", "IRS.INDEX*", "Bayes-UCB", "IDS", "OGI", "IRS.INDEX3", "IRS.INDEX.AVG"]

	alg_names = []
	regrets = []
	bounds = []

	times = []

	for an in desired_order:
		for ai, alg in enumerate(algs):
			if change_algname( alg.name ) != an:
				continue
			res = results[ai]
			T_max = res.Ts[-1]
			S = res.counts()[-1]

			alg_names.append( r"\textsc{%s}" % format_algname( change_algname( alg.name ) ) )

			regrets.append( "%.2f (%.3f)" % ( res.avg_regrets()[-1], res.stderr_regrets()[-1] ) )

			dual_bounds = res.avg_dual_bounds()
			if dual_bounds is not None:
				bounds.append( "%.2f (%.3f)" % ( dual_bounds[-1], res.stderr_dual_bounds()[-1] ) )
			else:
				bounds.append( "-" )
			times.append( format_timespan( res.avg_eps()[-1] ) )


	if get_option( "opt_regrets", None ) is not None:
		opt_regrets = get_option("opt_regrets",None)
		alg_names.append( r"\textsc{Opt}" )
		regrets.append( "%.2f (-)" % opt_regrets[ T_max ] )
		bounds.append( "%.2f (-)" % (2./3. * T_max - opt_regrets[ T_max ]) )
		times.append( "-" )

	ttt = r"""
\begin{tabular}{ c | c | c | c }
 \hline
\thead{Algorithm} & \thead{Bayesian regret} (std error) & \thead{Performance bound} (std error) & \thead{Run time} \\
 \hline
 \hline
"""
	
	for ai,an in enumerate(alg_names):
		ttt += r" %s & %s & %s & %s \\" % ( alg_names[ai], regrets[ai], bounds[ai], times[ai] ) 
		ttt += "\n"

	ttt += r"""
 \hline
\end{tabular}
"""
	
	fd = open( "results/tables/tabular_"+filename+".tex", "w" )
	fd.write( ttt )
	fd.close()







if __name__ == "__main__":

	bern2_opt = np.array( [0.000000, 0.500000, 1.083333, 1.666667, 2.277778, 2.888889, 3.508333, 4.131944, 4.759524, 5.387897, 6.021786, 6.656204, 7.292408, 7.930901, 8.570900, 9.211539, 9.853666, 10.496500, 11.140663, 11.785585, 12.431265, 13.077735, 13.724975, 14.372698, 15.020893, 15.669635, 16.318916, 16.968536, 17.618621, 18.269026, 18.919868, 19.571160, 20.222885, 20.874960, 21.527332, 22.179922, 22.832803, 23.485946, 24.139350, 24.792971, 25.446877, 26.100974, 26.755312, 27.409849, 28.064569, 28.719482, 29.374602, 30.029893, 30.685351, 31.340957, 31.996712, 32.652634, 33.308733, 33.964962, 34.621332, 35.277830, 35.934472, 36.591241, 37.248149, 37.905185, 38.562343, 39.219601, 39.876964, 40.534433, 41.192002, 41.849677, 42.507456, 43.165318, 43.823271, 44.481316, 45.139449, 45.797678, 46.456006, 47.114429, 47.772947, 48.431552, 49.090244, 49.749012, 50.407860, 51.066779, 51.725764, 52.384819, 53.043940, 53.703131, 54.362382, 55.021698, 55.681080, 56.340519, 57.000024, 57.659584, 58.319204, 58.978888, 59.638624, 60.298409, 60.958256, 61.618154, 62.278107, 62.938110, 63.598167, 64.258272, 64.918421, 65.578612, 66.238854, 66.899147, 67.559487, 68.219875, 68.880311, 69.540787, 70.201310, 70.861871, 71.522470, 72.183112, 72.843793, 73.504511, 74.165266, 74.826057, 75.486886, 76.147761, 76.808677, 77.469631, 78.130620, 78.791643, 79.452699, 80.113785, 80.774907, 81.436063, 82.097250, 82.758470, 83.419721, 84.081006, 84.742325, 85.403675, 86.065053, 86.726461, 87.387895, 88.049357, 88.710851, 89.372374, 90.033925, 90.695504, 91.357113, 92.018747, 92.680407, 93.342092, 94.003803, 94.665539, 95.327303, 95.989092, 96.650908, 97.312748, 97.974613, 98.636503, 99.298418, 99.960358, 100.622321, 101.284306, 101.946313, 102.608344, 103.270397, 103.932471, 104.594567, 105.256682, 105.918820, 106.580981, 107.243163, 107.905368, 108.567594, 109.229839, 109.892102, 110.554384, 111.216685, 111.879007, 112.541348, 113.203710, 113.866089, 114.528489, 115.190907, 115.853344, 116.515798, 117.178271, 117.840762, 118.503270, 119.165797, 119.828342, 120.490903, 121.153480, 121.816075, 122.478687, 123.141316, 123.803959, 124.466618, 125.129294, 125.791984, 126.454691, 127.117417, 127.780158, 128.442914, 129.105683, 129.768467, 130.431266, 131.094081, 131.756911] )
	bern2_opt_regrets = 2./3. * np.arange( bern2_opt.size ) - bern2_opt

	renew_all = ("all" in sys.argv)
	

	if renew_all or "bern_2arms" in sys.argv:
		publish_regret_plot( "bern_2arms", "bern_2arms_all", r"Beta-Bernoulli MAB ($K=2$)", (9,6),
			options = { "except":["OGI-T"], "emphasize":True, "opt_regrets":bern2_opt_regrets, "xlim":(-25,30) } )

	if renew_all or "bern_10arms" in sys.argv:
		publish_regret_plot( "bern_10arms", "bern_10arms_all", r"Beta-Bernoulli MAB ($K=10$)", (9,6),
			options = { "except":["OGI-T"], "emphasize":True, "xlim":(-25,100) } )


	if renew_all or "gauss_2arms" in sys.argv:
		publish_regret_plot( "gauss_2arms", "gauss_2arms_all", r"Gaussian MAB ($K=2$)", (9,6),
			options = { "except":["OGI-T"], "emphasize":True, "xlim":(-25,30) } )


	if renew_all or "gauss_10arms" in sys.argv:
		publish_regret_plot( "gauss_10arms", "gauss_10arms_all", r"Gaussian MAB ($K=10$)", (9,6),
			options = { "except":["OGI-T"], "emphasize":True, "xlim":(-25,100) } )


	if renew_all or "gauss_5arms_asym" in sys.argv:
		publish_regret_plot( "gauss_5arms_asym", "gauss_5arms_asym", r"Gaussian MAB ($K=5$) with Heteroscedastic Noise", (9,6),
			options = { "except":["OGI-T"], "emphasize":True, "xlim":(-25,100) } )

	if renew_all or "gauss_5arms_asym" in sys.argv:
		publish_regret_plot( "gauss_5arms_asym", "gauss_5arms_asym_shaded", r"Gaussian MAB ($K=5$) with Heteroscedastic Noise", (9,5),
			options = { "except":["OGI-T"], "emphasize":True, "xlim":(-25,100), "fill_between":True } )


