#import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio
import scipy.sparse.linalg as spalg

#from pylab import *
#%matplotlib inline

#import pylab as pl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.optimize as spopt

import pandas as pd

import pickle
import operator

import weblogolib


def get_logo(PFM, file_path, seq_length=6, normalize=False, output_format='png') :

		if normalize == True :
			for i in range(0, PFM.shape[0]) :
				if numpy.sum(PFM[i, :]) > 0 :
					PFM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])
				#PFM[i, :] *= 10000.0
			#print(PFM)
		

		#Create weblogo from API
		logo_output_format = output_format#"svg"
		#Load data from an occurence matrix
		data = weblogolib.LogoData.from_counts('ACGT', PFM[:seq_length, :])

		#Generate color scheme
		'''colors = weblogolib.ColorScheme([
		        weblogolib.ColorGroup("A", "yellow","CFI Binder" ),
		        weblogolib.ColorGroup("C", "green","CFI Binder" ),
		        weblogolib.ColorGroup("G", "red","CFI Binder" ),
		        weblogolib.ColorGroup("T", "blue","CFI Binder" ),
		        weblogolib.ColorGroup("a", "grey","CFI Binder" ),
		        weblogolib.ColorGroup("c", "grey","CFI Binder" ),
		        weblogolib.ColorGroup("g", "grey","CFI Binder" ),
		        weblogolib.ColorGroup("t", "grey","CFI Binder" )] )'''
		color_rules = []
		color_rules.append(weblogolib.SymbolColor("A", "yellow"))
		color_rules.append(weblogolib.SymbolColor("C", "green"))
		color_rules.append(weblogolib.SymbolColor("G", "red"))
		color_rules.append(weblogolib.SymbolColor("T", "blue"))
		colors = weblogolib.ColorScheme(color_rules)


		#Create options
		options = weblogolib.LogoOptions(fineprint=False,
		                                 logo_title="LOR filter", 
		                                 color_scheme=colors, 
		                                 stack_width=weblogolib.std_sizes["large"],
		                                 logo_start=1, logo_end=seq_length, stacks_per_line=seq_length)#seq_length)

		#Create logo
		logo_format = weblogolib.LogoFormat(data, options)

		#Generate image
		formatter = weblogolib.formatters[logo_output_format]
		png = formatter(data, logo_format)

		#Write it
		f = open(file_path, "w")
		f.write(png)
		f.close()


w = np.ravel(np.load('doubledope1_pas_6mers_logodds_ratio.npy'))

print(w.shape)


bases = "ACGT"

mer4 = []
mer6 = []
for base1 in bases:
	for base2 in bases:
		for base3 in bases:
			for base4 in bases:
				mer4.append(base1 + base2 + base3 + base4)
				for base5 in bases:
					for base6 in bases:
						mer6.append(base1 + base2 + base3 + base4 + base5 + base6)



def get_pwm_identity(seq) :
	pwm = np.zeros((len(seq), 4))

	for i in range(0, len(seq)) :
		if seq[i] == 'A' :
			pwm[i, 0] = 1
		elif seq[i] == 'C' :
			pwm[i, 1] = 1
		elif seq[i] == 'G' :
			pwm[i, 2] = 1
		elif seq[i] == 'T' :
			pwm[i, 3] = 1
		elif seq[i] == 'N' :
			pwm[i, :] = 0.25
	return pwm



w_selection = w[:]

mer_avg_score = {}
mer_counts = {}

mer = mer6
num_mers = 4096
for i in range(0, len(w_selection)) :
	motif = mer[i]

	#6mer acc
	if motif not in mer_avg_score:
		mer_avg_score[motif] = 0
		mer_counts[motif] = 0
	mer_avg_score[motif] += w_selection[i]
	mer_counts[motif] += 1

for motif in mer_avg_score :
	mer_avg_score[motif] /= mer_counts[motif]

n_filters_top = 1
filter_width = 6

curr_filter = 0

interpolation_factor = 0.45#0.65#0.45#0.65#0.25

#Enhancer filters
sorted_avg_score = sorted(mer_avg_score.items(), key=operator.itemgetter(1))[::-1]

for motif_tuple in sorted_avg_score :
	motif = motif_tuple[0]

	if curr_filter >= n_filters_top :
		break

	if len(motif) != 6 :
		continue

	#print(str(curr_filter) + ': ' + motif)
	pwm = np.zeros((len(motif), 4))
	pwm = get_pwm_identity(motif) * (np.exp(mer_avg_score[motif]) )

	
	#Smooth with 2-mut motifs
	neighbors = {}
	for i in range(0, 6) :
		for b1 in ['A', 'C', 'G', 'T'] :
			neighbor = motif[:i] + b1 + motif[i+1:]
			if mer_avg_score[neighbor] != 0.0 and mer_avg_score[neighbor] <= mer_avg_score[motif] :
				neighbors[neighbor] = (np.exp(mer_avg_score[neighbor]) )

	pwm_neighbors = np.zeros((len(motif), 4))

	sorted_neighbor_score = sorted(neighbors.items(), key=operator.itemgetter(1))[::-1]

	n_neighbors = len(sorted_neighbor_score)#8#8#4

	curr_neighbor = 0
	for neighbor in sorted_neighbor_score :
		if curr_neighbor >= n_neighbors :
			break

		if neighbors[neighbor[0]] > 0 :
			pwm_neighbors += get_pwm_identity(neighbor[0]) * neighbors[neighbor[0]]

		curr_neighbor += 1
		

	#pwm_neighbors /= float(n_neighbors)
	#pwm /= np.sum(pwm[1, :])
	#pwm_neighbors /= np.sum(pwm_neighbors[1, :])

	pwm = pwm_neighbors #* interpolation_factor

	#Normalize pwm
	pwm /= np.sum(pwm[1, :])

	get_logo(pwm, 'pwm_init_logodds_ratio_doubledope_pas.png', 6, False, output_format='png')
	get_logo(pwm, 'pwm_init_logodds_ratio_doubledope_pas.eps', 6, False, output_format='eps')

	curr_filter += 1





w = np.ravel(np.load('simple1_cutsite_2mer_logodds_ratio.npy'))

print(w.shape)


bases = "ACGT"

mer2 = []
for base1 in bases:
	for base2 in bases:
		mer2.append(base1 + base2)



def get_pwm_identity(seq) :
	pwm = np.zeros((len(seq), 4))

	for i in range(0, len(seq)) :
		if seq[i] == 'A' :
			pwm[i, 0] = 1
		elif seq[i] == 'C' :
			pwm[i, 1] = 1
		elif seq[i] == 'G' :
			pwm[i, 2] = 1
		elif seq[i] == 'T' :
			pwm[i, 3] = 1
		elif seq[i] == 'N' :
			pwm[i, :] = 0.25
	return pwm



w_selection = w[:]

mer_avg_score = {}
mer_counts = {}

mer = mer2
num_mers = 16
for i in range(0, len(w_selection)) :
	motif = mer[i]

	#6mer acc
	if motif not in mer_avg_score:
		mer_avg_score[motif] = 0
		mer_counts[motif] = 0
	mer_avg_score[motif] += w_selection[i]
	mer_counts[motif] += 1

for motif in mer_avg_score :
	mer_avg_score[motif] /= mer_counts[motif]

n_filters_top = 1
filter_width = 2

curr_filter = 0

interpolation_factor = 0.45#0.65#0.45#0.65#0.25

#Enhancer filters
sorted_avg_score = sorted(mer_avg_score.items(), key=operator.itemgetter(1))[::-1]

for motif_tuple in sorted_avg_score :
	motif = motif_tuple[0]

	if curr_filter >= n_filters_top :
		break

	if len(motif) != 2 :
		continue

	#print(str(curr_filter) + ': ' + motif)
	pwm = np.zeros((len(motif), 4))
	pwm = get_pwm_identity(motif) * (np.exp(mer_avg_score[motif]) )

	
	#Smooth with 2-mut motifs
	neighbors = {}
	for i in range(0, 2) :
		for b1 in ['A', 'C', 'G', 'T'] :
			neighbor = motif[:i] + b1 + motif[i+1:]
			if mer_avg_score[neighbor] > -0.3 and mer_avg_score[neighbor] <= mer_avg_score[motif] :
				neighbors[neighbor] = (np.exp(mer_avg_score[neighbor]) )

	pwm_neighbors = np.zeros((len(motif), 4))

	sorted_neighbor_score = sorted(neighbors.items(), key=operator.itemgetter(1))[::-1]

	n_neighbors = len(sorted_neighbor_score)#8#8#4

	curr_neighbor = 0
	for neighbor in sorted_neighbor_score :
		if curr_neighbor >= n_neighbors :
			break

		if neighbors[neighbor[0]] > 0 :
			pwm_neighbors += get_pwm_identity(neighbor[0]) * neighbors[neighbor[0]]

		curr_neighbor += 1
		

	#pwm_neighbors /= float(n_neighbors)
	#pwm /= np.sum(pwm[1, :])
	#pwm_neighbors /= np.sum(pwm_neighbors[1, :])

	pwm = pwm_neighbors #* interpolation_factor

	#Normalize pwm
	pwm /= np.sum(pwm[1, :])

	get_logo(pwm, 'pwm_init_logodds_ratio_simple_cutsite.png', 2, False, output_format='png')
	get_logo(pwm, 'pwm_init_logodds_ratio_simple_cutsite.eps', 2, False, output_format='eps')

	curr_filter += 1