import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

import regex as re

def compute_ratios(proximal_counts, total_counts_vs_distal, total_counts_vs_all):
	y = np.zeros((len(proximal_counts), 3))
	isprox = np.zeros((len(proximal_counts), 1))
	c = np.zeros((len(proximal_counts), 1))
	
	for i in range(0, len(proximal_counts)) :
		prox = float(proximal_counts[i])
		count_vs_distal = float(total_counts_vs_distal[i])
		count_vs_all = float(total_counts_vs_all[i])
		is_prox = 0
		if prox > 0 :
			is_prox = 1
		
		PR_vs_distal = 0
		if count_vs_distal > 0 :
			PR_vs_distal = prox / count_vs_distal
		PR_vs_all = prox / count_vs_all
		
		all_vs_distal = (prox + (count_vs_all - count_vs_distal)) / count_vs_all

		y[i,0] = PR_vs_distal
		y[i,1] = PR_vs_all
		y[i,2] = all_vs_distal
		isprox[i,0] = is_prox
		c[i,0] = count_vs_all
	return y, c, isprox



data1 = pd.read_csv('apa_nextseq_v2_library_merged2_20161001_general.csv',sep=',')
print(len(data1))
data1 = data1.ix[data1.total_count_vs_distal >= 12]
print(len(data1))

data2 = pd.read_csv('apa_sym_prx_library_20160916_general.csv',sep=',')
print(len(data2))
data2 = data2.ix[data2.library == 20]
data2 = data2.ix[data2.total_count_vs_distal >= 6]
print(len(data2))

data3 = pd.read_csv('simple_general.csv',sep=',')
print(len(data3))
data3 = data3.ix[data3.library == 22]
data3 = data3.ix[(data3.seq.str.slice(50, 56) == 'AATAAA') | (data3.seq.str.slice(50, 56) == 'ATTAAA')]
data3 = data3.ix[(data3.seq.str.slice(101, 107) == 'AATAAA') | (data3.seq.str.slice(101, 107) == 'ATTAAA')]
data3 = data3.ix[data3.total_count_vs_all >= 4]
print(len(data3))

data4 = pd.read_csv('apasix2_general.csv',sep=',')
print(len(data4))

data4 = data4.ix[
	(((((((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 30)) |
	((data4.seq.str.slice(50, 56) == 'CATAAA') & (data4.library == 31))) |
	((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 32))) |
	((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 33))) |
	((data4.seq.str.slice(50, 56) == 'AATAAT') & (data4.library == 34))) |
	((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 35)))
]

data4 = data4.ix[data4.total_count_vs_distal >= 10]
print(len(data4))


print('Removing mispriming candidates.')

misprime_regex1 = re.compile(r"(AAAAAAAAAAAA){s<=2}")
misprime_regex2 = re.compile(r"(AAAAAAAAAAAAAAAA){s<=4}")
misprime_regex3 = re.compile(r"(AAAAAAAAAAAAAAAAAAAA){s<=5}")

no_misprime_index1 = []
for index, row in data1.iterrows() :
	curr_seq = row['seq']

	if re.search(misprime_regex1, curr_seq) or re.search(misprime_regex2, curr_seq) or re.search(misprime_regex3, curr_seq) :
		no_misprime_index1.append(False)
	else :
		no_misprime_index1.append(True)

data1 = data1.ix[no_misprime_index1]
print(len(data1))

no_misprime_index2 = []
for index, row in data2.iterrows() :
	curr_seq = row['seq']

	if re.search(misprime_regex1, curr_seq) or re.search(misprime_regex2, curr_seq) or re.search(misprime_regex3, curr_seq) :
		no_misprime_index2.append(False)
	else :
		no_misprime_index2.append(True)

data2 = data2.ix[no_misprime_index2]
print(len(data2))

no_misprime_index3 = []
for index, row in data3.iterrows() :
	curr_seq = row['seq']

	if re.search(misprime_regex1, curr_seq) or re.search(misprime_regex2, curr_seq) or re.search(misprime_regex3, curr_seq) :
		no_misprime_index3.append(False)
	else :
		no_misprime_index3.append(True)

data3 = data3.ix[no_misprime_index3]
print(len(data3))

no_misprime_index4 = []
for index, row in data4.iterrows() :
	curr_seq = row['seq']

	if (re.search(misprime_regex1, curr_seq) or re.search(misprime_regex2, curr_seq) or re.search(misprime_regex3, curr_seq)) and row['library'] != 35 :
		no_misprime_index4.append(False)
	else :
		no_misprime_index4.append(True)

data4 = data4.ix[no_misprime_index4]
print(len(data4))



filtered = '_general3_antimisprime_orig'


action = '_seq'
#action = '_6mer_v'
#action = '_fullseq'
#action = '_fullseq_small'

proximal_counts = list(data1.proximal_count)
total_counts_vs_distal = list(data1.total_count_vs_distal)
total_counts_vs_all = list(data1.total_count_vs_all)

avgcut1 = np.ravel(np.array(list(data1.proximal_avgcut)))
stdcut1 = np.ravel(np.array(list(data1.proximal_stdcut)))

y1, c1, p1 = compute_ratios(proximal_counts, total_counts_vs_distal, total_counts_vs_all)

proximal_counts = list(data2.proximal_count)
total_counts_vs_distal = list(data2.total_count_vs_distal)
total_counts_vs_all = list(data2.total_count_vs_all)

y2, c2, p2 = compute_ratios(proximal_counts, total_counts_vs_distal, total_counts_vs_all)

proximal_counts = list(data3.proximal_count)
total_counts_vs_distal = list(data3.total_count_vs_distal)
total_counts_vs_all = list(data3.total_count_vs_all)

avgcut3 = np.ravel(np.array(list(data3.proximal_avgcut)))
stdcut3 = np.ravel(np.array(list(data3.proximal_stdcut)))

y3, c3, p3 = compute_ratios(proximal_counts, total_counts_vs_distal, total_counts_vs_all)

proximal_counts = list(data4.proximal_count)
total_counts_vs_distal = list(data4.total_count_vs_distal)
total_counts_vs_all = list(data4.total_count_vs_all)

avgcut4 = np.ravel(np.array(list(data4.proximal_avgcut)))
stdcut4 = np.ravel(np.array(list(data4.proximal_stdcut)))

y4, c4, p4 = compute_ratios(proximal_counts, total_counts_vs_distal, total_counts_vs_all)



print(y1.shape)
print(y2.shape)
print(y3.shape)
print(y4.shape)

distalpas = np.concatenate([np.ones((len(y1), 1)), np.ones((len(y2), 1)), np.zeros((len(y3), 1)), np.zeros((len(y4), 1))], axis=0)

library1 = list(data1.library)
library2 = list(data2.library)
library3 = list(data3.library)
library4 = list(data4.library)

avgcut = np.concatenate([avgcut1, np.zeros(len(y2)), avgcut3, avgcut4], axis=0)
stdcut = np.concatenate([stdcut1, np.zeros(len(y2)), stdcut3, stdcut4], axis=0)
print(avgcut.shape)
print(stdcut.shape)

y = np.concatenate([y1, y2, y3, y4], axis=0)
c = np.concatenate([c1, c2, c3, c4], axis=0)
p = np.concatenate([p1, p2, p3, p4], axis=0)
L = np.zeros((len(y), 1))
i = 0
for j in range(0, len(library1)) :
	L[i, 0] = library1[j]
	i += 1
for j in range(0, len(library2)) :
	L[i, 0] = library2[j]
	i += 1
for j in range(0, len(library3)) :
	L[i, 0] = library3[j]
	i += 1
for j in range(0, len(library4)) :
	L[i, 0] = library4[j]
	i += 1

print(y.shape)

np.save('apa' + filtered + '_avgcut', avgcut)
np.save('apa' + filtered + '_stdcut', stdcut)
np.save('apa' + filtered + '_output', y)
np.save('apa' + filtered + '_count', c)
np.save('apa' + filtered + '_ispas', p)
np.save('apa' + filtered + '_libindex', L)
np.save('apa' + filtered + '_distalpas', distalpas)

seq1 = (data1.seq.str.slice(1, 186)).values
fullseq1 = (data1.seq_ext.str.slice(1, 256)).values
fullseq_small1 = (data1.seq_ext.str.slice(77, 176)).values

seq2 = (data2.seq.str.slice(1, 186)).values
fullseq2 = (data2.seq_ext.str.slice(1, 256)).values
fullseq_small2 = (data2.seq_ext.str.slice(77, 176)).values

seq3 = (data3.seq.str.slice(1, 186)).values
fullseq3 = (data3.seq_ext.str.slice(1, 256)).values
fullseq_small3 = (data3.seq_ext.str.slice(77, 176)).values

seq4 = (data4.seq.str.slice(1, 186)).values
#fullseq4 = (data4.seq_ext.str.slice(1, 256)).values
#fullseq_small4 = (data4.seq_ext.str.slice(77, 176)).values

#Initialize the matrix of input feature vectors.

if action == '_seq' :
	X_motif = sp.lil_matrix((len(y), 185 * 4), dtype=np.int8)
if action == '_fullseq' :
	X_motif = sp.lil_matrix((len(y), 255 * 4), dtype=np.int8)
if action == '_fullseq_small' :
	X_motif = sp.lil_matrix((len(y), 99 * 4), dtype=np.int8)
if action == '_6mer_v' :
	X_motif = sp.lil_matrix((len(y), 4096 * 4), dtype=np.int8)

F = sp.lil_matrix((len(y), 6), dtype=np.int8)

cano_pas = 'AATAAA'
pas_mutex1 = []
for pos in range(0, 6) :
	for base in ['A', 'C', 'G', 'T'] :
		pas_mutex1.append(cano_pas[:pos] + base + cano_pas[1+pos:])


#Define library sequence bias masks

mask_dict = {
    2  : 'XXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    5  : 'XXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    8  : 'XXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    11 : 'XXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    20 : 'XXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXXX',
    22 : 'XXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    30 : 'XXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    31 : 'XXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    32 : 'XXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    33 : 'XNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNXNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    34 : 'XXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    35 : 'XXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
}

#Initialize mer dicts
bases = 'ACGT'
mer8 = []
mer8_dict = dict()
mer6 = []
mer6_dict = dict()
mer4 = []
mer4_dict = dict()
mer3 = []
mer3_dict = dict()
mer2 = []
mer2_dict = dict()
mer1 = []
mer1_dict = dict()

mer8_index = 0
mer6_index = 0
mer4_index = 0
mer3_index = 0
mer2_index = 0
mer1_index = 0
for base1 in bases:
	mer1.append(base1)
	mer1_dict[base1] = mer1_index
	mer1_index = mer1_index + 1
	for base2 in bases:
		mer2.append(base1 + base2)
		mer2_dict[base1 + base2] = mer2_index
		mer2_index = mer2_index + 1
		for base3 in bases:
			mer3.append(base1 + base2 + base3)
			mer3_dict[base1 + base2 + base3] = mer3_index
			mer3_index = mer3_index + 1
			for base4 in bases:
				mer4.append(base1 + base2 + base3 + base4)
				mer4_dict[base1 + base2 + base3 + base4] = mer4_index
				mer4_index = mer4_index + 1
				for base5 in bases:
					for base6 in bases:
						mer6.append(base1 + base2 + base3 + base4 + base5 + base6)
						mer6_dict[base1 + base2 + base3 + base4 + base5 + base6] = mer6_index
						mer6_index = mer6_index + 1
						for base7 in bases:
							for base8 in bases:
								mer8.append(base1 + base2 + base3 + base4 + base5 + base6 + base7 + base8)
								mer8_dict[base1 + base2 + base3 + base4 + base5 + base6 + base7 + base8] = mer8_index
								mer8_index = mer8_index + 1



#Empty dataframes.
data1 = None
data2 = None
data3 = None
data4 = None

dump_counter = 0
dump_max = 30000
X_motif_acc = None
X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
F_acc = None
F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
seqs_left = len(y1)

for i in range(0, len(y1)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	if dump_counter >= dump_max :
		
		if X_motif_acc == None :
			X_motif_acc = sp.csr_matrix(X_motif, dtype=np.int8)
			F_acc = sp.csr_matrix(F, dtype=np.int8)
		else :
			X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
			F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])
		
		if seqs_left >= dump_max :
			X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
		else :
			X_motif = sp.lil_matrix((seqs_left, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((seqs_left, F.shape[1]), dtype=np.int8)
		dump_counter = 0
	
	dump_counter += 1
	seqs_left -= 1

	full_seq = fullseq1[i]
	full_seq_small = fullseq_small1[i]
	seq = seq1[i]

	#Build extra features
	#0. Seq contains Canonical pPAS AATAAA
	if 'AATAAA' in seq[46:46+10] :
		F[i % dump_max, 0] = 1
	#1. Seq contains Canonical pPAS ATTAAA
	if 'ATTAAA' in seq[46:46+10] :
		F[i % dump_max, 1] = 1
	#2. Seq contains competing PAS AATAAA
	if 'AATAAA' in seq[:45] or 'AATAAA' in seq[57:] :
		F[i % dump_max, 2] = 1
	#3. Seq contains competing PAS ATTAAA
	if 'ATTAAA' in seq[:45] or 'ATTAAA' in seq[57:] :
		F[i % dump_max, 3] = 1
	#4. Seq contains 1-base mut of competing PAS AATAAA
	for pas_mut1 in pas_mutex1 :
		if pas_mut1 in seq[:45] or pas_mut1 in seq[57:] :
			F[i % dump_max, 4] = 1
			break

	if action == '_6mer_v' :
		lib_i = int(library1[i])

		seq_var = seq
		if lib_i in mask_dict :
		    library_mask = mask_dict[lib_i]
		    seq_var = ''
		    for j in range(0, len(seq)) :
		        if library_mask[j] == 'N' :
		            seq_var += seq[j]
		        elif j >= 2 and (library_mask[j-1] == 'N' or library_mask[j-2] == 'N') :
		        	seq_var += seq[j]
		        elif j <= len(seq) - 1 - 2 and (library_mask[j+1] == 'N' or library_mask[j+2] == 'N') :
		        	seq_var += seq[j]
		        else :
		            seq_var += 'X'

		#Upstream region
		seq_var_upstream = seq_var[0:49]
		#PAS region
		seq_var_pas = seq_var[49 - 1:55 + 1]
		#Downstream region
		seq_var_downstream = seq_var[55: 98]
		#Further Downstream region
		seq_var_fdownstream = seq_var[98:]

		for j in range(0, len(seq_var_upstream)-5):
			motif = seq_var_upstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_pas)-5):
			motif = seq_var_pas[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 1 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_downstream)-5):
			motif = seq_var_downstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 2 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_fdownstream)-5):
			motif = seq_var_fdownstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 3 * 4096 + mer6_dict[motif]] += 1
	
	if action == '_seq' :
		X_point = np.zeros((len(seq), 4))

		for j in range(0, len(seq)) :
			if seq[j] == "A" :
				X_point[j, 0] = 1
			elif seq[j] == "C" :
				X_point[j, 1] = 1
			elif seq[j] == "G" :
				X_point[j, 2] = 1
			elif seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(seq) * 4))
	
	if action == '_fullseq' :
		X_point = np.zeros((len(full_seq), 4))

		for j in range(0, len(full_seq)) :
			if full_seq[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq) * 4))
	
	if action == '_fullseq_small' :
		X_point = np.zeros((len(full_seq_small), 4))
		
		for j in range(0, len(full_seq_small)) :
			if full_seq_small[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq_small[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq_small[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq_small[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq_small) * 4))
	

X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])

dump_counter = 0
dump_max = 30000
X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
seqs_left = len(y2)

for i in range(0, len(y2)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	if dump_counter >= dump_max :
		
		if X_motif_acc == None :
			X_motif_acc = sp.csr_matrix(X_motif, dtype=np.int8)
			F_acc = sp.csr_matrix(F, dtype=np.int8)
		else :
			X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
			F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])
		
		if seqs_left >= dump_max :
			X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
		else :
			X_motif = sp.lil_matrix((seqs_left, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((seqs_left, F.shape[1]), dtype=np.int8)
		dump_counter = 0
	
	dump_counter += 1
	seqs_left -= 1

	full_seq = fullseq2[i]
	full_seq_small = fullseq_small2[i]
	seq = seq2[i]

	#Build extra features
	#0. Seq contains Canonical pPAS AATAAA
	if 'AATAAA' in seq[46:46+10] :
		F[i % dump_max, 0] = 1
	#1. Seq contains Canonical pPAS ATTAAA
	if 'ATTAAA' in seq[46:46+10] :
		F[i % dump_max, 1] = 1
	#2. Seq contains competing PAS AATAAA
	if 'AATAAA' in seq[:45] or 'AATAAA' in seq[57:] :
		F[i % dump_max, 2] = 1
	#3. Seq contains competing PAS ATTAAA
	if 'ATTAAA' in seq[:45] or 'ATTAAA' in seq[57:] :
		F[i % dump_max, 3] = 1
	#4. Seq contains 1-base mut of competing PAS AATAAA
	for pas_mut1 in pas_mutex1 :
		if pas_mut1 in seq[:45] or pas_mut1 in seq[57:] :
			F[i % dump_max, 4] = 1
			break
	
	if action == '_6mer_v' :
		lib_i = int(library2[i])

		seq_var = seq
		if lib_i in mask_dict :
		    library_mask = mask_dict[lib_i]
		    seq_var = ''
		    for j in range(0, len(seq)) :
		        if library_mask[j] == 'N' :
		            seq_var += seq[j]
		        elif j >= 2 and (library_mask[j-1] == 'N' or library_mask[j-2] == 'N') :
		        	seq_var += seq[j]
		        elif j <= len(seq) - 1 - 2 and (library_mask[j+1] == 'N' or library_mask[j+2] == 'N') :
		        	seq_var += seq[j]
		        else :
		            seq_var += 'X'

		#Upstream region
		seq_var_upstream = seq_var[0:49]
		#PAS region
		seq_var_pas = seq_var[49 - 1:55 + 1]
		#Downstream region
		seq_var_downstream = seq_var[55: 98]
		#Further Downstream region
		seq_var_fdownstream = seq_var[98:]

		for j in range(0, len(seq_var_upstream)-5):
			motif = seq_var_upstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_pas)-5):
			motif = seq_var_pas[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 1 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_downstream)-5):
			motif = seq_var_downstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 2 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_fdownstream)-5):
			motif = seq_var_fdownstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 3 * 4096 + mer6_dict[motif]] += 1

	if action == '_seq' :
		X_point = np.zeros((len(seq), 4))

		for j in range(0, len(seq)) :
			if seq[j] == "A" :
				X_point[j, 0] = 1
			elif seq[j] == "C" :
				X_point[j, 1] = 1
			elif seq[j] == "G" :
				X_point[j, 2] = 1
			elif seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(seq) * 4))
	
	if action == '_fullseq' :
		X_point = np.zeros((len(full_seq), 4))

		for j in range(0, len(full_seq)) :
			if full_seq[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq) * 4))
	
	if action == '_fullseq_small' :
		X_point = np.zeros((len(full_seq_small), 4))
		
		for j in range(0, len(full_seq_small)) :
			if full_seq_small[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq_small[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq_small[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq_small[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq_small) * 4))

X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])

dump_counter = 0
dump_max = 30000
X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
seqs_left = len(y3)

for i in range(0, len(y3)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	if dump_counter >= dump_max :
		
		if X_motif_acc == None :
			X_motif_acc = sp.csr_matrix(X_motif, dtype=np.int8)
			F_acc = sp.csr_matrix(F, dtype=np.int8)
		else :
			X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
			F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])
		
		if seqs_left >= dump_max :
			X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
		else :
			X_motif = sp.lil_matrix((seqs_left, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((seqs_left, F.shape[1]), dtype=np.int8)
		dump_counter = 0
	
	dump_counter += 1
	seqs_left -= 1

	full_seq = fullseq3[i]
	full_seq_small = fullseq_small3[i]
	seq = seq3[i]

	#Build extra features
	#0. Seq contains Canonical pPAS AATAAA
	if 'AATAAA' in seq[46:46+10] :
		F[i % dump_max, 0] = 1
	#1. Seq contains Canonical pPAS ATTAAA
	if 'ATTAAA' in seq[46:46+10] :
		F[i % dump_max, 1] = 1
	#2. Seq contains competing PAS AATAAA
	if 'AATAAA' in seq[:45] or 'AATAAA' in seq[57:] :
		F[i % dump_max, 2] = 1
	#3. Seq contains competing PAS ATTAAA
	if 'ATTAAA' in seq[:45] or 'ATTAAA' in seq[57:] :
		F[i % dump_max, 3] = 1
	#4. Seq contains 1-base mut of competing PAS AATAAA
	for pas_mut1 in pas_mutex1 :
		if pas_mut1 in seq[:45] or pas_mut1 in seq[57:] :
			F[i % dump_max, 4] = 1
			break
	if 'AATAAA' in seq[0:48] or 'AATAAA' in seq[58:96] :
		F[i % dump_max, 5] = 1

	if action == '_6mer_v' :
		lib_i = int(library3[i])

		seq_var = seq
		if lib_i in mask_dict :
		    library_mask = mask_dict[lib_i]
		    seq_var = ''
		    for j in range(0, len(seq)) :
		        if library_mask[j] == 'N' :
		            seq_var += seq[j]
		        elif j >= 2 and (library_mask[j-1] == 'N' or library_mask[j-2] == 'N') :
		        	seq_var += seq[j]
		        elif j <= len(seq) - 1 - 2 and (library_mask[j+1] == 'N' or library_mask[j+2] == 'N') :
		        	seq_var += seq[j]
		        else :
		            seq_var += 'X'

		#Upstream region
		seq_var_upstream = seq_var[0:49]
		#PAS region
		seq_var_pas = seq_var[49 - 1:55 + 1]
		#Downstream region
		seq_var_downstream = seq_var[55: 98]
		#Further Downstream region
		seq_var_fdownstream = seq_var[98:]

		for j in range(0, len(seq_var_upstream)-5):
			motif = seq_var_upstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_pas)-5):
			motif = seq_var_pas[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 1 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_downstream)-5):
			motif = seq_var_downstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 2 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_fdownstream)-5):
			motif = seq_var_fdownstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 3 * 4096 + mer6_dict[motif]] += 1
	
	if action == '_seq' :
		X_point = np.zeros((len(seq), 4))

		for j in range(0, len(seq)) :
			if seq[j] == "A" :
				X_point[j, 0] = 1
			elif seq[j] == "C" :
				X_point[j, 1] = 1
			elif seq[j] == "G" :
				X_point[j, 2] = 1
			elif seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(seq) * 4))
	
	if action == '_fullseq' :
		X_point = np.zeros((len(full_seq), 4))

		for j in range(0, len(full_seq)) :
			if full_seq[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq) * 4))
	
	if action == '_fullseq_small' :
		X_point = np.zeros((len(full_seq_small), 4))
		
		for j in range(0, len(full_seq_small)) :
			if full_seq_small[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq_small[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq_small[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq_small[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq_small) * 4))

X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])

dump_counter = 0
dump_max = 30000
X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
seqs_left = len(y4)

for i in range(0, len(y4)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	if dump_counter >= dump_max :
		
		if X_motif_acc == None :
			X_motif_acc = sp.csr_matrix(X_motif, dtype=np.int8)
			F_acc = sp.csr_matrix(F, dtype=np.int8)
		else :
			X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
			F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])
		
		if seqs_left >= dump_max :
			X_motif = sp.lil_matrix((dump_max, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((dump_max, F.shape[1]), dtype=np.int8)
		else :
			X_motif = sp.lil_matrix((seqs_left, X_motif.shape[1]), dtype=np.int8)
			F = sp.lil_matrix((seqs_left, F.shape[1]), dtype=np.int8)
		dump_counter = 0
	
	dump_counter += 1
	seqs_left -= 1

	full_seq = ''#fullseq4[i]
	full_seq_small = ''#fullseq_small4[i]
	seq = seq4[i]

	#Build extra features
	#0. Seq contains Canonical pPAS AATAAA
	if 'AATAAA' in seq[46:46+10] :
		F[i % dump_max, 0] = 1
	#1. Seq contains Canonical pPAS ATTAAA
	if 'ATTAAA' in seq[46:46+10] :
		F[i % dump_max, 1] = 1
	#2. Seq contains competing PAS AATAAA
	if 'AATAAA' in seq[:45] or 'AATAAA' in seq[57:] :
		F[i % dump_max, 2] = 1
	#3. Seq contains competing PAS ATTAAA
	if 'ATTAAA' in seq[:45] or 'ATTAAA' in seq[57:] :
		F[i % dump_max, 3] = 1
	#4. Seq contains 1-base mut of competing PAS AATAAA
	for pas_mut1 in pas_mutex1 :
		if pas_mut1 in seq[:45] or pas_mut1 in seq[57:] :
			F[i % dump_max, 4] = 1
			break

	if action == '_6mer_v' :
		lib_i = int(library4[i])

		seq_var = seq
		if lib_i in mask_dict :
		    library_mask = mask_dict[lib_i]
		    seq_var = ''
		    for j in range(0, len(seq)) :
		        if library_mask[j] == 'N' :
		            seq_var += seq[j]
		        elif j >= 2 and (library_mask[j-1] == 'N' or library_mask[j-2] == 'N') :
		        	seq_var += seq[j]
		        elif j <= len(seq) - 1 - 2 and (library_mask[j+1] == 'N' or library_mask[j+2] == 'N') :
		        	seq_var += seq[j]
		        else :
		            seq_var += 'X'

		#Upstream region
		seq_var_upstream = seq_var[0:49]
		#PAS region
		seq_var_pas = seq_var[49 - 1:55 + 1]
		#Downstream region
		seq_var_downstream = seq_var[55: 98]
		#Further Downstream region
		seq_var_fdownstream = seq_var[98:]

		for j in range(0, len(seq_var_upstream)-5):
			motif = seq_var_upstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_pas)-5):
			motif = seq_var_pas[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 1 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_downstream)-5):
			motif = seq_var_downstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 2 * 4096 + mer6_dict[motif]] += 1
		for j in range(0, len(seq_var_fdownstream)-5):
			motif = seq_var_fdownstream[j:j+6]
			if motif in mer6_dict :
				X_motif[i % dump_max, 3 * 4096 + mer6_dict[motif]] += 1
	
	if action == '_seq' :
		X_point = np.zeros((len(seq), 4))

		for j in range(0, len(seq)) :
			if seq[j] == "A" :
				X_point[j, 0] = 1
			elif seq[j] == "C" :
				X_point[j, 1] = 1
			elif seq[j] == "G" :
				X_point[j, 2] = 1
			elif seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(seq) * 4))
	
	if action == '_fullseq' :
		X_point = np.zeros((len(full_seq), 4))

		for j in range(0, len(full_seq)) :
			if full_seq[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq) * 4))
	
	if action == '_fullseq_small' :
		X_point = np.zeros((len(full_seq_small), 4))
		
		for j in range(0, len(full_seq_small)) :
			if full_seq_small[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq_small[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq_small[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq_small[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif[i % dump_max,:] = X_point.reshape((1, len(full_seq_small) * 4))

X_motif_acc = sp.vstack([X_motif_acc, sp.csr_matrix(X_motif, dtype=np.int8)])
F_acc = sp.vstack([F_acc, sp.csr_matrix(F, dtype=np.int8)])

csr_X_motif = X_motif_acc
csr_F = F_acc

np.savez('npz_apa' + action + filtered + '_features', data=csr_F.data, indices=csr_F.indices, indptr=csr_F.indptr, shape=csr_F.shape )
np.savez('npz_apa' + action + filtered + '_input', data=csr_X_motif.data, indices=csr_X_motif.indices, indptr=csr_X_motif.indptr, shape=csr_X_motif.shape )