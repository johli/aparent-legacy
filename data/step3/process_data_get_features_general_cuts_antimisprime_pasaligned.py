import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import regex as re

def spoof_cut_distribution_sym_prx(cut_distribution, start_index, proximal_counts, total_counts_vs_distal, total_counts_vs_all) :
	for i in range(0, len(proximal_counts)) :
		if i % 100000 == 0:
			print("Counted up to sequence: " + str(i))

		prox = float(proximal_counts[i])
		count_vs_distal = float(total_counts_vs_distal[i])
		count_vs_all = float(total_counts_vs_all[i])

		down_prox = count_vs_all - count_vs_distal

		dist = count_vs_distal - prox

		cut_distribution[start_index + i, 55+5:75+5] = prox / 20.0
		cut_distribution[start_index + i, 140+5:160+5] = down_prox / 20.0
		cut_distribution[start_index + i, 185] = dist

	return cut_distribution


def compute_cut_distribution(cut_distribution, start_index, cuts) :
	for i in range(0, len(cuts)) :
		if i % 100000 == 0:
			print("Counted up to sequence: " + str(i))

		cut_str = cuts[i]
		cut_str = cut_str[1:len(cut_str)-1]
		cuts_member = cut_str.split(', ')

		for j in range(0, len(cuts_member)) :
			cutpos = int(cuts_member[j])
			cut_distribution[start_index + i, cutpos - 1] += 1.0

	return cut_distribution



data1 = pd.read_csv('apa_nextseq_v2_library_merged2_20161001_general_cuts.csv',sep=',')
print(len(data1))
data1 = data1.ix[data1.total_count >= 12]
print(len(data1))

data2 = pd.read_csv('apa_sym_prx_library_20160916_general.csv',sep=',')
print(len(data2))
data2 = data2.ix[data2.library == 20]
data2 = data2.ix[data2.total_count_vs_distal >= 6]
print(len(data2))

data3 = pd.read_csv('simple_general_cuts.csv',sep=',')
print(len(data3))
data3 = data3.ix[data3.library == 22]
data3 = data3.ix[(data3.seq.str.slice(50, 56) == 'AATAAA') | (data3.seq.str.slice(50, 56) == 'ATTAAA')]
data3 = data3.ix[(data3.seq.str.slice(101, 107) == 'AATAAA') | (data3.seq.str.slice(101, 107) == 'ATTAAA')]
data3 = data3.ix[data3.total_count >= 4]
print(len(data3))

data4 = pd.read_csv('apasix_general_cuts.csv',sep=',')
print(len(data4))

data4 = data4.ix[
	(((((((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 30)) |
	((data4.seq.str.slice(50, 56) == 'CATAAA') & (data4.library == 31))) |
	((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 32))) |
	((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 33))) |
	((data4.seq.str.slice(50, 56) == 'AATAAT') & (data4.library == 34))) |
	((data4.seq.str.slice(50, 56) == 'AATAAA') & (data4.library == 35)))
]

data4 = data4.ix[data4.total_count >= 10]
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


print('Re-aligning on realized pPAS.')

#Re-Align on PAS

pas_1 = 'AATAAA'
pas_2 = 'ATTAAA'

cano_pas1 = 'AATAAA'
cano_pas2 = 'ATTAAA'

pas_mutex1_1 = {}
pas_mutex1_2 = {}

pas_mutex2_1 = {}

for pos in range(0, 6) :
	for base in ['A', 'C', 'G', 'T'] :
		if cano_pas1[:pos] + base + cano_pas1[pos+1:] not in pas_mutex1_1 :
			pas_mutex1_1[cano_pas1[:pos] + base + cano_pas1[pos+1:]] = True
		if cano_pas2[:pos] + base + cano_pas2[pos+1:] not in pas_mutex1_2 :
			pas_mutex1_2[cano_pas2[:pos] + base + cano_pas2[pos+1:]] = True

for pos1 in range(0, 6) :
	for pos2 in range(pos1 + 1, 6) :
		for base1 in ['A', 'C', 'G', 'T'] :
			for base2 in ['A', 'C', 'G', 'T'] :
				if cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:] not in pas_mutex2_1 :
					pas_mutex2_1[cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:]] = True

align_dict = {
	20 : True
}

n_data2_aligned = { 0 : 0 }
n_data2 = { 0 : 0 }

def align_on_pas(row) :
	align_up = 3
	align_down = 3
	
	align_index = 50
	new_align_index = 50
	
	align_score = 0
	
	if row['library'] not in align_dict :
		return row['seq']
	
	n_data2[0] += 1

	if row['seq'][align_index:align_index+6] == cano_pas1 :
		return row['seq']

	if row['seq'][align_index:align_index+6] == cano_pas2 :
		return row['seq']

	for j in range(align_index - align_up, align_index + align_up + 1) :
		candidate_pas = row['seq'][j:j+6]
		
		if candidate_pas == cano_pas1 :
			new_align_index = j
			align_score = 4
		elif candidate_pas == cano_pas2 and align_score < 3 :
			new_align_index = j
			align_score = 3
		elif candidate_pas in pas_mutex1_1 and align_score < 2 :
			new_align_index = j
			align_score = 2
	
	seq_aligned = row['seq']
	if align_score > 0 and align_index != new_align_index :
		n_data2_aligned[0] += 1

		align_diff = int(new_align_index - align_index)
		
		if align_diff > 0 :
			seq_aligned = seq_aligned[align_diff:] + ('X' * align_diff)
		elif align_diff < 0 :
			align_diff = np.abs(align_diff)
			seq_aligned = ('X' * align_diff) + seq_aligned[:-align_diff]
		
		if len(seq_aligned) != 186 :
			print('ERROR')
			print(align_diff)
			print(row['seq'])
			print(seq_aligned)
	
	return seq_aligned

data2['seq'] = data2.apply(align_on_pas, axis=1)

print('n_data2 = ' + str(n_data2[0]))
print('n_data2_aligned = ' + str(n_data2_aligned[0]))
print('Done.')


filtered = '_general_cuts_antimisprime_orig_pasaligned'


action = '_seq'
#action = '_fullseq'
#action = '_fullseq_small'


total_count1 = list(data1.total_count)
cuts1 = list(data1.counts)
n1 = len(cuts1)

c1 = np.zeros((len(total_count1), 1))
c1[:, 0] = np.ravel(np.array(total_count1))

proximal_counts2 = list(data2.proximal_count)
total_counts_vs_distal2 = list(data2.total_count_vs_distal)
total_counts_vs_all2 = list(data2.total_count_vs_all)
n2 = len(proximal_counts2)

c2 = np.zeros((len(total_counts_vs_all2), 1))
c2[:, 0] = np.ravel(np.array(total_counts_vs_all2))

total_count3 = list(data3.total_count)
cuts3 = list(data3.counts)
n3 = len(cuts3)

c3 = np.zeros((len(total_count3), 1))
c3[:, 0] = np.ravel(np.array(total_count3))

total_count4 = list(data4.total_count)
cuts4 = list(data4.counts)
n4 = len(cuts4)

c4 = np.zeros((len(total_count4), 1))
c4[:, 0] = np.ravel(np.array(total_count4))



distalpas = np.concatenate([np.ones((len(c1), 1)), np.ones((len(c2), 1)), np.zeros((len(c3), 1)), np.zeros((len(c4), 1))], axis=0)

library1 = list(data1.library)
library2 = list(data2.library)
library3 = list(data3.library)
library4 = list(data4.library)

c = np.concatenate([c1, c2, c3, c4], axis=0)
L = np.zeros((len(c), 1))
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

print(c1.shape)
print(c2.shape)
print(c3.shape)
print(c4.shape)
print(c.shape)

np.save('apa' + filtered + '_count', c)
np.save('apa' + filtered + '_libindex', L)
np.save('apa' + filtered + '_distalpas', distalpas)



#Read cached cut matrix
#cut_distribution = spio.loadmat('apa' + filtered + '_cutdistribution.mat')['cuts']
#cut_distribution = sp.csr_matrix(cut_distribution)
#


cut_distribution = sp.lil_matrix((n1 + n2 + n3 + n4, 185 + 1))

compute_cut_distribution(cut_distribution, 0, cuts1)
spoof_cut_distribution_sym_prx(cut_distribution, n1, proximal_counts2, total_counts_vs_distal2, total_counts_vs_all2)
compute_cut_distribution(cut_distribution, n1 + n2, cuts3)
compute_cut_distribution(cut_distribution, n1 + n2 + n3, cuts4)




#Normalize
for i in range(0, cut_distribution.shape[0]) :
	if i % 100000 == 0:
		print("Normalized up to sequence: " + str(i))

	cut_distribution[i, :] = cut_distribution[i, :] / cut_distribution[i, :].sum(axis = 1)


cut_distribution = sp.csr_matrix(cut_distribution)
cut_dict = dict()
cut_dict["cuts"] = cut_distribution
spio.savemat('apa' + filtered + '_cutdistribution', cut_dict)


print('Aligned Cut Distribution.')


cuts1 = None
cuts3 = None
cuts4 = None
cut_distribution = None
cut_dict = None


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
	X_motif = sp.lil_matrix((len(c), 185 * 4), dtype=np.int8)
if action == '_fullseq' :
	X_motif = sp.lil_matrix((len(c), 255 * 4), dtype=np.int8)
if action == '_fullseq_small' :
	X_motif = sp.lil_matrix((len(c), 99 * 4), dtype=np.int8)

F = sp.lil_matrix((len(c), 6), dtype=np.int8)

cano_pas = 'AATAAA'
pas_mutex1 = []
for pos in range(0, 6) :
	for base in ['A', 'C', 'G', 'T'] :
		pas_mutex1.append(cano_pas[:pos] + base + cano_pas[1+pos:])

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
seqs_left = len(c1)

for i in range(0, len(c1)):
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
seqs_left = len(c2)

for i in range(0, len(c2)):
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
seqs_left = len(c3)

for i in range(0, len(c3)):
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
seqs_left = len(c4)

for i in range(0, len(c4)):
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