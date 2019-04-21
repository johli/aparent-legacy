import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

import regex as re
import pickle


library_name = 'apa_nextseq_v2_library_merged2_20161001'
data = pd.read_csv(library_name + '.csv',sep=',')

proximal_counts = list(data.proximal_count)
distal_counts = list(data.distal_count)
counts = list(data.total_count)

C_mat = spio.loadmat('apa_nextseq_v2_library_merged2_20161001.mat')['cuts']
C_mat = sp.csr_matrix(C_mat)

c_tot = np.ravel(C_mat[:, :].sum(axis = 1))
c_sort_index = np.argsort(c_tot)
C_mat = C_mat[c_sort_index,:]


pas1_regex = re.compile(r"(AATAAA)")
pas2_regex = re.compile(r"(ATTAAA)")
pas1_mut1_regex = re.compile(r"(AATAAA){s<=1}")
pas1_mut2_regex = re.compile(r"(AATAAA){s<=2}")

pas_interval_to_cut_interval = [[0, 20, 140, 154], [20, 40, 150, 167]]

new_cut_interval = [140, 167]
std_cut_interval = [168, 183]
new_down_cut_interval = [187, 200]


up_start = 'CTGCT'
up_end = 'CTAAA'
down_start = 'GAAAC'
down_end = 'ACCCT'

degenerate_region_1 = data.upstream.str.slice(0,40)
degenerate_region_1 = (degenerate_region_1).values
	
degenerate_region_2 = data.downstream.str.slice(0,20)
degenerate_region_2 = (degenerate_region_2).values

pas_region = data.pas.str.slice(0,15)
pas_region = (pas_region).values

full_region = data.seq.str.slice(0,93).values


emitted_total_count = []
emitted_seq = []
emitted_seq_ext = []
emitted_L_index = []
emitted_counts = []

up_constant = 'xxxxxxxxxxxxxxxxxxxxxxxxggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaaggactgatagtaaggcccattacctgcggccgcaattctgct'.upper()
dn_constant = 'acccttatccctgtgacgtttggcctctgacaatactggtataattgtaaataatgtcaaactccgttttctagcaagtattaagggagctgtgtctgaaat'.upper()

std_up_start = 'CTGCT'
std_up_end = 'CTAAA'
std_down_start = 'GAAAC'
std_down_end = 'ACCCT'


lib_up_variant_a = re.compile(r"([ACGTN]{20}CTGGTAACTGACCTTCAAAG){s<=3}")
lib_up_variant_b = re.compile(r"(TGTTAAGAAC[ACGTN]{20}ACCTTCAAAG){s<=3}")
lib_up_variant_c = re.compile(r"(TGTTAAGAACAAGTTTGGCT[ACGTN]{20}){s<=3}")
lib_up_variant_d = re.compile(r"(TGTTAAGAAC[ACGTN]{10}CTGGTAACTGACCTTCAAAG){s<=1}")
lib_up_variant_e = re.compile(r"(TGTTAAGAACAAGTT[ACGTN]{10}AACTGACCTTCAAAG){s<=1}")
lib_up_variant_f = re.compile(r"(TGTTAAGAACAAGTTTGGCT[ACGTN]{10}ACCTTCAAAG){s<=1}")

lib_dn_variant_a = re.compile(r"(GATGTCTCGTGATCTGGTGT){s<=1}")
lib_dn_variant_b = re.compile(r"....................")


libs = [
['up_e_dn_a_umi', lib_up_variant_e, lib_dn_variant_a, [[15, 25], [0, 0]], [], [], [], []],
['up_d_dn_a_umi', lib_up_variant_d, lib_dn_variant_a, [[10, 20], [0, 0]], [], [], [], []],
['up_a_dn_a_umi', lib_up_variant_a, lib_dn_variant_a, [[0, 20], [0, 0]], [], [], [], []],
['up_f_dn_a_umi', lib_up_variant_f, lib_dn_variant_a, [[20, 30], [0, 0]], [], [], [], []],
['up_b_dn_a_umi', lib_up_variant_b, lib_dn_variant_a, [[10, 30], [0, 0]], [], [], [], []],
['up_c_dn_a_umi', lib_up_variant_c, lib_dn_variant_a, [[20, 40], [0, 0]], [], [], [], []],
['up_e_dn_b_umi', lib_up_variant_e, lib_dn_variant_b, [[15, 25], [0, 20]], [], [], [], []],
['up_d_dn_b_umi', lib_up_variant_d, lib_dn_variant_b, [[10, 20], [0, 20]], [], [], [], []],
['up_a_dn_b_umi', lib_up_variant_a, lib_dn_variant_b, [[0, 20], [0, 20]], [], [], [], []],
['up_f_dn_b_umi', lib_up_variant_f, lib_dn_variant_b, [[20, 30], [0, 20]], [], [], [], []],
['up_b_dn_b_umi', lib_up_variant_b, lib_dn_variant_b, [[10, 30], [0, 20]], [], [], [], []],
['up_c_dn_b_umi', lib_up_variant_c, lib_dn_variant_b, [[20, 40], [0, 20]], [], [], [], []],
]


misprime_regex = re.compile(r"(AAAAAAAA){s<=2}")

end_padding = 5

for i in range(0, len(full_region)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	degenerate_seq_1 = degenerate_region_1[i]
	degenerate_seq_2 = degenerate_region_2[i]
	
	full_seq = full_region[i]
	
	pas_seq = pas_region[i]

	if re.search(misprime_regex, degenerate_seq_2) :
		continue
	
	
	lib_index = -1
	up_index = [0, 40]
	dn_index = [0, 20]
	for j in range(0, len(libs)) :
		up_regex = libs[j][1]
		dn_regex = libs[j][2]
		
		if re.search(up_regex, degenerate_seq_1) and re.search(dn_regex, degenerate_seq_2) :
			lib_index = j
			break


	c = C_mat[i, :].sum(axis = 1)
		
	emitted_total_count.append(int(c))

	aligned_cuts = []
	for cut_pos in range(130, C_mat.shape[1]) :
		if C_mat[i, cut_pos] > 0 :
			for cut_pos_rep in range(0, int(C_mat[i, cut_pos])) :
				aligned_cuts.append(cut_pos - 110)

	for cut_pos_rep in range(0, int(C_mat[i, 0])) :
		aligned_cuts.append(186)

	emitted_counts.append(str(aligned_cuts))
	
	
	new_full_seq = up_constant[len(up_constant)-5:] + full_seq + dn_constant[0:88]
	new_full_seq_ext = up_constant[len(up_constant)-80:] + full_seq + dn_constant[0:83]
	
	emitted_seq.append(new_full_seq)
	emitted_seq_ext.append(new_full_seq_ext)

	
	if pas_seq != 'CTAAAATATAAAACT' :
		emitted_L_index.append(12)
	else :
		emitted_L_index.append(lib_index)


df = pd.DataFrame({'seq'  : emitted_seq,
					'seq_ext'  : emitted_seq_ext,
					'total_count'  : emitted_total_count,
					'library'  : emitted_L_index,
					'counts' : emitted_counts
				})

df = df[['seq', 'seq_ext', 'total_count', 'library', 'counts']]

df = df.sort_values(by='total_count')

df.to_csv(library_name + '_general_cuts.csv', header=True, index=False, sep=',')