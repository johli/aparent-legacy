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


emitted_proximal_counts = []
emitted_total_counts_vs_distal = []
emitted_total_counts_vs_all = []
emitted_seq = []
emitted_seq_ext = []
emitted_var_seq_ext = []
emitted_fullvar_seq_ext = []
emitted_count_string = []
emitted_L_index = []
emitted_proximal_avgcut = []
emitted_proximal_stdcut = []

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
	
	
	lib_index = -1
	up_index = [0, 40]
	dn_index = [0, 20]
	for j in range(0, len(libs)) :
		up_regex = libs[j][1]
		dn_regex = libs[j][2]
		
		if re.search(up_regex, degenerate_seq_1) and re.search(dn_regex, degenerate_seq_2) :
			lib_index = j
			break
	
	
	degenerate_seq_1_pad = std_up_start[len(up_start)-end_padding:] + degenerate_seq_1 + std_up_end[:end_padding]
	degenerate_seq_2_pad = std_down_start[len(down_start)-end_padding:] + degenerate_seq_2 + std_down_end[:end_padding]
	
	if lib_index != -1 :
		up_index[0] = libs[lib_index][3][0][0]
		up_index[1] = libs[lib_index][3][0][1]
		dn_index[0] = libs[lib_index][3][1][0]
		dn_index[1] = libs[lib_index][3][1][1]
	
	up_index[1] = up_index[1] + 2 * end_padding
	if dn_index[1] != 0 :
		dn_index[1] = dn_index[1] + 2 * end_padding
	
	degenerate_seq_1_pad_v = ('X' * up_index[0]) + degenerate_seq_1_pad[up_index[0]:up_index[1]] + ('X' * (40 + 2 * end_padding - up_index[1]))
	degenerate_seq_2_pad_v = ('X' * dn_index[0]) + degenerate_seq_2_pad[dn_index[0]:dn_index[1]] + ('X' * (20 + 2 * end_padding - dn_index[1]))
	
	
	if re.search(misprime_regex, degenerate_seq_2) :
		continue
	
	pas_index = -1
	outcome = re.search(pas1_regex, degenerate_seq_1)
	if pas_index == -1 and outcome != None :
		pas_index = outcome.start()
	
	outcome = re.search(pas2_regex, degenerate_seq_1)
	if pas_index == -1 and outcome != None :
		pas_index = outcome.start()
	
	outcome = re.search(pas1_mut1_regex, degenerate_seq_1)
	if pas_index == -1 and outcome != None :
		pas_index = outcome.start()
	
	if pas_index != -1 :
		cut_start = -1
		cut_end = -1
		for cut_interval in pas_interval_to_cut_interval :
			if pas_index >= cut_interval[0] and pas_index < cut_interval[1] :
				cut_start = cut_interval[2]
				cut_end = cut_interval[3]
				break
		
		c_prox = C_mat[i, cut_start:cut_end+1].sum(axis = 1)
		c_vs_all = C_mat[i, 0] + C_mat[i, std_cut_interval[0]:std_cut_interval[1]+1].sum(axis = 1) + C_mat[i, new_down_cut_interval[0]:new_down_cut_interval[1]+1].sum(axis = 1)
		c_dist = C_mat[i, 0]
	
		emitted_proximal_counts.append(int(c_prox))
		emitted_total_counts_vs_distal.append(int(c_prox + c_dist))
		emitted_total_counts_vs_all.append(int(c_prox + c_vs_all))

		proximal_cuts = []
		for cut_pos in range(cut_start, cut_end+1) :
			if C_mat[i, cut_pos] > 0 :
				for cut_pos_rep in range(0, int(C_mat[i, cut_pos])) :
					proximal_cuts.append(cut_pos)
		proximal_cuts = np.array(proximal_cuts)
	
		proximal_avgcut = 0
		proximal_stdcut = 0
		if c_prox > 0 :
			proximal_avgcut = round(np.mean(proximal_cuts), 2)
			proximal_stdcut = round(np.std(proximal_cuts), 2)
		
		emitted_proximal_avgcut.append(proximal_avgcut)
		emitted_proximal_stdcut.append(proximal_stdcut)

		
		new_full_seq = (up_constant[len(up_constant)-46-5+pas_index:] + full_seq + dn_constant)[0:186]
		new_full_seq_ext = (up_constant[pas_index:] + full_seq + dn_constant)[0:256]
		fullvar_full_seq_ext = (('X' * (len(up_constant[pas_index:]) - 5)) + std_up_start[len(up_start)-end_padding:] + full_seq + std_down_end[:end_padding] + ('X' * 83))[0:256]
		var_full_seq_ext = ''
		if pas_seq != 'CTAAAATATAAAACT' :
			var_full_seq_ext = (('X' * (len(up_constant[pas_index:]) - 5)) + degenerate_seq_1_pad_v[0:45] + pas_seq + ('X' * 13) + degenerate_seq_2_pad_v + ('X' * 83))[0:256]
		else :
			var_full_seq_ext = (('X' * (len(up_constant[pas_index:]) - 5)) + degenerate_seq_1_pad_v + ('X' * 23) + degenerate_seq_2_pad_v + ('X' * 83))[0:256]
		
		emitted_seq.append(new_full_seq)
		emitted_seq_ext.append(new_full_seq_ext)
		emitted_var_seq_ext.append('')
		emitted_fullvar_seq_ext.append('')
		
		if pas_seq != 'CTAAAATATAAAACT' :
			emitted_L_index.append(14)
		else :
			emitted_L_index.append(13)
		
		count_string = ''
		C_vec = np.ravel(C_mat[i,:].todense())
		for j in range(0, len(C_vec)) :
			c_pos = C_vec[j]
			if c_pos > 0 :
				count_string += str(j) + ': ' + str(int(c_pos)) + ' - '
		if len(count_string) > 0 :
			count_string = count_string[:len(count_string) - 3]
		emitted_count_string.append(count_string)

		
	c_prox = C_mat[i, std_cut_interval[0]:std_cut_interval[1]+1].sum(axis = 1)
	c_vs_all = C_mat[i, 0] + C_mat[i, new_cut_interval[0]:new_cut_interval[1]+1].sum(axis = 1) + C_mat[i, new_down_cut_interval[0]:new_down_cut_interval[1]+1].sum(axis = 1)
	c_dist = C_mat[i, 0]
		
	emitted_proximal_counts.append(int(c_prox))
	emitted_total_counts_vs_distal.append(int(c_prox + c_dist))
	emitted_total_counts_vs_all.append(int(c_prox + c_vs_all))

	proximal_cuts = []
	for cut_pos in range(std_cut_interval[0], std_cut_interval[1]+1) :
		if C_mat[i, cut_pos] > 0 :
			for cut_pos_rep in range(0, int(C_mat[i, cut_pos])) :
				proximal_cuts.append(cut_pos)
	proximal_cuts = np.array(proximal_cuts)
	
	proximal_avgcut = 0
	proximal_stdcut = 0
	if c_prox > 0 :
		proximal_avgcut = round(np.mean(proximal_cuts), 2)
		proximal_stdcut = round(np.std(proximal_cuts), 2)
	
	emitted_proximal_avgcut.append(proximal_avgcut)
	emitted_proximal_stdcut.append(proximal_stdcut)
	
	new_full_seq = up_constant[len(up_constant)-5:] + full_seq + dn_constant[0:88]
	new_full_seq_ext = up_constant[len(up_constant)-80:] + full_seq + dn_constant[0:83]
	fullvar_full_seq_ext = ('X' * 75) + std_up_start[len(up_start)-end_padding:] + full_seq + std_down_end[:end_padding] + ('X' * 78)
	var_full_seq_ext = ''
	if pas_seq != 'CTAAAATATAAAACT' :
		var_full_seq_ext = ('X' * 75) + degenerate_seq_1_pad_v[0:45] + pas_seq + ('X' * 13) + degenerate_seq_2_pad_v + ('X' * 78)
	else :
		var_full_seq_ext = ('X' * 75) + degenerate_seq_1_pad_v + ('X' * 23) + degenerate_seq_2_pad_v + ('X' * 78)
	
	emitted_seq.append(new_full_seq)
	emitted_seq_ext.append(new_full_seq_ext)
	emitted_var_seq_ext.append('')
	emitted_fullvar_seq_ext.append('')
	
	if pas_seq != 'CTAAAATATAAAACT' :
		emitted_L_index.append(12)
	else :
		emitted_L_index.append(lib_index)
	
	count_string = ''
	C_vec = np.ravel(C_mat[i,:].todense())
	for j in range(0, len(C_vec)) :
		c_pos = C_vec[j]
		if c_pos > 0 :
			count_string += str(j) + ': ' + str(int(c_pos)) + ' - '
	if len(count_string) > 0 :
		count_string = count_string[:len(count_string) - 3]
	emitted_count_string.append(count_string)

df = pd.DataFrame({'seq'  : emitted_seq,
					'seq_ext'  : emitted_seq_ext,
					'var_seq_ext'  : emitted_var_seq_ext,
					'fullvar_seq_ext'  : emitted_fullvar_seq_ext,
					'proximal_count'  : emitted_proximal_counts,
					'total_count_vs_distal'  : emitted_total_counts_vs_distal,
					'total_count_vs_all'  : emitted_total_counts_vs_all,
					'proximal_avgcut' : emitted_proximal_avgcut,
					'proximal_stdcut' : emitted_proximal_stdcut,
					'library'  : emitted_L_index,
					'pos_counts' : emitted_count_string
				})

df = df[['seq', 'seq_ext', 'var_seq_ext', 'fullvar_seq_ext', 'proximal_count', 'total_count_vs_distal', 'total_count_vs_all', 'proximal_avgcut', 'proximal_stdcut', 'library', 'pos_counts']]

df = df.sort_values(by='total_count_vs_all')

df.to_csv(library_name + '_general.csv', header=True, index=False, sep=',')