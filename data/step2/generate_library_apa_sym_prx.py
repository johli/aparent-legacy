import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

import regex as re

library_name = 'apa_sym_prx_library_20160916'
data = pd.read_csv(library_name + '.csv',sep=',')

data = data.ix[~data.sequence.str.contains('N')]
data = data.ix[data.total_count >= 4]
data = data.ix[~data.sequence.str.slice(51, 119).str.contains('AAAA')]
data = data.ix[~data.sequence.str.slice(136, 171).str.contains('AAAA')]

end_padding = 5

proximal_upstream_counts = list(data.proximal1_count)
proximal_downstream_counts = list(data.proximal2_count)
distal_counts = list(data.distal_count)
counts = list(data.total_count)

up_start = 'ATCCA'
up_end = 'CAGCC'
down_start = 'AAGCC'
down_end = 'CTACG'

degenerate_fullseq = (data.sequence.str.slice(0, 171)).values

degenerate_region_1 = data.sequence.str.slice(15,86)
degenerate_region_1 = (degenerate_region_1).values
	
degenerate_region_2 = data.sequence.str.slice(100,171)
degenerate_region_2 = (degenerate_region_2).values

emitted_proximal_counts = []
emitted_total_counts_vs_distal = []
emitted_total_counts_vs_all = []
emitted_seq = []
emitted_seq_ext = []
emitted_var_seq_ext = []
emitted_fullvar_seq_ext = []
emitted_L_index = []

up_constant = 'xxxxxxxxxxxxtgatagtaaggcccattacctgcctctttccctacacgacgctcttccgatctxxxxxxxxxxxxxxxxxxxx'.upper()
dn_constant = 'ctacgaactcccagcgcagaacacagcggttcgactgtgccttctagttgccagccatctgttgtttgcccctcccccgtgcctt'.upper()


for i in range(0, len(degenerate_fullseq)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	degenerate_seq_1 = degenerate_region_1[i]
	degenerate_seq_2 = degenerate_region_2[i]

	full_seq = degenerate_fullseq[i]
	
	#Pad with constant end regions
	degenerate_seq_1 = up_start[len(up_start)-end_padding:] + degenerate_seq_1 + up_end[:end_padding]
	degenerate_seq_2 = down_start[len(down_start)-end_padding:] + degenerate_seq_2 #+ down_end[:end_padding]
	
	fullseq_v = ('X' * (15 - end_padding)) + degenerate_seq_1 + ('X' * (14 - 2 * end_padding)) + degenerate_seq_2
	fullseq_fullv = ('X' * (15 - end_padding)) + up_start[len(up_start)-end_padding:] + full_seq[15:]

	#Emit 5' Proximal PAS variant
	
	emitted_seq.append(up_constant[-10:] + full_seq + dn_constant[:5])
	emitted_seq_ext.append(up_constant + full_seq)
	emitted_var_seq_ext.append('')
	emitted_fullvar_seq_ext.append('')
	emitted_L_index.append(20)
	
	emitted_proximal_counts.append(proximal_upstream_counts[i])
	emitted_total_counts_vs_distal.append(proximal_upstream_counts[i] + distal_counts[i])
	emitted_total_counts_vs_all.append(proximal_upstream_counts[i] + proximal_downstream_counts[i] + distal_counts[i])
	
	
	#Emit 3' Proximal PAS variant
	
	emitted_seq.append(up_constant[-10:] + full_seq + dn_constant[:5])
	emitted_seq_ext.append(full_seq + dn_constant)
	emitted_var_seq_ext.append('')
	emitted_fullvar_seq_ext.append('')
	emitted_L_index.append(21)
	
	emitted_proximal_counts.append(proximal_downstream_counts[i])
	emitted_total_counts_vs_distal.append(proximal_downstream_counts[i] + distal_counts[i])
	emitted_total_counts_vs_all.append(proximal_upstream_counts[i] + proximal_downstream_counts[i] + distal_counts[i])
	


df = pd.DataFrame({'seq'  : emitted_seq,
					'seq_ext'  : emitted_seq_ext,
					'var_seq_ext'  : emitted_var_seq_ext,
					'fullvar_seq_ext'  : emitted_fullvar_seq_ext,
					'proximal_count'  : emitted_proximal_counts,
					'total_count_vs_distal'  : emitted_total_counts_vs_distal,
					'total_count_vs_all'  : emitted_total_counts_vs_all,
					'library'  : emitted_L_index
				})

df = df[['seq', 'seq_ext', 'var_seq_ext', 'fullvar_seq_ext', 'proximal_count', 'total_count_vs_distal', 'total_count_vs_all', 'library']]

df = df.sort_values(by='total_count_vs_all')

df.to_csv(library_name + '_general.csv', header=True, index=False, sep=',')