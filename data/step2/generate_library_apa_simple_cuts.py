import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

library_name = 'simple'

data = pd.read_csv(library_name + '.csv',sep=',')

cuts = list(data.cut)


n45al = (data.N45a.str.slice(0,45)).values
n45bl = (data.N45b.str.slice(0,45)).values
n45cl = (data.N45c.str.slice(0,45)).values

pPasl = (data.pPAS.str.slice(0,6)).values
dPasl = (data.dPAS.str.slice(0,6)).values


emitted_total_count = []
emitted_seq = []
emitted_seq_ext = []
emitted_L_index = []
emitted_counts = []

up_constant = 'ACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTCTTGATCCCTACACGACGCTCTTCCGATCT'
dn_constant = 'CGCCTAACCCTAAGCAGATTCTTCATGCAATTGTCGGTCAAGCCTTGCCTTGTTGTAGCTTAAATTTTGCTCGCGCACTA'


for i in range(0, len(n45al)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	n45a = n45al[i]
	n45b = n45bl[i]
	n45c = n45cl[i]

	pPas = pPasl[i]
	dPas = dPasl[i]
	
	full_seq = n45a + pPas + n45b + dPas + n45c

	cut_str = cuts[i]
	cut_str = cut_str[1:len(cut_str)-1]
	cuts_member = cut_str.split(', ')
	
	aligned_counts = []

	for j in range(0, len(cuts_member)) :
		cutpos = int(cuts_member[j])

		if cutpos == 154 :
			aligned_counts.append(186)
		else :
			aligned_counts.append(cutpos + 5)


	#Emit 5' Proximal PAS variant
	
	#seq length 186
	emitted_seq.append(up_constant[-5:] + full_seq + dn_constant[:34])
	#seq length 256
	emitted_seq_ext.append(up_constant[-80:] + full_seq + dn_constant[:29])
	emitted_L_index.append(22)
	
	emitted_total_count.append(len(cuts_member))

	emitted_counts.append(str(aligned_counts))
	
	
	#Emit 3' Proximal PAS variant
	
	#seq length 186
	'''emitted_seq.append(up_constant[-5:] + full_seq + dn_constant[:34])
	#seq length 256
	emitted_seq_ext.append(up_constant[-29:] + full_seq + dn_constant[:80])
	emitted_var_seq_ext.append('')
	emitted_fullvar_seq_ext.append('')
	emitted_L_index.append(23)
	
	emitted_proximal_counts.append(proximal_downstream_counts[i])
	emitted_total_counts_vs_distal.append(proximal_downstream_counts[i] + distal_counts[i])
	emitted_total_counts_vs_all.append(counts[i])
	emitted_proximal_avgcut.append(proximal_downstream_avgcuts[i])
	emitted_proximal_stdcut.append(proximal_downstream_stdcuts[i])'''
	
	


df = pd.DataFrame({'seq'  : emitted_seq,
					'seq_ext'  : emitted_seq_ext,
					'total_count'  : emitted_total_count,
					'library'  : emitted_L_index,
					'counts'  : emitted_counts
				})

df = df[['seq', 'seq_ext', 'total_count', 'library', 'counts']]

df = df.sort_values(by='total_count')


print(df.head())
print(df.tail())


df.to_csv(library_name + '_general_cuts.csv', header=True, index=False, sep=',')
