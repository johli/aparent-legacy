import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

library_name = 'simple'

data = pd.read_csv(library_name + '.csv',sep=',')

cuts = list(data.cut)


proximal_upstream_interval = [54, 80]
proximal_downstream_interval = [105, 131]
distal_interval = [154, 154]

proximal_upstream_counts = []
proximal_downstream_counts = []
distal_counts = []
counts = []

proximal_upstream_avgcuts = []
proximal_upstream_stdcuts = []
proximal_downstream_avgcuts = []
proximal_downstream_stdcuts = []

for i in range(0, len(cuts)) :
	if i % 10000 == 0 :
		print('Accumulating cuts for member: ' + str(i))

	cut_str = cuts[i]
	cut_str = cut_str[1:len(cut_str)-1]
	cuts_member = cut_str.split(', ')
	
	
	proximal_upstream_count = 0
	proximal_downstream_count = 0
	distal_count = 0
	count = 0
	
	proximal_upstream_cuts = []
	proximal_downstream_cuts = []
	
	
	for j in range(0, len(cuts_member)) :
		cutpos = int(cuts_member[j])
		
		if cutpos >= proximal_upstream_interval[0] and cutpos <= proximal_upstream_interval[1] :
			proximal_upstream_count += 1
			proximal_upstream_cuts.append(float(cutpos))
		elif cutpos >= proximal_downstream_interval[0] and cutpos <= proximal_downstream_interval[1] :
			proximal_downstream_count += 1
			proximal_downstream_cuts.append(float(cutpos))
		elif cutpos >= distal_interval[0] and cutpos <= distal_interval[1] :
			distal_count += 1
		count += 1
	
	proximal_upstream_cuts = np.array(proximal_upstream_cuts)
	proximal_downstream_cuts = np.array(proximal_downstream_cuts)
	
	proximal_upstream_avgcut = 0
	proximal_upstream_stdcut = 0
	if proximal_upstream_count > 0 :
		proximal_upstream_avgcut = round(np.mean(proximal_upstream_cuts), 2)
		proximal_upstream_stdcut = round(np.std(proximal_upstream_cuts), 2)
	proximal_downstream_avgcut = 0
	proximal_downstream_stdcut = 0
	if proximal_downstream_count > 0 :
		proximal_downstream_avgcut = round(np.mean(proximal_downstream_cuts), 2)
		proximal_downstream_stdcut = round(np.std(proximal_downstream_cuts), 2)
	
	proximal_upstream_counts.append(proximal_upstream_count)
	proximal_downstream_counts.append(proximal_downstream_count)
	distal_counts.append(distal_count)
	counts.append(count)
	
	proximal_upstream_avgcuts.append(proximal_upstream_avgcut)
	proximal_upstream_stdcuts.append(proximal_upstream_stdcut)
	proximal_downstream_avgcuts.append(proximal_downstream_avgcut)
	proximal_downstream_stdcuts.append(proximal_downstream_stdcut)



n45al = (data.N45a.str.slice(0,45)).values
n45bl = (data.N45b.str.slice(0,45)).values
n45cl = (data.N45c.str.slice(0,45)).values

pPasl = (data.pPAS.str.slice(0,6)).values
dPasl = (data.dPAS.str.slice(0,6)).values



emitted_proximal_counts = []
emitted_total_counts_vs_distal = []
emitted_total_counts_vs_all = []
emitted_seq = []
emitted_seq_ext = []
emitted_var_seq_ext = []
emitted_fullvar_seq_ext = []
emitted_L_index = []
emitted_proximal_avgcut = []
emitted_proximal_stdcut = []

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

	#Emit 5' Proximal PAS variant
	
	#seq length 186
	emitted_seq.append(up_constant[-5:] + full_seq + dn_constant[:34])
	#seq length 256
	emitted_seq_ext.append(up_constant[-80:] + full_seq + dn_constant[:29])
	emitted_var_seq_ext.append('')
	emitted_fullvar_seq_ext.append('')
	emitted_L_index.append(22)
	
	emitted_proximal_counts.append(proximal_upstream_counts[i])
	emitted_total_counts_vs_distal.append(proximal_upstream_counts[i] + distal_counts[i])
	emitted_total_counts_vs_all.append(counts[i])
	emitted_proximal_avgcut.append(proximal_upstream_avgcuts[i])
	emitted_proximal_stdcut.append(proximal_upstream_stdcuts[i])
	
	
	#Emit 3' Proximal PAS variant
	
	#seq length 186
	emitted_seq.append(up_constant[-5:] + full_seq + dn_constant[:34])
	#seq length 256
	emitted_seq_ext.append(up_constant[-29:] + full_seq + dn_constant[:80])
	emitted_var_seq_ext.append('')
	emitted_fullvar_seq_ext.append('')
	emitted_L_index.append(23)
	
	emitted_proximal_counts.append(proximal_downstream_counts[i])
	emitted_total_counts_vs_distal.append(proximal_downstream_counts[i] + distal_counts[i])
	emitted_total_counts_vs_all.append(counts[i])
	emitted_proximal_avgcut.append(proximal_downstream_avgcuts[i])
	emitted_proximal_stdcut.append(proximal_downstream_stdcuts[i])
	
	


df = pd.DataFrame({'seq'  : emitted_seq,
					'seq_ext'  : emitted_seq_ext,
					'var_seq_ext'  : emitted_var_seq_ext,
					'fullvar_seq_ext'  : emitted_fullvar_seq_ext,
					'proximal_count'  : emitted_proximal_counts,
					'total_count_vs_distal'  : emitted_total_counts_vs_distal,
					'total_count_vs_all'  : emitted_total_counts_vs_all,
					'proximal_avgcut' : emitted_proximal_avgcut,
					'proximal_stdcut' : emitted_proximal_stdcut,
					'library'  : emitted_L_index
				})

df = df[['seq', 'seq_ext', 'var_seq_ext', 'fullvar_seq_ext', 'proximal_count', 'total_count_vs_distal', 'total_count_vs_all', 'proximal_avgcut', 'proximal_stdcut', 'library']]

df = df.sort_values(by='total_count_vs_all')

df.to_csv(library_name + '_general.csv', header=True, index=False, sep=',')
