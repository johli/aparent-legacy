import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

library_name = 'apasix'

data = pd.read_csv(library_name + '.csv',sep=',')

utrs = list(data.utr)
cuts = list(data.cuts)


utr_dict = {}
utr_dict['aar'] = 0
utr_dict['atr'] = 1
utr_dict['hsp'] = 2
utr_dict['snh'] = 3
utr_dict['sox'] = 4
utr_dict['wha'] = 5

proximal_interval = [
	[[54, 69]],#aar
	[[50, 71]],#atr
	[[53, 70]],#hsp
	[[55, 76]],#snh
	[[54, 68]],#sox
	[[53, 79]] #wha
]
distal_interval = [
	[145, 145],#aar
	[132, 144],#atr
	[121, 144],#hsp
	[110, 144],#snh
	[128, 144],#sox
	[110, 144] #wha
]


proximal_counts = []
distal_counts = []
counts = []

proximal_avgcuts = []
proximal_stdcuts = []

for i in range(0, len(cuts)) :
	if i % 10000 == 0 :
		print('Accumulating cuts for member: ' + str(i))

	cut_str = cuts[i]
	cut_str = cut_str[1:len(cut_str)-1]
	cuts_member = cut_str.split(', ')
	
	
	proximal_count = 0
	distal_count = 0
	count = 0
	
	proximal_cuts = []
	
	utr_index = utr_dict[utrs[i]]
	
	for j in range(0, len(cuts_member)) :
		cutpos = int(cuts_member[j])
		
		prox_cut = False
		for k in range(0, len(proximal_interval[utr_index])) :
			if cutpos >= proximal_interval[utr_index][k][0] and cutpos <= proximal_interval[utr_index][k][1] :
				proximal_count += 1
				proximal_cuts.append(float(cutpos))
				prox_cut = True
				break
		
		if prox_cut == False and cutpos >= distal_interval[utr_index][0] and cutpos <= distal_interval[utr_index][1] :
			distal_count += 1
		count += 1
	
	proximal_cuts = np.array(proximal_cuts)
	
	proximal_avgcut = 0
	proximal_stdcut = 0
	if proximal_count > 0 :
		proximal_avgcut = round(np.mean(proximal_cuts), 2)
		proximal_stdcut = round(np.std(proximal_cuts), 2)
	
	proximal_counts.append(proximal_count)
	distal_counts.append(distal_count)
	counts.append(count)
	
	proximal_avgcuts.append(proximal_avgcut)
	proximal_stdcuts.append(proximal_stdcut)




n25al = (data.N25a.str.slice(0,25)).values
n25bl = (data.N25b.str.slice(0,25)).values

h50l = (data.h50.str.slice(0,50)).values
	

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


up_alignmentl = [
	12,
	7,
	12,
	0,
	15,
	11
]

up_constantl = [
	'GAACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTCTTGATCCCTACACGACGCTCTTCCGATCTCTGAGCTTT',
	'AACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTCTTGATCCCTACACGACGCTCTTCCGATCTAATGCATTTG',
	'AACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTCTTGATCCCTACACGACGCTCTTCCGATCTTCTGAAATCT',
	'AACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTCTTGATCCCTACACGACGCTCTTCCGATCTAACATGAACA',
	'AACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTCTTGATCCCTACACGACGCTCTTCCGATCTTCTTTTTTTA',
	'AACAGTACGAACGCGCCGAGGGCCGCCACTCCACCGGCGGCATGGACGAGCTGTACAAGTCTTGATCCCTACACGACGCTCTTCCGATCTTGAATTTCAT'
]

dn_constantl = [
	'GGCTCTTTTGACAGCCTTTGGCGTCTGTAGAATAAATGCTGTGGCTCCTGCTGGCTGCTGTGGTGTTCACCTAGTCCAGCCCCAGAACCCGCTCGCGCAC',
	'AATATACATTCAGTTATTAAGAAATAAACTGCTTTCTTAATACATACTGTGCATTATAATTGGAGAAATAGAATATCATGCTCGCGCACTACTCAGCGAC',
	'CACTTCCAAATAAAAATATGTAAATGAGTGGTTAATCTTTAGTTATTTTAAGATGATTTTAGGGTTTTGCTCGCGCACTACTCAGCGACCTCCAACACAC',
	'TAAAGTGTTTTCTTTTAAATCAACTCTAAATAGCTCCATTCTCATAGTCACTAGTCAGACCGCTCGCGCACTACTCAGCGACCTCCAACACACAAGCAGG',
	'GTTTTCTGATGACATAATAAAGACAGATCATTTCAGAATCTGGCCCTTGTGCAGGGGAGGAGGGAGGCTGGCCTAAGCTCGCGCACTACTCAGCGACCTC',
	'AGTGCTCAATAAAAAGAATAAAGAGGAAACAGCACTGGATCTATACCTATACAAAACAAGCTACCAGCGCTCGCGCACTACTCAGCGACCTCCAACACAC'
]

t0 = 0
t1 = 0
t2 = 0
t3 = 0
t4 = 0
t5 = 0

for i in range(0, len(n25al)):
	if i % 100000 == 0:
		print("Read up to sequence: " + str(i))
	
	utr_index = utr_dict[utrs[i]]
	
	n25a = n25al[i]
	n25b = n25bl[i]

	h50 = h50l[i]
	
	full_seq = n25a + h50 + n25b

	up_alignment = up_alignmentl[utr_index]
	up_constant = up_constantl[utr_index]
	dn_constant = dn_constantl[utr_index]
	
	#Emit Proximal PAS variant
	
	#seq length 186
	emitted_seq.append((up_constant[-up_alignment-2:] + full_seq + dn_constant)[:186])
	#seq length 256
	emitted_seq_ext.append((up_constant[-up_alignment-2-75:] + full_seq + dn_constant)[:256])
	emitted_var_seq_ext.append('')
	emitted_fullvar_seq_ext.append('')

	if utr_index == 5 and h50[12:12+6] != 'AATAAA' :#To account for wha sequencing error on PAS
		emitted_L_index.append(29)
	else :
		emitted_L_index.append(30 + utr_index)
	
	emitted_proximal_counts.append(proximal_counts[i])
	emitted_total_counts_vs_distal.append(proximal_counts[i] + distal_counts[i])
	emitted_total_counts_vs_all.append(counts[i])
	emitted_proximal_avgcut.append(proximal_avgcuts[i])
	emitted_proximal_stdcut.append(proximal_stdcuts[i])
	
	


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

df.to_csv(library_name + '2_general.csv', header=True, index=False, sep=',')
