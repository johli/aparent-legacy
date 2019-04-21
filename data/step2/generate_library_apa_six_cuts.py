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


n25al = (data.N25a.str.slice(0,25)).values
n25bl = (data.N25b.str.slice(0,25)).values

h50l = (data.h50.str.slice(0,50)).values
	

emitted_total_count = []
emitted_seq = []
emitted_seq_ext = []
emitted_L_index = []
emitted_counts = []


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


	cut_str = cuts[i]
	cut_str = cut_str[1:len(cut_str)-1]
	cuts_member = cut_str.split(', ')
	
	aligned_counts = []

	for j in range(0, len(cuts_member)) :
		cutpos = int(cuts_member[j])

		if cutpos >= 144 :
			aligned_counts.append(186)
		else :
			aligned_counts.append(cutpos + 2 + up_alignment)



	if utr_index == 5 and h50[12:12+6] != 'AATAAA' : #To account for wha sequencing error on PAS
		emitted_L_index.append(29)
	else :
		emitted_L_index.append(30 + utr_index)
	

	emitted_total_count.append(len(cuts_member))
	emitted_counts.append(str(aligned_counts))

	
	


df = pd.DataFrame({'seq'  : emitted_seq,
					'seq_ext'  : emitted_seq_ext,
					'total_count'  : emitted_total_count,
					'library'  : emitted_L_index,
					'counts' : emitted_counts
				})

df = df[['seq', 'seq_ext', 'total_count', 'library', 'counts']]

df = df.sort_values(by='total_count')

df.to_csv(library_name + '_general_cuts.csv', header=True, index=False, sep=',')
