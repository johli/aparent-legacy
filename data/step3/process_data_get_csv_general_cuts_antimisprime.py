import pandas as pd
import scipy
import numpy
import scipy.sparse as sp
import scipy.io as spio

import regex as re


data1 = pd.read_csv('apa_nextseq_v2_library_merged2_20161001_general_cuts.csv',sep=',')
print(len(data1))
data1 = data1.ix[data1.total_count >= 12]
print(len(data1))

data2 = pd.read_csv('apa_sym_prx_library_20160916_general.csv',sep=',')
print(len(data2))
data2 = data2.ix[data2.library == 20]
data2 = data2.ix[data2.total_count_vs_distal >= 6]
print(len(data2))
data2['total_count'] = data2['total_count_vs_all']

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


filtered = '_general_cuts_antimisprime_orig'


#Read cached cut matrix
cut_distribution = spio.loadmat('apa' + filtered + '_cutdistribution.mat')['cuts']
cut_distribution = sp.csr_matrix(cut_distribution)

c = numpy.ravel(numpy.load('apa' + filtered + '_count.npy'))
#


data = pd.concat([data1, data2, data3, data4], ignore_index=True)

if len(data) != cut_distribution.shape[0] :
	print('ERROR! Cut matrix not synced with Dataframe!')

cut_distribution_sum = numpy.ravel(cut_distribution.sum(axis=1)) * c
data_sum = numpy.ravel(numpy.array(list(data.total_count)))
if cut_distribution_sum[1000] != data_sum[1000] :
	print('ERROR 1! Cut matrix not synced with Dataframe!')
	print(str(cut_distribution_sum[1000]) + ' ' + str(data_sum[1000]))
if cut_distribution_sum[100000] != data_sum[100000] :
	print('ERROR 2! Cut matrix not synced with Dataframe!')
	print(str(cut_distribution_sum[100000]) + ' ' + str(data_sum[100000]))
if cut_distribution_sum[1000000] != data_sum[1000000] :
	print('ERROR 3! Cut matrix not synced with Dataframe!')
	print(str(cut_distribution_sum[1000000]) + ' ' + str(data_sum[1000000]))
if cut_distribution_sum[3000000] != data_sum[3000000] :
	print('ERROR 4! Cut matrix not synced with Dataframe!')
	print(str(cut_distribution_sum[3000000]) + ' ' + str(data_sum[3000000]))


#Append library names
library_dict = {
	2 : 'TOMM5_UPN20WT20_DN_WT20',
	5 : 'TOMM5_UPWT20N20_DN_WT20',
	8 : 'TOMM5_UPN20WT20_DN_N20',
	11 : 'TOMM5_UPWT20N20_DN_N20',
	12 : 'TOMM5_OUTSIDE_MUTATION',
	20 : 'DoubleDope',
	22 : 'Simple',
	30 : 'AARS',
	31 : 'ATR',
	32 : 'HSPE1',
	33 : 'SNHG6',
	34 : 'SOX13',
	35 : 'WHAMMP2'
}

def map_name(library) :
	if library in library_dict :
		return library_dict[library]
	else :
		return 'UNKNOWN'

data['library_name'] = data['library'].apply(lambda x: map_name(x))

data = data[['seq', 'seq_ext', 'total_count', 'library', 'library_name']]


data.to_csv('apa_general_cuts_antimisprime.csv', header=True, index=False, sep=',')
