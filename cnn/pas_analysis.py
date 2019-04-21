import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.cm as cm


data = pd.read_csv('apa_human_v2.csv',sep=',')

print('Unfiltered data size = ' + str(len(data)))
#print(data.head())

data = data.ix[data.proximal_pas != -1]
data = data.ix[data.distal_pas != -1]

data = data.ix[data.proximal_cutsite_length <= 30]
data = data.ix[data.distal_cutsite_length <= 30]

data = data.ix[(data.proximal_type == 'UTR3') | (data.proximal_type == 'Extension')]
data = data.ix[(data.distal_type == 'UTR3') | (data.distal_type == 'Extension')]

#data = data.ix[data.num_sites <= 4]
data = data.ix[data.site_distance >= 100]
data = data.ix[data.site_distance <= 4000]

data = data.ix[data.total_count >= 50]#400,1000,2000

#data = data.ix[data.proximal_sitenum == 1]

data = data.ix[(data.distal_seq.str.slice(50, 50 + 6) == 'AATAAA') | (data.distal_seq.str.slice(50, 50 + 6) == 'ATTAAA')]

data = data.sort_values(by='total_count').reset_index(drop=True)
#data = data.sort_values(by='proximal_id', ascending=False).reset_index(drop=True)


print('Filtered data size = ' + str(len(data)))


data['prox_usage'] = data['proximal_count'] / (data['proximal_count'] + data['distal_count'])

apadb_pas = list(set(data.proximal_seq.str.slice(50, 50 + 6).values))

apadb_pas_dict = {}

for pas in apadb_pas :
	n_has_pas = len(np.ravel(data[data.proximal_seq.str.slice(50, 50 + 6) == pas].prox_usage))

	has_pas = np.mean(np.ravel(data[data.proximal_seq.str.slice(50, 50 + 6) == pas].prox_usage))
	has_not_pas = np.mean(np.ravel(data[data.proximal_seq.str.slice(50, 50 + 6) != pas].prox_usage))

	lor = np.log((has_pas / (1.0 - has_pas)) / (has_not_pas / (1.0 - has_not_pas)))
	
	if n_has_pas >= 50 :
		apadb_pas_dict[pas] = lor

print(apadb_pas_dict)

cnn_pas_dict = {
	'AATAAA': 0.84,
	'ATTAAA': -1.90,
	'AGTAAA': -3.00,
	'TATAAA': -3.82,
	'ACTAAA': -3.87,
	'GATAAA': -4.06,
	'CATAAA': -4.39,
	'AATGAA': -5.12,
	'AATAAT': -5.33,
	'TTTAAA': -5.52
}

logoddsratio_pas_dict = {
	'AATAAA': 3.1,
	'ATTAAA': 1.4,
	'AGTAAA': 0.1,
	'TATAAA': -0.1,
	'GATAAA': -0.4,
	'CATAAA': -0.4
}

zavalon_pas_dict = {
	'AATAAA': 8000,
	'ATTAAA': 2500,
	'AGTAAA': 500,
	'TATAAA': 500,
	'ACTAAA': 100,
	'GATAAA': 200,
	'CATAAA': 250,
	'AATGAA': 150,
	'AATAAT': 80
}

pas_mat = np.zeros((len(cnn_pas_dict), 3))
pas_list = []

i = 0
for pas in cnn_pas_dict :
	pas_list.append(pas)

	pas_mat[i, 0] = cnn_pas_dict[pas]

	if pas in apadb_pas_dict :
		pas_mat[i, 1] = apadb_pas_dict[pas]
	else :
		pas_mat[i, 1] = -np.inf

	if pas in zavalon_pas_dict :
		pas_mat[i, 2] = zavalon_pas_dict[pas]
	else :
		pas_mat[i, 2] = -np.inf

	i += 1

pas_list = np.array(pas_list)

pas_sort_index = np.argsort(np.ravel(pas_mat[:, 0]))
pas_list = pas_list[pas_sort_index]
pas_mat[:, :] = pas_mat[pas_sort_index, :]




fig = plt.figure(figsize=(2, 10))

pas_index_apadb = np.ravel(pas_mat[:, 1]) != -np.inf

plt.pcolor(pas_mat[pas_index_apadb, 1].reshape((pas_mat[pas_index_apadb, 1].shape[0], 1)), cmap=plt.get_cmap('Reds'), vmin=pas_mat[pas_index_apadb, 1].min(), vmax=pas_mat[pas_index_apadb, 1].max())
plt.xticks([0], [''])

y_pas_list = [str(round(b, 2)) + ' ' + a for a, b in zip(pas_list[pas_index_apadb], pas_mat[pas_index_apadb, 1].tolist())]
plt.yticks(np.arange(pas_mat[pas_index_apadb, 1].shape[0]) + 0.5, y_pas_list)
plt.ylabel('APADB PAS Logodds Ratio')

plt.axis([0, 1, 0, pas_mat[pas_index_apadb, 1].shape[0]])

plt.title('APADB sorted PAS bar')

plt.tight_layout()

plt.savefig("apadb_sorted_pas_bar.png")
plt.savefig("apadb_sorted_pas_bar.svg")
plt.savefig("apadb_sorted_pas_bar.eps")
plt.show()
plt.close()


f = plt.figure()

pas_index_apadb = np.ravel(pas_mat[:, 1]) != -np.inf

r, p_val = pearsonr(pas_mat[pas_index_apadb, 0], pas_mat[pas_index_apadb, 1])
r_square = r * r

plt.scatter(pas_mat[pas_index_apadb, 0], pas_mat[pas_index_apadb, 1], color='black')
plt.plot(pas_mat[pas_index_apadb, 0], pas_mat[pas_index_apadb, 1], color='black', linewidth=2, linestyle='--')

plt.plot([np.min(pas_mat[pas_index_apadb, 0]) * 1.1,np.max(pas_mat[pas_index_apadb, 0]) * 1.1], [np.min(pas_mat[pas_index_apadb, 1]) * 1.1,np.max(pas_mat[pas_index_apadb, 1]) * 1.1], color='darkblue')

x_pas_list = [str(round(b, 2)) + ' ' + a for a, b in zip(pas_list[pas_index_apadb], pas_mat[pas_index_apadb, 0].tolist())]
y_pas_list = [str(round(b, 2)) + ' ' + a for a, b in zip(pas_list[pas_index_apadb], pas_mat[pas_index_apadb, 1].tolist())]

plt.xticks(pas_mat[pas_index_apadb, 0], x_pas_list, rotation=45)
plt.yticks(pas_mat[pas_index_apadb, 1], y_pas_list, rotation=45)

plt.axis([np.min(pas_mat[pas_index_apadb, 0]) * 1.1, np.max(pas_mat[pas_index_apadb, 0]) * 1.1, np.min(pas_mat[pas_index_apadb, 1]) * 1.1, np.max(pas_mat[pas_index_apadb, 1]) * 1.1])

plt.xlabel('CNN PAS')
plt.ylabel('APADB PAS')

plt.title('PAS comparison (R^2 = ' + str(round(r_square, 2)) + ')')

plt.tight_layout()

plt.savefig("cnn_apadb_pas_comparison.png")
plt.savefig("cnn_apadb_pas_comparison.svg")
plt.savefig("cnn_apadb_pas_comparison.eps")
plt.show()





fig = plt.figure(figsize=(2, 10))

pas_index_apadb = np.ravel(pas_mat[:, 2]) != -np.inf

pas_mat[pas_index_apadb, 2] = np.log(pas_mat[pas_index_apadb, 2])

plt.pcolor(pas_mat[pas_index_apadb, 2].reshape((pas_mat[pas_index_apadb, 2].shape[0], 1)), cmap=plt.get_cmap('Reds'), vmin=pas_mat[pas_index_apadb, 2].min(), vmax=pas_mat[pas_index_apadb, 2].max())
plt.xticks([0], [''])

y_pas_list = [str(round(b, 2)) + ' ' + a for a, b in zip(pas_list[pas_index_apadb], pas_mat[pas_index_apadb, 2].tolist())]
plt.yticks(np.arange(pas_mat[pas_index_apadb, 2].shape[0]) + 0.5, y_pas_list)
plt.ylabel('Zavalon PAS Logodds Ratio')

plt.axis([0, 1, 0, pas_mat[pas_index_apadb, 2].shape[0]])

plt.title('Zavalon sorted PAS bar')

plt.tight_layout()

plt.savefig("zavalon_sorted_pas_bar.png")
plt.savefig("zavalon_sorted_pas_bar.svg")
plt.savefig("zavalon_sorted_pas_bar.eps")
plt.show()
plt.close()



f = plt.figure()


pas_index_apadb = np.ravel(pas_mat[:, 2]) != -np.inf

#pas_mat[pas_index_apadb, 2] = np.log(pas_mat[pas_index_apadb, 2])

r, p_val = pearsonr(pas_mat[pas_index_apadb, 0], pas_mat[pas_index_apadb, 2])
r_square = r * r

plt.scatter(pas_mat[pas_index_apadb, 0], pas_mat[pas_index_apadb, 2], color='black')
plt.plot(pas_mat[pas_index_apadb, 0], pas_mat[pas_index_apadb, 2], color='black', linewidth=2, linestyle='--')

plt.plot([np.min(pas_mat[pas_index_apadb, 0]) * 1.1,np.max(pas_mat[pas_index_apadb, 0]) * 1.1], [np.min(pas_mat[pas_index_apadb, 2]) * 1.1,np.max(pas_mat[pas_index_apadb, 2]) * 1.1], color='darkblue')
plt.axis([np.min(pas_mat[pas_index_apadb, 0]) * 1.1, np.max(pas_mat[pas_index_apadb, 0]) * 1.1, np.min(pas_mat[pas_index_apadb, 2]) * 1.1, np.max(pas_mat[pas_index_apadb, 2]) * 1.1])

x_pas_list = [str(round(b, 2)) + ' ' + a for a, b in zip(pas_list[pas_index_apadb], pas_mat[pas_index_apadb, 0].tolist())]
y_pas_list = [str(round(b, 2)) + ' ' + a for a, b in zip(pas_list[pas_index_apadb], pas_mat[pas_index_apadb, 2].tolist())]

plt.xticks(pas_mat[pas_index_apadb, 0], x_pas_list, rotation=45)
plt.yticks(pas_mat[pas_index_apadb, 2], y_pas_list, rotation=45)

plt.xlabel('CNN PAS')
plt.ylabel('Zavalon PAS')

plt.title('PAS comparison (R^2 = ' + str(round(r_square, 2)) + ')')

plt.tight_layout()

plt.savefig("cnn_zavalon_pas_comparison.png")
plt.savefig("cnn_zavalon_pas_comparison.svg")
plt.savefig("cnn_zavalon_pas_comparison.eps")
plt.show()


