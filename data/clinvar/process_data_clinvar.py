import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

data = pd.read_csv('apadb_snps_combined.csv',sep='\t')#_mirna
filtered = '_apadb_clinvar2combined2'#mirna



#data = data.ix[((data.ref_seq.str.contains('AATAAA')) | (data.ref_seq.str.contains('TATAA'))) | (data.ref_seq.str.contains('ATTAAA'))]

#data = data.ix[data.num_sites <= 4]
#data = data.ix[data.up_site_dist >= 100]
#data = data.ix[data.dn_site_dist >= 100]
#data = data.ix[data.up_site_dist >= 500]
#data = data.ix[data.dn_site_dist >= 500]

#data = data.ix[data.reads >= 40]

'''data['snp_ref_seq'] = data.apply(lambda row: row['ref_seq'][row['var_pos'] - 50: row['var_pos'] + 50], axis=1)
data['snp_var_seq'] = data.apply(lambda row: row['var_seq'][row['var_pos'] - 50: row['var_pos'] + 50], axis=1)

data = data.ix[
	((((((((((((data.snp_ref_seq.str.contains('AATAAA')) | (data.snp_ref_seq.str.contains('ATAA'))) | (data.snp_ref_seq.str.contains('ATTAAA')))
	| data.snp_var_seq.str.contains('AATAAA')) | data.snp_var_seq.str.contains('ATAA')) | data.snp_var_seq.str.contains('ATTAAA'))
	| data.snp_ref_seq.str.contains('AGTAA')) | data.snp_var_seq.str.contains('AGTAA'))
	| data.snp_ref_seq.str.contains('AACAA')) | data.snp_var_seq.str.contains('AACAA'))
	| data.snp_ref_seq.str.contains('AAGAA')) | data.snp_var_seq.str.contains('AAGAA'))
]'''



action = '_fullseq'

gene = data.gene.values
ids = data.clinvar_id.values

snp_pos = list(data.var_pos)
snp_type = data.vartype.values
snp_region = data.region.values
snp_significance = data.significance.values
rel_use = list(data.rel_use)

gene = np.ravel(np.array(gene))
print(gene.shape)
ids = np.ravel(np.array(ids))
print(ids.shape)

snptype = np.zeros((len(snp_type), 1))
snpregion = np.zeros((len(snp_region), 1))

snpsign = np.zeros((len(snp_significance), 1))
apadist = np.zeros((len(snp_pos), 1))

y_ref = np.zeros((len(rel_use), 1))

for i in range(0, len(rel_use)) :
	y_ref[i, 0] = float(rel_use[i])
	
	if snp_type[i] == 'single nucleotide substitution' :
		snptype[i, 0] = 0
	elif snp_type[i] in ['indel', 'insertion', 'deletion'] :
		snptype[i, 0] = 1
	
	if snp_region[i] == 'UTR3' :
		snpregion[i, 0] = 1
	elif snp_region[i] == 'Intron' :
		snpregion[i, 0] = 2
	elif snp_region[i] == 'Exon' :
		snpregion[i, 0] = 3
	elif snp_region[i] == 'Extension' :
		snpregion[i, 0] = 4
	elif snp_region[i] == 'UTR5' :
		snpregion[i, 0] = 5

	if snp_significance[i] == 'Benign' or ('benign' in snp_significance[i].lower() and 'likely' not in snp_significance[i].lower()) :
		snpsign[i, 0] = 1
	elif snp_significance[i] == 'Likely benign' or ('benign' in snp_significance[i].lower() and 'likely' in snp_significance[i].lower()) :
		snpsign[i, 0] = 2
	elif snp_significance[i] == 'Uncertain significance' :
		snpsign[i, 0] = 3
	elif snp_significance[i] == 'Conflicting interpretations of pathogenicity' :
		snpsign[i, 0] = 4
	elif snp_significance[i] == 'Likely pathogenic' or snp_significance[i] == 'Pathogenic/Likely pathogenic' or ('pathogenic' in snp_significance[i].lower() and 'likely' in snp_significance[i].lower()):
		snpsign[i, 0] = 5
	elif snp_significance[i] == 'Pathogenic' or ('pathogenic' in snp_significance[i].lower() and 'likely' not in snp_significance[i].lower()) :
		snpsign[i, 0] = 6
	
	apadist[i, 0] = float(snp_pos[i]) - 150.0

print(y_ref.shape)


degenerate_fullseq_global_ref = (data.ref_seq.str.slice(0, 350)).values
degenerate_fullseq_global_var = (data.var_seq.str.slice(0, 350)).values


#Initialize the matrix of input feature vectors.
#This matrix is from the beginning transposed to better work with Pandas.
#Shape of X is (rows, cols) = (8192, N), where 8192 is the number of possible 6-mers in each random region.


if action == '_fullseq' :
	X_motif_ref = sp.lil_matrix((len(y_ref), 350 * 4))
	X_motif_var = sp.lil_matrix((len(y_ref), 350 * 4))



dump_counter = 0
dump_max = min(30000, len(y_ref))
X_motif_acc_ref = None
X_motif_ref = sp.lil_matrix((dump_max, X_motif_ref.shape[1]))
X_motif_acc_var = None
X_motif_var = sp.lil_matrix((dump_max, X_motif_var.shape[1]))
seqs_left = len(y_ref)

valid_snps = []

for i in range(0, len(y_ref)):
	if i % 10000 == 0:
		print("Read up to sequence: " + str(i))
	
	if dump_counter >= dump_max :
		
		if X_motif_acc_ref == None :
			X_motif_acc_ref = sp.csr_matrix(X_motif_ref)
			X_motif_acc_var = sp.csr_matrix(X_motif_var)
		else :
			X_motif_acc_ref = sp.vstack([X_motif_acc_ref, sp.csr_matrix(X_motif_ref)])
			X_motif_acc_var = sp.vstack([X_motif_acc_var, sp.csr_matrix(X_motif_var)])
		
		if seqs_left >= dump_max :
			X_motif_ref = sp.lil_matrix((dump_max, X_motif_ref.shape[1]))
			X_motif_var = sp.lil_matrix((dump_max, X_motif_var.shape[1]))
		else :
			X_motif_ref = sp.lil_matrix((seqs_left, X_motif_ref.shape[1]))
			X_motif_var = sp.lil_matrix((seqs_left, X_motif_var.shape[1]))
		dump_counter = 0
	
	dump_counter += 1
	seqs_left -= 1
	
	full_seq_global_ref = degenerate_fullseq_global_ref[i]
	full_seq_global_var = degenerate_fullseq_global_var[i]
	
	
	if action == '_fullseq' :
		if full_seq_global_ref != full_seq_global_var :
			valid_snps.append(i)
	
		X_point = np.zeros((len(full_seq_global_ref), 4))

		for j in range(0, len(full_seq_global_ref)) :
			if full_seq_global_ref[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq_global_ref[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq_global_ref[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq_global_ref[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif_ref[i % dump_max,:] = X_point.reshape((1, len(full_seq_global_ref) * 4))
		
		X_point = np.zeros((len(full_seq_global_var), 4))

		for j in range(0, len(full_seq_global_var)) :
			if full_seq_global_var[j] == "A" :
				X_point[j, 0] = 1
			elif full_seq_global_var[j] == "C" :
				X_point[j, 1] = 1
			elif full_seq_global_var[j] == "G" :
				X_point[j, 2] = 1
			elif full_seq_global_var[j] == "T" :
				X_point[j, 3] = 1
		
		X_motif_var[i % dump_max,:] = X_point.reshape((1, len(full_seq_global_var) * 4))
	
	



X_motif_acc_ref = sp.csr_matrix(X_motif_ref)
X_motif_acc_var = sp.csr_matrix(X_motif_var)
print(X_motif_acc_ref.shape)

csr_X_motif_ref = X_motif_acc_ref
csr_X_motif_var = X_motif_acc_var
#print(csr_X.shape)

gene = gene[valid_snps]
ids = ids[valid_snps]

csr_X_motif_ref = csr_X_motif_ref[valid_snps, :]
csr_X_motif_var = csr_X_motif_var[valid_snps, :]

y_ref = y_ref[valid_snps, :]

snptype = snptype[valid_snps, :]
snpregion = snpregion[valid_snps, :]
snpsign = snpsign[valid_snps, :]

apadist = apadist[valid_snps, :]

save_path = '/media/johli/OS/Users/Johannes/Desktop/apa_general/snps/'

np.save(save_path + 'apa' + filtered + '_gene', gene)
np.save(save_path + 'apa' + filtered + '_id', ids)

np.save(save_path + 'apa' + filtered + '_output_ref', y_ref)

np.save(save_path + 'apa' + filtered + '_snptype', snptype)
np.save(save_path + 'apa' + filtered + '_snpregion', snpregion)
np.save(save_path + 'apa' + filtered + '_snpsign', snpsign)

np.save(save_path + 'apa' + filtered + '_apadist', apadist)


#Store the input matrix X in a matrix market file format, which is a sparse coordinate-based markup.
X_dict = dict()

X_dict["ref"] = csr_X_motif_ref
X_dict["var"] = csr_X_motif_var


spio.savemat(save_path + 'apa' + action + filtered + '_input', X_dict)
