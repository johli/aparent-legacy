import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

filtered = '_general_cuts_antimisprime_orig'

#Read cached cut matrix
cut_distribution = spio.loadmat('apa' + filtered + '_cutdistribution.mat')['cuts']
cut_distribution = sp.csr_matrix(cut_distribution)

c = np.ravel(np.load('apa' + filtered + '_count.npy'))


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


data = pd.read_csv('apa_general_cuts_antimisprime.csv',sep=',')

def map_name(library) :
	if library in library_dict :
		return library_dict[library]
	else :
		return 'UNKNOWN'

data['library_name'] = data['library'].apply(lambda x: map_name(x))


L = np.ravel(np.array(data.library))


for library in library_dict :
	library_name  = library_dict[library]

	L_lib = np.nonzero(L == library)[0]

	cut_distribution_lib = cut_distribution[L_lib, :185]
	c_lib = c[L_lib]

	cut_summary_lib = np.zeros(185)
	for i in range(0, len(L_lib)) :
		cut_summary_lib += np.ravel(cut_distribution_lib[i, :185].todense()) * c_lib[i]

	prob_summary_lib = np.ravel(cut_distribution_lib[:, :185].mean(axis = 0))

	plt.plot(np.arange(185), cut_summary_lib, color = 'green')
	plt.xlabel('Aligned Sequence Position')
	plt.ylabel('Cut Distribution')
	plt.title('Aligned Cut Count')
	plt.grid(True)
	plt.savefig("general_cut_distribution_" + library_name + "_count.png")
	plt.savefig("general_cut_distribution_" + library_name + "_count.svg")
	plt.close()


	plt.plot(np.arange(185), prob_summary_lib, color = 'red')
	plt.xlabel('Aligned Sequence Position')
	plt.ylabel('Mean Cut Probability')
	plt.title('Aligned Cut Probability')
	plt.grid(True)
	plt.savefig("general_cut_distribution_" + library_name + "_meanprob.png")
	plt.savefig("general_cut_distribution_" + library_name + "_meanprob.svg")
	plt.close()









'''
cut_summary = np.zeros((4, 185))
cut_summary[0, :] = np.ravel(cut_distribution[0:n1, :185].mean(axis = 0))
cut_summary[1, :] = np.ravel(cut_distribution[n1:n1 + n2, :185].mean(axis = 0))
cut_summary[2, :] = np.ravel(cut_distribution[n1 + n2:n1 + n2 + n3, :185].mean(axis = 0))
cut_summary[3, :] = np.ravel(cut_distribution[n1 + n2 + n3:n1 + n2 + n3 + n4, :185].mean(axis = 0))


plt.plot(np.arange(185), cut_summary[0, :], color = 'red')
plt.plot(np.arange(185), cut_summary[1, :], color = 'blue')
plt.plot(np.arange(185), cut_summary[2, :], color = 'green')
plt.plot(np.arange(185), cut_summary[3, :], color = 'purple')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Mean Cut Probability')
plt.title('Aligned Cut Probability')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_meanprob.png")
plt.close()


tomm5_prox_c = np.sum(np.ravel(cut_distribution[0:n1, :185].mean(axis = 0)))
tomm5_dist_c = np.mean(np.ravel(cut_distribution[0:n1, 185].todense()))
print('TOMM5  Mean Prox Ratio: ' + str(round(tomm5_prox_c, 2)))
print('            Dist Ratio: ' + str(round(tomm5_dist_c, 2)))

symprx_prox_c = np.sum(np.ravel(cut_distribution[n1:n1+n2, :185].mean(axis = 0)))
symprx_dist_c = np.mean(np.ravel(cut_distribution[n1:n1+n2, 185].todense()))
print('SYMPRX Mean Prox Ratio: ' + str(round(symprx_prox_c, 2)))
print('            Dist Ratio: ' + str(round(symprx_dist_c, 2)))

simple_prox_c = np.sum(np.ravel(cut_distribution[n1+n2:n1+n2+n3, :185].mean(axis = 0)))
simple_dist_c = np.mean(np.ravel(cut_distribution[n1+n2:n1+n2+n3, 185].todense()))
print('SIMPLE Mean Prox Ratio: ' + str(round(simple_prox_c, 2)))
print('            Dist Ratio: ' + str(round(simple_dist_c, 2)))

six_prox_c = np.sum(np.ravel(cut_distribution[n1+n2+n3:n1+n2+n3+n4, :185].mean(axis = 0)))
six_dist_c = np.mean(np.ravel(cut_distribution[n1+n2+n3:n1+n2+n3+n4, 185].todense()))
print('SIX    Mean Prox Ratio: ' + str(round(six_prox_c, 2)))
print('            Dist Ratio: ' + str(round(six_dist_c, 2)))




cut_summary = np.zeros((4, 185))

for i in range(0, n1) :
	c_curr = c[i, 0]
	cut_summary[0, :] += np.ravel(cut_distribution[i, :185].todense()) * c_curr

for i in range(n1, n1 + n2) :
	c_curr = c[i, 0]
	cut_summary[1, :] += np.ravel(cut_distribution[i, :185].todense()) * c_curr

for i in range(n1 + n2, n1 + n2 + n3) :
	c_curr = c[i, 0]
	cut_summary[2, :] += np.ravel(cut_distribution[i, :185].todense()) * c_curr

for i in range(n1 + n2 + n3, n1 + n2 + n3 + n4) :
	c_curr = c[i, 0]
	cut_summary[3, :] += np.ravel(cut_distribution[i, :185].todense()) * c_curr


plt.plot(np.arange(185), cut_summary[0, :], color = 'red')
plt.plot(np.arange(185), cut_summary[1, :], color = 'blue')
plt.plot(np.arange(185), cut_summary[2, :], color = 'green')
plt.plot(np.arange(185), cut_summary[3, :], color = 'purple')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Cut Distribution')
plt.title('Aligned Cut Count')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_count.png")
plt.close()


tomm5_prox_c = np.sum(np.ravel(cut_summary[0, :]))
tomm5_dist_c = np.sum(np.multiply(np.ravel(cut_distribution[0:n1, 185].todense()), np.ravel(c[0:n1, 0])))
print('TOMM5  Sum Prox Count: ' + str(round(tomm5_prox_c, 2)))
print('           Dist Count: ' + str(round(tomm5_dist_c, 2)))

symprx_prox_c = np.sum(np.ravel(cut_summary[1, :]))
symprx_dist_c = np.sum(np.multiply(np.ravel(cut_distribution[n1:n1+n2, 185].todense()), np.ravel(c[n1:n1+n2, 0])))
print('SYMPRX Sum Prox Count: ' + str(round(symprx_prox_c, 2)))
print('           Dist Count: ' + str(round(symprx_dist_c, 2)))

simple_prox_c = np.sum(np.ravel(cut_summary[2, :]))
simple_dist_c = np.sum(np.multiply(np.ravel(cut_distribution[n1+n2:n1+n2+n3, 185].todense()), np.ravel(c[n1+n2:n1+n2+n3, 0])))
print('SIMPLE Sum Prox Count: ' + str(round(simple_prox_c, 2)))
print('           Dist Count: ' + str(round(simple_dist_c, 2)))

six_prox_c = np.sum(np.ravel(cut_summary[3, :]))
six_dist_c = np.sum(np.multiply(np.ravel(cut_distribution[n1+n2+n3:n1+n2+n3+n4, 185].todense()), np.ravel(c[n1+n2+n3:n1+n2+n3+n4, 0])))
print('SIX    Sum Prox Count: ' + str(round(six_prox_c, 2)))
print('           Dist Count: ' + str(round(six_dist_c, 2)))





#TOMM5
c_vec = np.ravel(np.array(total_count1))

cut_summary = np.zeros(185)

for i in range(0, n1) :
	c_curr = c_vec[i]
	cut_summary += np.ravel(cut_distribution[i, :185].todense()) * c_curr


plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Cut Distribution')
plt.title('Aligned Cut Count')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_tomm5_count.png")
plt.close()

cut_summary = np.ravel(cut_distribution[0:n1, :185].mean(axis = 0))

plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Mean Cut Probability')
plt.title('Aligned Cut Probability')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_tomm5_meanprob.png")
plt.close()


#SYMPRX
c_vec = np.ravel(np.array(total_counts_vs_all2))

cut_summary = np.zeros(185)

for i in range(0, n2) :
	c_curr = c_vec[i]
	cut_summary += np.ravel(cut_distribution[n1 + i, :185].todense()) * c_curr


plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Cut Distribution')
plt.title('Aligned Cut Count')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_symprx_count.png")
plt.close()

cut_summary = np.ravel(cut_distribution[n1:n1 + n2, :185].mean(axis = 0))

plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Mean Cut Probability')
plt.title('Aligned Cut Probability')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_symprx_meanprob.png")
plt.close()


#SIMPLE
c_vec = np.ravel(np.array(total_count3))

cut_summary = np.zeros(185)

for i in range(0, n3) :
	c_curr = c_vec[i]
	cut_summary += np.ravel(cut_distribution[n1 + n2 + i, :185].todense()) * c_curr


plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Cut Distribution')
plt.title('Aligned Cut Count')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_simple_count.png")
plt.close()

cut_summary = np.ravel(cut_distribution[n1 + n2:n1 + n2 + n3, :185].mean(axis = 0))

plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Mean Cut Probability')
plt.title('Aligned Cut Probability')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_simple_meanprob.png")
plt.close()


#APASIX
c_vec = np.ravel(np.array(total_count4))

cut_summary = np.zeros(185)

for i in range(0, n4) :
	c_curr = c_vec[i]
	cut_summary += np.ravel(cut_distribution[n1 + n2 + n3 + i, :185].todense()) * c_curr


plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Cut Distribution')
plt.title('Aligned Cut Count')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_six_count.png")
plt.close()

cut_summary = np.ravel(cut_distribution[n1 + n2 + n3:n1 + n2 + n3 + n4, :185].mean(axis = 0))

plt.plot(np.arange(185), cut_summary, color = 'red')

plt.xlabel('Aligned Sequence Position')
plt.ylabel('Mean Cut Probability')
plt.title('Aligned Cut Probability')
#plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.savefig("general_cut_distribution_six_meanprob.png")
plt.close()'''


