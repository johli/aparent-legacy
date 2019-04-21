#import pandas as pd
import scipy
import numpy as np
import scipy.sparse as sp
import scipy.io as spio
import scipy.sparse.linalg as spalg

#from pylab import *
#%matplotlib inline

#import pylab as pl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.optimize as spopt

import pandas as pd

import pickle

def safe_log(x, minval=0.02):#0.01
    return np.log(x.clip(min=minval))

def print_performance_stats(X, L_input, y, w, w_L, w_0) :
	y_hat = compute_probability_one(X, L_input, w, w_L, w_0)

	SSE = (y - y_hat).T.dot(y - y_hat)

	y_average = np.average(y, axis=0)

	SStot = (y - y_average).T.dot(y - y_average)

	RMSE = np.sqrt(SSE / float(X.shape[0]))

	MAE = np.mean(np.abs(y_hat - y))

	dir_accuracy = np.count_nonzero(np.sign(y - 0.5) == np.sign(y_hat - 0.5))

	print("")
	#print("Training Residual SSE:")
	#print(SSE)
	#print("Training RMSE:")
	#print(RMSE)
	print("Training R^2:")
	print(1.0 - (SSE / SStot))
	print("Training mean abs error:")
	print(MAE)


#Compute the log-loss function value for the current set of weights.
def log_loss(w_bundle, *fun_args) :
	(X, L_input, y, lambda_penalty) = fun_args
	w = w_bundle[1+36:]
	w_0 = w_bundle[0]
	w_L = w_bundle[1:1+36]

	log_y_zero = safe_log(compute_probability_zero(X, L_input, w, w_L, w_0))
	log_y_one = safe_log(compute_probability_one(X, L_input, w, w_L, w_0))
	
	log_loss = (1.0 / 2.0) * lambda_penalty * np.square(np.linalg.norm(w)) - (1.0 / float(X.shape[0])) * (y.T.dot(log_y_one) + (1 - y).T.dot(log_y_zero))
	
	print("Log loss: " + str(log_loss))
	print_performance_stats(X, L_input, y, w, w_L, w_0)
	return log_loss

def log_loss_gradient(w_bundle, *fun_args) :
	(X, L_input, y, lambda_penalty) = fun_args
	w = w_bundle[1+36:]
	w_0 = w_bundle[0]
	w_L = w_bundle[1:1+36]
	N = X.shape[0]

	predicted_y_prob = compute_probability_one(X, L_input, w, w_L, w_0)
        
	w_0_gradient = - (1.0 / float(N)) * np.sum(y - predicted_y_prob)
	w_L_gradient = - (1.0 / float(N)) * L_input.T.dot(y - predicted_y_prob)
	w_gradient = 1.0 * lambda_penalty * w - (1.0 / float(N)) * X.T.dot(y - predicted_y_prob)
	
	return np.concatenate([[w_0_gradient], w_L_gradient, w_gradient])

#Compute the probability of data point x belonging to class 1.
def compute_probability_one(X, L_input, w, w_L, w_0) :
	exponential = np.exp(X.dot(w) + L_input.dot(w_L) + w_0)
	sigmoid = exponential / (1.0 + exponential)
	return sigmoid

def compute_logodds(p) :
	return safe_log(p / (1 - p))

#Compute the probability of data point x belonging to class 0.
def compute_probability_zero(X, L_input, w, w_L, w_0) :
	exponential = np.exp(X.dot(w) + L_input.dot(w_L) + w_0)
	sigmoid = 1.0 / (1.0 + exponential)
	return sigmoid

def translate_to_seq(x) :
	X_point = np.ravel(x.todense())
	X_point = X_point.reshape((len(X_point) / 4, 4))
	
	seq = ""
	for j in range(0, X_point.shape[0]) :
		if X_point[j, 0] == 1 :
			seq += "A"
		elif X_point[j, 1] == 1 :
			seq += "C"
		elif X_point[j, 2] == 1 :
			seq += "G"
		elif X_point[j, 3] == 1 :
			seq += "T"
	return seq

dataset = 'general3_antimisprime_orig_pasaligned_margin'
action = '6mer_v'
loader = np.load('npz_apa_' + action + '_' + dataset + '_input.npz')
X = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=np.int8)

y = np.ravel(np.load('apa_' + dataset + '_output.npy')[:,1])
c = np.ravel(np.load('apa_' + dataset + '_count.npy'))
L = np.ravel(np.load('apa_' + dataset + '_libindex.npy'))

L_not_apasix = np.nonzero(L < 30)[0]
X = X[L_not_apasix,:]
y = y[L_not_apasix]
c = c[L_not_apasix]
L = L[L_not_apasix]

L_filter = [2, 5, 8, 11, 20, 22]
c_filter = 0#80#0

train_set_size = 0.985#0.95#0.985#0.90#0.95#0.90

constant_set_split = True
constant_test_set_size = 48000#120000#54000#20000#9000#80000#9000#36000

model_instance = 'all_lib_test1'#'tomm5_lib_test1'


L_included = L_filter

arranged_index_len = 0
min_join_len = len(np.nonzero(L == L_included[0])[0])
if c_filter is not None :
	min_join_len = len(np.nonzero((L == L_included[0]) & (c > c_filter))[0])

for lib in L_included :
	lib_len = len(np.nonzero(L == lib)[0])
	if c_filter is not None :
		lib_len = len(np.nonzero((L == lib) & (c > c_filter))[0])

	arranged_index_len += lib_len
	if lib_len < min_join_len :
		min_join_len = lib_len

arranged_index = np.zeros(arranged_index_len, dtype=np.int)

arranged_remainder_index = 0
arranged_join_index = arranged_index_len - len(L_included) * min_join_len

for lib_i in range(0, len(L_included)) :
	lib = L_included[lib_i]

	print('Arranging lib ' + str(lib))

	#1. Get indexes of each Library

	apa_lib_index = np.nonzero(L == lib)[0]
	if c_filter is not None :
		apa_lib_index = np.nonzero((L == lib) & (c > c_filter))[0]

	#2. Sort indexes of each library by count
	c_apa_lib = c[apa_lib_index]
	sort_index_apa_lib = np.argsort(c_apa_lib)
	apa_lib_index = apa_lib_index[sort_index_apa_lib]

	#3. Shuffle indexes of each library modulo 2
	even_index_apa_lib = np.arange(len(apa_lib_index)) % 2 == 0
	odd_index_apa_lib = np.arange(len(apa_lib_index)) % 2 == 1

	apa_lib_index_even = apa_lib_index[even_index_apa_lib]
	apa_lib_index_odd = apa_lib_index[odd_index_apa_lib]

	apa_lib_index = np.concatenate([apa_lib_index_even, apa_lib_index_odd])

	#4. Join modulo 2
	i = 0
	for j in range(len(apa_lib_index) - min_join_len, len(apa_lib_index)) :
		arranged_index[arranged_join_index + i * len(L_included) + lib_i] = apa_lib_index[j]
		i += 1

	#5. Append remainder
	for j in range(0, len(apa_lib_index) - min_join_len) :
		arranged_index[arranged_remainder_index] = apa_lib_index[j]
		arranged_remainder_index += 1

print('Arranged index:')
print(len(arranged_index))
print(arranged_index)

X = X[arranged_index,:]
L = L[arranged_index]
c = c[arranged_index]
y = y[arranged_index]


L_input = np.zeros((len(L), 36))
for i in range(0, len(L)) :
	L_input[i, int(L[i])] = 1

#6mer weights
weights_to_motifs = [
	[0, 4096, 'Upstream', 6]
	,[4096, 8192, 'PAS', 6]
	,[8192, 12288, 'Downstream', 6]
	,[12288, 16384, 'Further Downstream', 6]
]


print(X.shape)
print(y.shape)

prox = np.multiply(c, y)
dist = np.multiply(c, 1-y)


X_sum = np.ravel(X.sum(axis=0))
i = np.ravel(range(0, len(X_sum)))

print(X.min())
print(X_sum.min())

plt.plot(i, X_sum)
plt.show()
plt.close()



if constant_set_split == False :
	X_train = X[:int(train_set_size * X.shape[0]),:]
	X_test = X[X_train.shape[0]:,:]

	L_train = L_input[:int(train_set_size * X.shape[0]),:]
	L_test = L_input[X_train.shape[0]:,:]

	y_train = y[:X_train.shape[0]]
	y_test = y[X_train.shape[0]:]
else :
	X_train = X[:-constant_test_set_size,:]
	X_test = X[-constant_test_set_size:,:]

	L_train = L_input[:-constant_test_set_size,:]
	L_test = L_input[-constant_test_set_size:,:]

	y_train = y[:-constant_test_set_size]
	y_test = y[-constant_test_set_size:]

	train_set_size = int(constant_test_set_size)


print("Starting regression.")

w_init = np.array(np.zeros(X.shape[1] + 1 + 36))

lambda_penalty = 0

(w_bundle, _, _) = spopt.fmin_l_bfgs_b(log_loss, w_init, fprime=log_loss_gradient, args=(X_train, L_train, y_train, lambda_penalty), maxiter = 100)

np.save('apa_' + action + '_' + dataset + '_' + model_instance + '_weights', w_bundle)

w = w_bundle[1 + 36:]
w_L = w_bundle[1:1 + 36]
w_0 = w_bundle[0]

print('w_0 = ' + str(w_0))
print('w_L = ' + str(w_L))


y_train_hat = np.ravel(compute_probability_one(X_train, L_train, w, w_L, w_0))
y_test_hat = np.ravel(compute_probability_one(X_test, L_test, w, w_L, w_0))


logodds_train_hat = safe_log(y_train_hat / (1 - y_train_hat))
logodds_train = safe_log(y_train / (1 - y_train))
logodds_train_isinf = np.isinf(logodds_train)
logodds_train_hat = logodds_train_hat[logodds_train_isinf == False]
logodds_train = logodds_train[logodds_train_isinf == False]

logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
logodds_test = safe_log(y_test / (1 - y_test))
logodds_test_isinf = np.isinf(logodds_test)
logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
logodds_test = logodds_test[logodds_test_isinf == False]
L_test = L_test[logodds_test_isinf == False, :]


SSE_train = np.dot(logodds_train - logodds_train_hat, logodds_train - logodds_train_hat)
SSE_test = np.dot(logodds_test - logodds_test_hat, logodds_test - logodds_test_hat)

y_train_average = np.mean(logodds_train)
y_test_average = np.mean(logodds_test)

SStot_train = np.dot(logodds_train - y_train_average, logodds_train - y_train_average)
SStot_test = np.dot(logodds_test - y_test_average, logodds_test - y_test_average)

RMSE_train = np.sqrt(SSE_train / float(X_train.shape[0]))
RMSE_test = np.sqrt(SSE_test / float(X_test.shape[0]))

MAE_train = np.mean(np.abs(logodds_train_hat - logodds_train))
MAE_test = np.mean(np.abs(logodds_test_hat - logodds_test))

print("")
print("Training Residual SSE:")
print(SSE_train)
print("Training RMSE:")
print(RMSE_train)
print("Training R^2:")
print(1.0 - (SSE_train / SStot_train))
print("Training mean abs error:")
print(MAE_train)

print("Test Residual SSE:")
print(SSE_test)
print("Test RMSE:")
print(RMSE_test)
print("Test R^2:")
print(1.0 - (SSE_test / SStot_test))
print("Test mean abs error:")
print(MAE_test)

print("")



#Plot prediction scatters

lib_map = {
	2 : [2, 'TOMM52', 'red'],
	5 : [5, 'TOMM55', 'red'],
	8 : [8, 'TOMM58', 'red'],
	11 : [11, 'TOMM511', 'red'],
	20 : [20, 'DoubleDope', 'blue'],
	22 : [22, 'Simple', 'green'],
	30 : [30, 'AARS', 'purple'],
	31 : [31, 'ATR', 'purple'],
	32 : [32, 'HSPE1', 'purple'],
	33 : [33, 'SNHG6', 'purple'],
	34 : [34, 'SOX13', 'purple'],
	35 : [35, 'WHAMMP2', 'purple']
}

L_test_index = np.ravel(np.argmax(L_test, axis=1))

for i in range(0, len(L_filter)) :
	lib = lib_map[L_filter[i]]
	lib_index = lib[0]
	lib_name = lib[1]
	lib_color = lib[2]

	if len(L_test_index[L_test_index == lib_index]) == 0 :
		continue

	logodds_test_curr = np.ravel(logodds_test[L_test_index == lib_index])
	logodds_test_hat_curr = np.ravel(logodds_test_hat[L_test_index == lib_index])

	#Calculate Pearson r
	logodds_test_hat_avg = np.mean(logodds_test_hat_curr)
	logodds_test_hat_std = np.sqrt(np.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_hat_curr - logodds_test_hat_avg))

	logodds_test_avg = np.mean(logodds_test_curr)
	logodds_test_std = np.sqrt(np.dot(logodds_test_curr - logodds_test_avg, logodds_test_curr - logodds_test_avg))

	cov = np.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_curr - logodds_test_avg)
	test_r = cov / (logodds_test_hat_std * logodds_test_std)

	test_rsquare = test_r * test_r

	f = plt.figure(figsize=(7, 6))

	plt.scatter(logodds_test_hat_curr, logodds_test_curr, s = np.pi * (2 * np.ones(1))**2, alpha=0.20, color='black')
	
	min_x = max(np.min(logodds_test_hat_curr), np.min(logodds_test_curr))
	max_x = min(np.max(logodds_test_hat_curr), np.max(logodds_test_curr))
	min_y = max(np.min(logodds_test_hat_curr), np.min(logodds_test_curr))
	max_y = min(np.max(logodds_test_hat_curr), np.max(logodds_test_curr))
	plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

	plt.axis([np.min(logodds_test_hat_curr) - 0.05, np.max(logodds_test_hat_curr) + 0.05, np.min(logodds_test_curr) - 0.05, np.max(logodds_test_curr) + 0.05])
	plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
	plt.savefig("lr_test_" + model_instance + "_" + lib_name + "_train_size_" + str(train_set_size) + "_black.png")
	plt.savefig("lr_test_" + model_instance + "_" + lib_name + "_train_size_" + str(train_set_size) + "_black.svg")
	plt.show()
	plt.close()


#Calculate Pearson r
logodds_test_hat_avg = np.mean(logodds_test_hat)
logodds_test_hat_std = np.sqrt(np.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

logodds_test_avg = np.mean(logodds_test)
logodds_test_std = np.sqrt(np.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

cov = np.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test - logodds_test_avg)
test_r = cov / (logodds_test_hat_std * logodds_test_std)

test_rsquare = test_r * test_r

f = plt.figure(figsize=(7, 6))

plt.scatter(logodds_test_hat, logodds_test, s = np.pi * (2 * np.ones(1))**2, alpha=0.20, color='black')
min_x = max(np.min(logodds_test_hat), np.min(logodds_test))
max_x = min(np.max(logodds_test_hat), np.max(logodds_test))
min_y = max(np.min(logodds_test_hat), np.min(logodds_test))
max_y = min(np.max(logodds_test_hat), np.max(logodds_test))
plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

plt.xlim(np.min(logodds_test_hat), np.max(logodds_test_hat))
plt.ylim(np.min(logodds_test), np.max(logodds_test))

plt.axis([np.min(logodds_test_hat) - 0.05, np.max(logodds_test_hat) + 0.05, np.min(logodds_test) - 0.05, np.max(logodds_test) + 0.05])
plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
plt.savefig("lr_test_" + model_instance + "_train_size_" + str(train_set_size) + "_total_black.png")
plt.savefig("lr_test_" + model_instance + "_train_size_" + str(train_set_size) + "_total_black.svg")
plt.show()
plt.close()





num_non_zero_weights = 0
num_weights = 0
for i in range(0, len(w)) :
	if(w[i] > 0.0 or w[i] < 0.0) :
		num_non_zero_weights = num_non_zero_weights + 1
	num_weights = num_weights + 1

print("Number of non-zero weights: " + str(num_non_zero_weights))
print("Number of weights: " + str(num_weights))


bases = "ACGT"

mer4 = []
mer6 = []
for base1 in bases:
	for base2 in bases:
		for base3 in bases:
			for base4 in bases:
				mer4.append(base1 + base2 + base3 + base4)
				for base5 in bases:
					for base6 in bases:
						mer6.append(base1 + base2 + base3 + base4 + base5 + base6)


top_n = 50

for weight_region in weights_to_motifs :
	w_start = weight_region[0]
	w_end = weight_region[1]
	w_name = weight_region[2]
	w_nmer = weight_region[3]
	
	w_selection = w[w_start:w_end]
	
	highest_weight_index = np.argsort(w_selection)[::-1]
	#Pick the 10 first ones of the reversed sorted vector.
	highest_weight_index_top = highest_weight_index[0:top_n]

	lowest_weight_index = np.argsort(w_selection)
	#Pick the 10 first ones of the reversed sorted vector.
	lowest_weight_index_top = lowest_weight_index[0:top_n]
	
	mer = None
	num_mers = 0
	if w_nmer == 4 :
		mer = mer4
		num_mers = 256
	elif w_nmer == 6 :
		mer = mer6
		num_mers = 4096
	
	print("")
	print(w_name + " regulatory elements:")
	
	print("")
	print("Largest enhancers:")
	
	for i in range(0, top_n) :
		print(str(highest_weight_index_top[i]) + ", " + str(mer[highest_weight_index_top[i] % num_mers]) + ": " + str(w_selection[highest_weight_index_top[i]]))

	print("")
	print("Largest silencers")

	for i in range(0, top_n) :
		print(str(lowest_weight_index_top[i]) + ", " + str(mer[lowest_weight_index_top[i] % num_mers]) + ": " + str(w_selection[lowest_weight_index_top[i]]))

