import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import scipy
import numpy
import scipy.sparse as sp
import scipy.io as spio


data = pd.read_csv('cross_test_Onesided2AntimisprimeOrigDropout_All.csv',sep='\t')
print(len(data))

run_name = 'Onesided2AntimisprimeOrigDropout_All_All_Libs'

model_map = [
	'TOMM5',
	'DoubleDope',
	'Simple',
	'AARS',
	'ATR',
	'SOX13',
	'Combined'
]

lib_map = [
	'TOMM5',
	'DoubleDope',
	'Simple',
	'AARS',
	'ATR',
	'SOX13',
	'HSPE1',
	'SNHG6',
	'WHAMMP2'
]


f = plt.figure(figsize=(8, 8))


cross_test_r2 = numpy.zeros((len(model_map), len(lib_map) + 1))

for i in range(0, len(model_map)) :
	model_name = model_map[i]
	for j in range(0, len(lib_map)) :
		lib_name = lib_map[j]

		data_model_lib = data.ix[(data.model == model_name) & (data.library_name == lib_name)].copy(deep=True)

		logodds_test_curr = numpy.ravel(numpy.array(list(data_model_lib.observed_logodds)))
		logodds_test_hat_curr = numpy.ravel(numpy.array(list(data_model_lib.predicted_logodds)))

		#Calculate Pearson r
		logodds_test_hat_avg = numpy.average(logodds_test_hat_curr)
		logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_hat_curr - logodds_test_hat_avg))

		logodds_test_avg = numpy.average(logodds_test_curr)
		logodds_test_std = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg, logodds_test_curr - logodds_test_avg))

		cov = numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_curr - logodds_test_avg)
		test_r = cov / (logodds_test_hat_std * logodds_test_std)

		test_rsquare = test_r * test_r

		cross_test_r2[i, j] = test_rsquare

		print('Model: ' + model_name + ', Test set: ' + lib_name + ', R^2 = ' + str(round(test_rsquare, 2)))

	data_model = data.ix[(data.model == model_name)].copy(deep=True)

	logodds_test_curr = numpy.ravel(numpy.array(list(data_model.observed_logodds)))
	logodds_test_hat_curr = numpy.ravel(numpy.array(list(data_model.predicted_logodds)))

	#Calculate Pearson r
	logodds_test_hat_avg = numpy.average(logodds_test_hat_curr)
	logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_hat_curr - logodds_test_hat_avg))

	logodds_test_avg = numpy.average(logodds_test_curr)
	logodds_test_std = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg, logodds_test_curr - logodds_test_avg))

	cov = numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_curr - logodds_test_avg)
	test_r = cov / (logodds_test_hat_std * logodds_test_std)

	test_rsquare = test_r * test_r

	cross_test_r2[i, len(lib_map)] = test_rsquare

	print('Model: ' + model_name + ', Test set: Combined, R^2 = ' + str(round(test_rsquare, 2)))



plt.pcolor(cross_test_r2[::-1,:],cmap=plt.get_cmap('Reds'),vmin=0, vmax=1)
plt.colorbar()

plt.xlabel('Library test set', fontsize=28)
plt.ylabel('DNN model', fontsize=28)

lib_map.append('Combined')

plt.xticks(numpy.arange(cross_test_r2.shape[1]) + 0.5, lib_map, rotation=45)


plt.yticks(numpy.arange(cross_test_r2.shape[0]) + 0.5, model_map[::-1], rotation=45)

plt.axis([0, cross_test_r2.shape[1], 0, cross_test_r2.shape[0]])

plt.savefig('cross_test_' + run_name + '.svg')
plt.close()
