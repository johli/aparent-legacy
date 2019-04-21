import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import scipy
import numpy
import scipy.sparse as sp
import scipy.io as spio


#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_pred_32_33_35'
run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_32_33_35_pred_30_31_34'

#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_pred_30_to_35'
#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_32_33_35_pred_30_to_35'

save_folder = 'scatter_plots/v3/'

data = pd.read_csv('test_predictions_' + run_name + '.csv',sep='\t')
print(len(data))

#data = data.ix[(data.observed_logodds > -1.5) & (data.observed_logodds < 1.5)]
#data = data.ix[data['count'] >= 100]



scatter_alpha=0.15#0.10

#data = data.ix[data.library == 32]

#data = data.ix[data.library != 32]
#data = data.ix[data.library != 33]
#data = data.ix[data.library != 35]


data_index = numpy.arange(len(data))
data = data.ix[data_index >= len(data) - 60000]#96000


'''data_index = numpy.arange(len(data))
from_fraction = 0.33
data_from = int(from_fraction * len(data))
data = data.ix[data_index >= data_from]'''

test_size = '20000'#'002'#'003'

print(len(data))

#print(len(data[data.library == 32]))
#print(len(data[data.library == 33]))
#print(len(data[data.library == 35]))

lib_map = [
	[2, 'TOMM52', 'red'],
	[5, 'TOMM55', 'red'],
	[8, 'TOMM58', 'red'],
	[11, 'TOMM511', 'red'],
	[0, 'TOMM5', 'red'],
	[20, 'DoubleDope', 'blue'],
	[22, 'Simple', 'green'],
	[30, 'AARS', 'purple'],
	[31, 'ATR', 'purple'],
	[32, 'HSPE1', 'purple'],
	[33, 'SNHG6', 'purple'],
	[34, 'SOX13', 'purple'],
	[35, 'WHAMMP2', 'purple'],
]


for i in range(0, len(lib_map)) :
	lib = lib_map[i]
	lib_index = lib[0]
	lib_name = lib[1]
	lib_color = lib[2]

	if lib_name == 'TOMM5' :
		data_lib = data.ix[data.library <= 11].copy(deep=True)
	else :
		data_lib = data.ix[data.library == lib_index].copy(deep=True)

	if len(data_lib) == 0 :
		continue

	logodds_test_curr = numpy.ravel(numpy.array(list(data_lib.observed_logodds)))
	logodds_test_hat_curr = numpy.ravel(numpy.array(list(data_lib.predicted_logodds)))

	#Calculate Pearson r
	logodds_test_hat_avg = numpy.average(logodds_test_hat_curr)
	logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_hat_curr - logodds_test_hat_avg))

	logodds_test_avg = numpy.average(logodds_test_curr)
	logodds_test_std = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg, logodds_test_curr - logodds_test_avg))

	cov = numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_curr - logodds_test_avg)
	test_r = cov / (logodds_test_hat_std * logodds_test_std)

	test_rsquare = test_r * test_r

	f = plt.figure(figsize=(7, 6))

	plt.scatter(logodds_test_hat_curr, logodds_test_curr, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=scatter_alpha, color=lib_color)
	
	min_x = max(numpy.min(logodds_test_hat_curr), numpy.min(logodds_test_curr))
	max_x = min(numpy.max(logodds_test_hat_curr), numpy.max(logodds_test_curr))
	min_y = max(numpy.min(logodds_test_hat_curr), numpy.min(logodds_test_curr))
	max_y = min(numpy.max(logodds_test_hat_curr), numpy.max(logodds_test_curr))
	plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

	plt.axis([numpy.min(logodds_test_hat_curr) - 0.05, numpy.max(logodds_test_hat_curr) + 0.05, numpy.min(logodds_test_curr) - 0.05, numpy.max(logodds_test_curr) + 0.05])
	plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
	plt.savefig(save_folder + "cnn_test" + run_name + "_" + lib_name + "_test_size_" + str(test_size) + ".png")
	plt.savefig(save_folder + "cnn_test" + run_name + "_" + lib_name + "_test_size_" + str(test_size) + ".svg")
	#plt.show()
	plt.close()

	f = plt.figure(figsize=(7, 6))

	plt.scatter(logodds_test_hat_curr, logodds_test_curr, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=scatter_alpha, color='black')
	
	min_x = max(numpy.min(logodds_test_hat_curr), numpy.min(logodds_test_curr))
	max_x = min(numpy.max(logodds_test_hat_curr), numpy.max(logodds_test_curr))
	min_y = max(numpy.min(logodds_test_hat_curr), numpy.min(logodds_test_curr))
	max_y = min(numpy.max(logodds_test_hat_curr), numpy.max(logodds_test_curr))
	plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

	plt.axis([numpy.min(logodds_test_hat_curr) - 0.05, numpy.max(logodds_test_hat_curr) + 0.05, numpy.min(logodds_test_curr) - 0.05, numpy.max(logodds_test_curr) + 0.05])
	plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
	plt.savefig(save_folder + "cnn_test" + run_name + "_" + lib_name + "_test_size_" + str(test_size) + "_black.png")
	plt.savefig(save_folder + "cnn_test" + run_name + "_" + lib_name + "_test_size_" + str(test_size) + "_black.svg")
	#plt.show()
	plt.close()


logodds_test = numpy.ravel(numpy.array(list(data.observed_logodds)))
logodds_test_hat = numpy.ravel(numpy.array(list(data.predicted_logodds)))

#Calculate Pearson r
logodds_test_hat_avg = numpy.average(logodds_test_hat)
logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

logodds_test_avg = numpy.average(logodds_test)
logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

cov = numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test - logodds_test_avg)
test_r = cov / (logodds_test_hat_std * logodds_test_std)

test_rsquare = test_r * test_r

f = plt.figure(figsize=(7, 6))

plt.scatter(logodds_test_hat, logodds_test, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color='black')
min_x = max(numpy.min(logodds_test_hat), numpy.min(logodds_test))
max_x = min(numpy.max(logodds_test_hat), numpy.max(logodds_test))
min_y = max(numpy.min(logodds_test_hat), numpy.min(logodds_test))
max_y = min(numpy.max(logodds_test_hat), numpy.max(logodds_test))
plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

plt.xlim(numpy.min(logodds_test_hat), numpy.max(logodds_test_hat))
plt.ylim(numpy.min(logodds_test), numpy.max(logodds_test))

plt.axis([numpy.min(logodds_test_hat) - 0.05, numpy.max(logodds_test_hat) + 0.05, numpy.min(logodds_test) - 0.05, numpy.max(logodds_test) + 0.05])
plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
plt.savefig(save_folder + "cnn_test" + run_name + "_test_size_" + str(test_size) + "_total_black.png")
plt.savefig(save_folder + "cnn_test" + run_name + "_test_size_" + str(test_size) + "_total_black.svg")
#plt.show()
plt.close()

f = plt.figure(figsize=(7, 6))

plt.scatter(logodds_test_hat, logodds_test, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color='teal')
min_x = max(numpy.min(logodds_test_hat), numpy.min(logodds_test))
max_x = min(numpy.max(logodds_test_hat), numpy.max(logodds_test))
min_y = max(numpy.min(logodds_test_hat), numpy.min(logodds_test))
max_y = min(numpy.max(logodds_test_hat), numpy.max(logodds_test))
plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

plt.xlim(numpy.min(logodds_test_hat), numpy.max(logodds_test_hat))
plt.ylim(numpy.min(logodds_test), numpy.max(logodds_test))

plt.axis([numpy.min(logodds_test_hat) - 0.05, numpy.max(logodds_test_hat) + 0.05, numpy.min(logodds_test) - 0.05, numpy.max(logodds_test) + 0.05])
plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
plt.savefig(save_folder + "cnn_test" + run_name + "_test_size_" + str(test_size) + "_total_teal.png")
plt.savefig(save_folder + "cnn_test" + run_name + "_test_size_" + str(test_size) + "_total_teal.svg")
#plt.show()
plt.close()
