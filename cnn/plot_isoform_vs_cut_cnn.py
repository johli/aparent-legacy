import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
import scipy
import numpy
import scipy.sparse as sp
import scipy.io as spio


iso_run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_pred_30_to_35'
cut_run_name = '_Global2_Onesided2Cuts2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34'

save_folder = 'scatter_plots/v2/'

data_iso = pd.read_csv('test_predictions_' + iso_run_name + '.csv',sep='\t')
print(len(data_iso))

data_cut = pd.read_csv('test_predictions_' + cut_run_name + '.csv',sep='\t')
print(len(data_cut))

data = data_iso.set_index(['seq']).join(data_cut.set_index('seq'), how='inner', lsuffix='_iso', rsuffix='_cut')

#print(data.head())
print(len(data))

save_as_run = 'Onesided2AntimisprimeOrigDropout_IsoformVsCuts_pred_30_to_35'
#save_as_run = 'Onesided2AntimisprimeOrigDropout_IsoformVsCuts_pred_30_31_34'
#save_as_run = 'Onesided2AntimisprimeOrig_IsoformVsCuts_pred_30_to_35'
#save_as_run = 'Onesided2AntimisprimeOrig_IsoformVsCuts_pred_30_31_34'


#data = data.ix[data.library_iso != 32]
#data = data.ix[data.library_iso != 33]
#data = data.ix[data.library_iso != 35]

print(len(data))

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

#Observed correlation scatter
logodds_test_iso = numpy.ravel(numpy.array(list(data.observed_logodds_iso)))
logodds_test_cut = numpy.ravel(numpy.array(list(data.observed_logodds_cut)))

#Calculate Pearson r
logodds_test_iso_avg = numpy.average(logodds_test_iso)
logodds_test_iso_std = numpy.sqrt(numpy.dot(logodds_test_iso - logodds_test_iso_avg, logodds_test_iso - logodds_test_iso_avg))

logodds_test_cut_avg = numpy.average(logodds_test_cut)
logodds_test_cut_std = numpy.sqrt(numpy.dot(logodds_test_cut - logodds_test_cut_avg, logodds_test_cut - logodds_test_cut_avg))

cov = numpy.dot(logodds_test_iso - logodds_test_iso_avg, logodds_test_cut - logodds_test_cut_avg)
test_r = cov / (logodds_test_iso_std * logodds_test_cut_std)

test_rsquare = test_r * test_r

f = plt.figure(figsize=(7, 6))

plt.scatter(logodds_test_iso, logodds_test_cut, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.10, color='black')
min_x = max(numpy.min(logodds_test_iso), numpy.min(logodds_test_cut))
max_x = min(numpy.max(logodds_test_iso), numpy.max(logodds_test_cut))
min_y = max(numpy.min(logodds_test_iso), numpy.min(logodds_test_cut))
max_y = min(numpy.max(logodds_test_iso), numpy.max(logodds_test_cut))
plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

plt.xlim(numpy.min(logodds_test_iso), numpy.max(logodds_test_iso))
plt.ylim(numpy.min(logodds_test_cut), numpy.max(logodds_test_cut))

plt.axis([numpy.min(logodds_test_iso) - 0.05, numpy.max(logodds_test_iso) + 0.05, numpy.min(logodds_test_cut) - 0.05, numpy.max(logodds_test_cut) + 0.05])
plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
plt.savefig(save_folder + "cnn_test" + save_as_run + "_observed_iso_cut_black.png")
plt.savefig(save_folder + "cnn_test" + save_as_run + "_observed_iso_cut_black.svg")
plt.show()
plt.close()






#Predicted correlation scatter
logodds_test_iso = numpy.ravel(numpy.array(list(data.predicted_logodds_iso)))
logodds_test_cut = numpy.ravel(numpy.array(list(data.predicted_logodds_cut)))

#Calculate Pearson r
logodds_test_iso_avg = numpy.average(logodds_test_iso)
logodds_test_iso_std = numpy.sqrt(numpy.dot(logodds_test_iso - logodds_test_iso_avg, logodds_test_iso - logodds_test_iso_avg))

logodds_test_cut_avg = numpy.average(logodds_test_cut)
logodds_test_cut_std = numpy.sqrt(numpy.dot(logodds_test_cut - logodds_test_cut_avg, logodds_test_cut - logodds_test_cut_avg))

cov = numpy.dot(logodds_test_iso - logodds_test_iso_avg, logodds_test_cut - logodds_test_cut_avg)
test_r = cov / (logodds_test_iso_std * logodds_test_cut_std)

test_rsquare = test_r * test_r

f = plt.figure(figsize=(7, 6))

plt.scatter(logodds_test_iso, logodds_test_cut, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.10, color='black')
min_x = max(numpy.min(logodds_test_iso), numpy.min(logodds_test_cut))
max_x = min(numpy.max(logodds_test_iso), numpy.max(logodds_test_cut))
min_y = max(numpy.min(logodds_test_iso), numpy.min(logodds_test_cut))
max_y = min(numpy.max(logodds_test_iso), numpy.max(logodds_test_cut))
plt.plot([min_x, max_x], [min_y, max_y], alpha=0.5, color='darkblue', linewidth=3)

plt.xlim(numpy.min(logodds_test_iso), numpy.max(logodds_test_iso))
plt.ylim(numpy.min(logodds_test_cut), numpy.max(logodds_test_cut))

plt.axis([numpy.min(logodds_test_iso) - 0.05, numpy.max(logodds_test_iso) + 0.05, numpy.min(logodds_test_cut) - 0.05, numpy.max(logodds_test_cut) + 0.05])
plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=36)
plt.savefig(save_folder + "cnn_test" + save_as_run + "_predicted_iso_cut_black.png")
plt.savefig(save_folder + "cnn_test" + save_as_run + "_predicted_iso_cut_black.svg")
plt.show()
plt.close()