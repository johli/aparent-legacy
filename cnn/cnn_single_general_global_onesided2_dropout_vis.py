
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import theano.sparse as Tsparse

from logistic_sgd_global_onesided2_vis import LogisticRegression, TrainableImage
from mlp import HiddenLayer

#import pylab as pl
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.sparse as sp
import scipy.io as spio

import weblogolib

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class LeNetConvPoolLayer(object):
	"""Pool Layer of a convolutional network """

	def store_w(self, w_file, W):
		numpy.save(w_file, W)
	
	def store_b(self, b_file, b):
		numpy.save(b_file, b)
	
	def store_model(self, W, b):
		self.store_w(self.w_file, W)
		self.store_b(self.b_file, b)
	
	def load_w(self, w_file):
		return numpy.load(w_file)
	
	def load_b(self, b_file):
		return numpy.load(b_file)
	
	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), use_relu=True, load_model = False, w_file = '', b_file = ''):
		"""
		Allocate a LeNetConvPoolLayer with shared variable internal parameters.

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dtensor4
		:param input: symbolic image tensor, of shape image_shape

		:type filter_shape: tuple or list of length 4
		:param filter_shape: (number of filters, num input feature maps,
							  filter height, filter width)

		:type image_shape: tuple or list of length 4
		:param image_shape: (batch size, num input feature maps,
							 image height, image width)

		:type poolsize: tuple or list of length 2
		:param poolsize: the downsampling (pooling) factor (#rows, #cols)
		"""

		self.w_file = w_file
		self.b_file = b_file
		
		assert image_shape[1] == filter_shape[1]

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = numpy.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
				   numpy.prod(poolsize))
		# initialize weights with random weights
		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		
		if load_model == False : 
			self.W = theano.shared(
				numpy.asarray(
					rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
					dtype=theano.config.floatX
				),
				borrow=True
			)

			# the bias is a 1D tensor -- one bias per output feature map
			b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, borrow=True)
		else :
			self.W = theano.shared(value=self.load_w(w_file + '.npy'), name='W', borrow=True)
			self.b = theano.shared(value=self.load_b(b_file + '.npy'), name='b', borrow=True)

		# convolve input feature maps with filters
		conv_out = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			image_shape=image_shape
		)

		if(use_relu == True):
			activation = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		else:
			activation = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# downsample each feature map individually, using maxpooling
		pooled_out = downsample.max_pool_2d(
			input=activation,
			ds=poolsize,
			ignore_border=True
		)

		self.conv_out = conv_out
		self.activation = activation

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		
		if(use_relu == True):
			self.output = pooled_out
		else:
			self.output = pooled_out
		
		# store parameters of this layer
		self.params = [self.W, self.b]

def relu(x):
    return T.switch(x<0, 0, x)

class DualCNN(object):

	def generate_sequence_logos(self, test_set):
		test_set_x, test_set_y = test_set
		self.set_data(test_set_x, test_set_y)

		layer0_left = self.layer0_left

		index = T.lscalar()
		batch_size = self.batch_size
		
		input_x = test_set_x.eval()

		n_batches = input_x.shape[0] / batch_size
		
		randomized_regions = self.randomized_regions
		
		x_left = self.x_left
		x_right = self.x_right
		y = self.y

		get_layer0_activations = theano.function(
			[index],
			layer0_left.activation,
			givens={
				x_left: self.reshape_batch(test_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
			},
			on_unused_input='ignore'
		)

		activations = numpy.concatenate([get_layer0_activations(i) for i in xrange(n_batches)], axis=0)

		input_x = input_x[:activations.shape[0],:]
		input_x = numpy.asarray(input_x.todense()).reshape((activations.shape[0], self.input_size, self.num_features))[:, 0:self.left_random_size, :]

		filter_width = 7

		#(num_data_points, num_filters, seq_length, 1)
		for k in range(0, activations.shape[1]) :
			filter_activations = activations[:, k, :, :].reshape((activations.shape[0], activations.shape[2]))
			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))

			spike_index = numpy.nonzero(total_activations > 0)[0]

			filter_activations = filter_activations[spike_index, :]

			print(input_x.shape)
			print(spike_index.shape)

			filter_inputs = input_x[spike_index, :, :]

			max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

			PFM = numpy.zeros((filter_width, self.num_features))
			for i in range(0, filter_activations.shape[0]) :
				filter_input = filter_inputs[i, max_spike[i]:max_spike[i]+filter_width, :] #* filter_activations[i, max_spike[i]]
				PFM = PFM + filter_input

			print(k)
			print(PFM)
			self.get_logo(k, PFM)

	def get_logo(self, k, PFM, file_path='cnn_motif_analysis/fullseq_v/', seq_length=6, normalize=False, base_seq='') :

		if normalize == True :
			for i in range(0, PFM.shape[0]) :
				if numpy.sum(PFM[i, :]) > 0 :
					PFM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])
				#PFM[i, :] *= 10000.0
			#print(PFM)
		

		#Create weblogo from API
		logo_output_format = "png"#"svg"
		#Load data from an occurence matrix
		data = weblogolib.LogoData.from_counts('ACGT', PFM[:seq_length, :])

		#Generate color scheme
		'''colors = weblogolib.ColorScheme([
		        weblogolib.ColorGroup("A", "yellow","CFI Binder" ),
		        weblogolib.ColorGroup("C", "green","CFI Binder" ),
		        weblogolib.ColorGroup("G", "red","CFI Binder" ),
		        weblogolib.ColorGroup("T", "blue","CFI Binder" ),
		        weblogolib.ColorGroup("a", "grey","CFI Binder" ),
		        weblogolib.ColorGroup("c", "grey","CFI Binder" ),
		        weblogolib.ColorGroup("g", "grey","CFI Binder" ),
		        weblogolib.ColorGroup("t", "grey","CFI Binder" )] )'''
		color_rules = []
		for j in range(0, len(base_seq)) :
			if base_seq[j] != 'N' :
				color_rules.append(weblogolib.IndexColor([j], 'grey'))
		color_rules.append(weblogolib.SymbolColor("A", "yellow"))
		color_rules.append(weblogolib.SymbolColor("C", "green"))
		color_rules.append(weblogolib.SymbolColor("G", "red"))
		color_rules.append(weblogolib.SymbolColor("T", "blue"))
		colors = weblogolib.ColorScheme(color_rules)


		#Create options
		options = weblogolib.LogoOptions(fineprint=False,
		                                 logo_title="APA Max Class Model (CNN)", 
		                                 color_scheme=colors, 
		                                 stack_width=weblogolib.std_sizes["large"],
		                                 logo_start=1, logo_end=seq_length, stacks_per_line=seq_length)#seq_length)

		#Create logo
		logo_format = weblogolib.LogoFormat(data, options)

		#Generate image
		formatter = weblogolib.formatters[logo_output_format]
		png = formatter(data, logo_format)

		#Write it
		with open(file_path, "w") as f:
		    f.write(png)

	def generate_heat_maps(self):
		layer0_left = self.layer0_left
		
		filters = layer0_left.W.eval()
		#(n_kerns, 1, filter_width, 4)

		for k in range(0, filters.shape[0]) :
			kernel = filters[k, 0, :, :].reshape((filters.shape[2], filters.shape[3])).T
			kernel_mean = numpy.mean(kernel, axis=0)

			for j in range(0, kernel.shape[1]) :
				kernel[:, j] -= kernel_mean[j]

			kernel = numpy.fliplr(kernel)

			plt.pcolor(kernel,cmap=cm.RdBu_r)
			plt.colorbar()

			plt.xlabel('Sequence')
			plt.ylabel('Bases')
			plt.title('Kernel Heat Map')
			#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
			#xticks = mer_sorted
			plt.yticks([0.5, 1.5, 2.5, 3.5], ['T', 'G', 'C', 'A'])#BASEPAIR TO INDEX FLIPPED ON PURPOSE TO COUNTER CONVOLVE

			plt.savefig("cnn_motif_analysis/up/kernel" + str(k) + ".png")
			plt.close()


	def dropit(self, srng, weight, drop):
		# proportion of probability to retain
		retain_prob = 1 - drop
		# a masking variable
		mask = srng.binomial(n=1, p=retain_prob, size=weight.shape, dtype=theano.config.floatX)
		# final weight with dropped neurons
		return T.cast(weight * mask, theano.config.floatX)

	def dont_dropit(self, weight, drop):
		return (1 - drop)*T.cast(weight, theano.config.floatX)

	def dropout_layer(self, srng, weight, drop, train = 1):
		return T.switch(theano.tensor.eq(train, 1), self.dropit(srng, weight, drop), self.dont_dropit(weight, drop))

	def __init__(self, n_images, base_seq, cost_func, cost_layer_filter, train_set_y, train_set_L, train_set_d, learning_rate=0.1, drop=0, n_epochs=30, nkerns=[30, 40, 50], num_features=4, randomized_regions=[(2, 37), (45, 80)], dataset='default', cell_line='default', name_prefix='default', start_seqs=None):
		#numpy.random.seed()#(23455)
		rng = numpy.random.RandomState()#(23455)

		srng = RandomStreams(rng.randint(999999))
		self.randomized_regions = randomized_regions

		L_input = T.matrix('L_input')
		d_input = T.matrix('d_input')

		
		y = T.matrix('y')  # the labels are presented as 1D vector of


		self.y = y
		self.L_input = L_input
		self.d_input = d_input
		
		left_random_size = randomized_regions[0][1] - randomized_regions[0][0]
		right_random_size = randomized_regions[1][1] - randomized_regions[1][0]
		
		self.left_random_size = left_random_size
		self.right_random_size = right_random_size
		self.num_features = num_features
		self.input_size = left_random_size + right_random_size
		
		######################
		# BUILD ACTUAL MODEL #
		######################
		print('... building the model')

		layer_input = TrainableImage(rng, n_images=n_images, n_length=left_random_size, seq=base_seq, start_seqs=start_seqs)
		
		self.layer_input = layer_input


		layer0_input = layer_input.outputs.reshape((n_images, 1, left_random_size, num_features))


		# Construct the first convolutional pooling layer:
		# filtering reduces the image size to (101-6+1 , 4-4+1) = (96, 1)
		# maxpooling reduces this further to (96/1, 1/1) = (96, 1)
		# 4D output tensor is thus of shape (batch_size, nkerns[0], 96, 1)
		layer0_left = LeNetConvPoolLayer(
			rng,
			input=layer0_input,
			image_shape=(n_images, 1, left_random_size, num_features),
			filter_shape=(nkerns[0], 1, 8, num_features),
			poolsize=(2, 1),
			use_relu=True
			,load_model = True,
			w_file='model_store/' + dataset + '_' + cell_line + '_conv0_left_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_conv0_left_b'
		)
		
		# Construct the second convolutional pooling layer
		# filtering reduces the image size to (96-5+1, 1-1+1) = (92, 1)
		# maxpooling reduces this further to (92/2, 1/1) = (46, 1)
		# 4D output tensor is thus of shape (batch_size, nkerns[1], 46, 1)
		layer1 = LeNetConvPoolLayer(
			rng,
			input=layer0_left.output,
			image_shape=(n_images, nkerns[0], 89, 1),
			filter_shape=(nkerns[1], nkerns[0], 6, 1),
			poolsize=(1, 1),
			use_relu=True
			,load_model = True,
			w_file='model_store/' + dataset + '_' + cell_line + '_conv1_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_conv1_b'
		)

		# the HiddenLayer being fully-connected, it operates on 2D matrices of
		# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
		# This will generate a matrix of shape (batch_size, nkerns[1] * 16 * 1),
		# or (500, 50 * 21 * 1) = (500, 800) with the default values.
		
		#layer3_input = layer2.output.flatten(2)
		layer3_input_cnn = layer1.output.flatten(2)
		#layer3_input = layer0_left.output.flatten(2)

		layer3_input = T.concatenate([layer3_input_cnn, d_input], axis=1)

		# construct a fully-connected sigmoidal layer
		layer3 = HiddenLayer(
			rng,
			input=layer3_input,
			n_in=nkerns[1] * (84) * 1 + 1,
			n_out=256,#80,
			activation=relu#T.tanh#relu#T.tanh
			,load_model = True,
			w_file='model_store/' + dataset + '_' + cell_line + '_mlp_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_mlp_b'
		)

		layer3_output = layer3.output

		'''if drop != 0 and train_model_flag == True :
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = 1)
		elif drop != 0 :
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = 0)'''

		train_drop = T.lscalar()
		if drop != 0 :
			print('Using dropout = ' + str(drop))
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = train_drop)

		layer4_input = T.concatenate([layer3_output, L_input], axis=1)
		#layer4_input = layer3.output

		self.train_drop = train_drop

		self.layer0_left = layer0_left
		#self.layer0_right = layer0_right
		self.layer1 = layer1
		#self.layer2 = layer2
		self.layer3 = layer3

		# classify the values of the fully-connected sigmoidal layer
		layer4 = LogisticRegression(self, image=layer0_input, input=layer4_input, n_in=256 + 36, n_out=2, load_model = True, cost_func=cost_func, cost_layer_filter=cost_layer_filter,
			w_file='model_store/' + dataset + '_' + cell_line + '_lr_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_lr_b')

		
		self.output_layer = layer4

		epoch_i = T.lscalar()
		
		# the cost we minimize during training is the NLL of the model
		cost = layer4.negative_log_likelihood(y, epoch_i)

		#costs = layer4.negative_log_likelihood_unsummed(y)

		# create a list of all model parameters to be fit by gradient descent
		params = layer_input.params #layer4.params + layer3.params + layer1.params + layer0_left.params
		#params = layer4.params + layer3.params + layer1.params + layer0_left.params# + layer0_right.params
		#params = layer3.params + layer2.params + layer0_left.params + layer0_right.params
		
		# create a list of gradients for all model parameters
		grads = T.grad(cost, params)
		
		#grads = T.grad(cost, params[0])
		#grads = T.switch(T.gt(T.nonzero(T.eq(costs, 0))[0].shape[0], 0), T.set_subtensor(grads[T.nonzero(T.eq(costs, 0))[0], :, :], srng.uniform(size=params[0].shape, low=-10.0, high=10.0, dtype=theano.config.floatX)[T.nonzero(T.eq(costs, 0))[0], :, :]), grads)
		#updates = [ (params[0], params[0] - learning_rate * grads) ]

		# train_model is a function that updates the model parameters by
		# SGD Since this model has many parameters, it would be tedious to
		# manually create an update rule for each model parameter. We thus
		# create the updates list by automatically looping over all
		# (params[i], grads[i]) pairs.
		
		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		]

		train_model = theano.function(
			[epoch_i],
			cost,
			updates=updates,
			givens={
				y: train_set_y,
				L_input: train_set_L,
				d_input: train_set_d
				,train_drop: 0
			},
			on_unused_input='ignore'
		)

		self.predict = theano.function(
			[],
			layer4.recall(),
			givens={
				y: train_set_y,
				L_input: train_set_L,
				d_input: train_set_d
				,train_drop: 0
			},
			on_unused_input='ignore'
		)
		# end-snippet-1
		
		###############
		# TRAIN MODEL #
		###############
		print('... training')
		start_time = time.clock()

		epoch = 0

		logodds_series = numpy.zeros((n_images, n_epochs))

		pwms = numpy.zeros((n_epochs + 1, n_images, 185, 4))
		y_hats = numpy.zeros((n_epochs + 1, n_images))

		pwms[0, :, :, :] = numpy.array(self.layer_input.outputs.eval())

		while (epoch < n_epochs) :
			epoch = epoch + 1

			cost_ij = train_model(epoch)

			y_hat = numpy.ravel(self.predict())
			logodds_hat = numpy.log(y_hat / (1 - y_hat))
			logodds_series[:, epoch-1] = logodds_hat

			pwms[epoch, :, :, :] = numpy.array(self.layer_input.outputs.eval())
			y_hats[epoch, :] = y_hat

			if epoch % 100 == 0 :
				print('training @ epoch = ', epoch)
				print(cost_ij)
				print('logodds_hat mean = ' + str(numpy.mean(logodds_hat)))


		end_time = time.clock()

		#numpy.save("cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/" + name_prefix + "_pwms", pwms)
		#numpy.save("cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/" + name_prefix + "_usages", y_hats)

		#if cost_func in ['target', 'max_score'] :
		if True :
			f = plt.figure()


			cmap = get_cmap(n_images)
			for k in range(0, n_images) :
				plt.plot(numpy.arange(n_epochs), logodds_series[k], c=cmap(k), alpha=0.7)

			plt.xlim([0, n_epochs-1])
			plt.ylim([numpy.min(logodds_series), numpy.max(logodds_series)])
			plt.xlabel('Training Iterations', fontsize=24)
			plt.ylabel('Predicted Proximal Logodds', fontsize=24)
			f.suptitle('Max Class Training Convergence', fontsize=24)
			#plt.title('CNN Max Class Training Convergence', fontsize=24)

			plt.subplots_adjust(top=0.90)

			plt.savefig("cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/apa_array_v2/" + name_prefix + "_trainingplot.svg")
			#plt.show()
			plt.close()

			
		print('Optimization complete.')
		print >> sys.stderr, ('The code for file ' +
								  os.path.split(__file__)[1] +
								  ' ran for %.2fm' % ((end_time - start_time) / 60.))

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def generate_max_class_models(base_seq, n_tries, n_image_list, n_epochs, cost_func='max_score', target=4.0, name_prefix='') :

	for n_images in n_image_list :
		print("n_images=" + str(n_images))
		f = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/' + name_prefix + "_predictions_" + cost_func + '_' + str(n_images) + "_images.txt",'w')
		for n_try in range(0, n_tries) :
			print("n_try=" + str(n_try))
			f.write('n_try: ' + str(n_try) + '\n')
			
			train_set_y = numpy.matrix([[-target, target]])
			for i in range(0, n_images - 1) :
				train_set_y = numpy.vstack([train_set_y, numpy.matrix([[-target, target]])])


			train_set_y_t = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)

			train_set_L = numpy.zeros((n_images, 36))
			#train_set_L[:, 20] = 1
			train_set_L_t = theano.shared(numpy.asarray(train_set_L, dtype=theano.config.floatX), borrow=True)

			train_set_d = numpy.ones((n_images, 1))
			train_set_d_t = theano.shared(numpy.asarray(train_set_d, dtype=theano.config.floatX), borrow=True)
			
			
			cnn = DualCNN(
				n_images,
				base_seq,
				cost_func,
				0,#Layer filter
				train_set_y_t,
				train_set_L_t,
				train_set_d_t,
				learning_rate=0.5,
				drop=0.2,
				n_epochs=n_epochs,
				nkerns=[70, 110, 70],
				#nkerns=[50, 90, 70],
				num_features=4,
				randomized_regions=[(0, 185), (185, 185)],
				dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
				#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_small',
				name_prefix=name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try)
			)
			
			#print(cnn.layer_input.W.eval()[0])
			#print("")
			#print(cnn.layer_input.outputs.eval()[0])
			#print("")
			#print(get_consensus(cnn.layer_input.outputs.eval()))

			y_hat = cnn.predict()

			#PFM = get_consensus(cnn.layer_input.outputs.eval()[0])
			PFM = numpy.array(cnn.layer_input.outputs.eval()[0])

			print('y_hat: ' + str(y_hat[0]))
			f.write('y_hat: ' + str(y_hat[0]) + '\n')
			print(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[0])))
			f.write(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[0])) + '\n')
			for i in range(1, n_images) :
				#PFM = PFM + get_consensus(cnn.layer_input.outputs.eval()[i])
				PFM = PFM + numpy.array(cnn.layer_input.outputs.eval()[i])

				print('y_hat: ' + str(y_hat[i]))
				f.write('y_hat: ' + str(y_hat[i]) + '\n')
				print(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[i])))
				f.write(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[i])) + '\n')
			f.write('\n')

			logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + ".svg"
			cnn.get_logo(n_images, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/' + logo_name, 120, False, base_seq)
		f.close()


def generate_max_class_models_single_runs(base_seq, n_tries, n_image_list, n_epochs, cost_func='max_score', target=1.0, name_prefix='', lib_bias=None) :

	for n_images in n_image_list :
		print("n_images=" + str(n_images))
		
		PFMs = numpy.zeros((n_tries, 185, 4))
		y_hats = numpy.zeros((n_tries, 1))

		for n_try in range(0, n_tries) :
			print("n_try=" + str(n_try))
			
			train_set_y = numpy.matrix([[1.0-target, target]])
			for i in range(0, n_images - 1) :
				train_set_y = numpy.vstack([train_set_y, numpy.matrix([[1.0-target, target]])])


			train_set_y_t = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)

			train_set_L = numpy.zeros((n_images, 36))
			#train_set_L[:, 20] = 1
			if lib_bias is not None :
				train_set_L[:, lib_bias] = 1
			train_set_L_t = theano.shared(numpy.asarray(train_set_L, dtype=theano.config.floatX), borrow=True)

			train_set_d = numpy.ones((n_images, 1))
			train_set_d_t = theano.shared(numpy.asarray(train_set_d, dtype=theano.config.floatX), borrow=True)
			
			target_str = ''
			if cost_func == 'target' :
				target_str = str(target).replace('.', '')
			
			cnn = DualCNN(
				n_images,
				base_seq,
				cost_func,
				0,#Layer filter
				train_set_y_t,
				train_set_L_t,
				train_set_d_t,
				learning_rate=0.5,
				drop=0.2,
				n_epochs=n_epochs,
				nkerns=[70, 110, 70],
				#nkerns=[50, 90, 70],
				num_features=4,
				randomized_regions=[(0, 185), (185, 185)],
				dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
				#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_small',
				name_prefix=name_prefix + "_max_class_" + cost_func + target_str + '_' + str(n_images) + "_images_try_" + str(n_try)
			)
			
			#print(cnn.layer_input.W.eval()[0])
			#print("")
			#print(cnn.layer_input.outputs.eval()[0])
			#print("")
			#print(get_consensus(cnn.layer_input.outputs.eval()))

			y_hat = cnn.predict()

			#PFM = get_consensus(cnn.layer_input.outputs.eval()[0])
			PFMs[n_try * n_images:(n_try + 1) * n_images, :, :] = numpy.array(cnn.layer_input.outputs.eval())
			y_hats[n_try * n_images:(n_try + 1) * n_images, 0] = y_hat[0]

			logo_name = name_prefix + "_max_class_" + cost_func + target_str + '_' + str(n_images) + "_images_" + str(n_tries) + "_tries_try_" + str(n_try) +".png"
			cnn.get_logo(1, PFMs[n_try], 'cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/apa_array_v2/' + logo_name, 120, False, base_seq)

		logo_name = name_prefix + "_max_class_" + cost_func + target_str + '_' + str(n_images) + "_images_" + str(n_tries) + "_tries.png"
		cnn.get_logo(n_images, numpy.sum(PFMs, axis=0), 'cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/apa_array_v2/' + logo_name, 120, False, base_seq)
		
		mat_name = name_prefix + "_max_class_" + cost_func + target_str + '_' + str(n_images) + "_images_" + str(n_tries) + "_tries"
		numpy.save("cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/apa_array_v2/" + mat_name + "_final_pwms", PFMs)
		numpy.save("cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/apa_array_v2/" + mat_name + "_final_usages", y_hats)



def generate_max_layer2_models(base_seq, n_tries, n_image_list, n_epochs, cost_func='max_layer2_score', target=4.0) :

	num_filters = 110
	filter_width = 19

	rand_start_seqs = False

	filter_start_seqs = []
	for k in range(0, num_filters) :
		filter_start_seqs.append([])

	with open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/doubledope_pPas_aataaa_40000/avg_filter_highseqs.txt', 'r') as f_cons :
		for line in f_cons :
			line = line[:-1]

			lineparts = line.split('\t')

			k = int(lineparts[0].split('_')[1])
			start_seq = lineparts[3]

			filter_start_seqs[k].append(start_seq + ('.' * (185-filter_width)))


	for k in range(75, num_filters) :
		for n_images in n_image_list :
			print("n_images=" + str(n_images))

			if rand_start_seqs == True :
				start_seqs = None
			else :
				start_seqs = filter_start_seqs[k][-n_images:]
				print('Starting from following seqs: ')
				print([start_seq[:19] for start_seq in start_seqs])

			for n_try in range(0, n_tries) :
				print("n_try=" + str(n_try))
				
				train_set_y = numpy.matrix([[-target, target]])
				for i in range(0, n_images - 1) :
					train_set_y = numpy.vstack([train_set_y, numpy.matrix([[-target, target]])])


				train_set_y_t = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)

				train_set_L = numpy.zeros((n_images, 36))
				#train_set_L[:, 20] = 1
				train_set_L_t = theano.shared(numpy.asarray(train_set_L, dtype=theano.config.floatX), borrow=True)

				train_set_d = numpy.ones((n_images, 1))
				train_set_d_t = theano.shared(numpy.asarray(train_set_d, dtype=theano.config.floatX), borrow=True)
				
				
				cnn = DualCNN(
					n_images,
					base_seq,
					cost_func,
					k,#Layer filter
					train_set_y_t,
					train_set_L_t,
					train_set_d_t,
					learning_rate=0.5,
					drop=0.2,
					n_epochs=n_epochs,
					nkerns=[70, 110, 70],
					num_features=4,
					randomized_regions=[(0, 185), (185, 185)],
					dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
					name_prefix='max_class_' + cost_func,
					start_seqs=start_seqs
				)
				
				PFM = get_consensus(cnn.layer_input.outputs.eval()[0])
				for i in range(1, n_images) :
					#PFM = PFM + get_consensus(cnn.layer_input.outputs.eval()[i])
					PFM = PFM + numpy.array(cnn.layer_input.outputs.eval()[i])

				logo_name = "filter_" + str(k) + "_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + ".svg"
				cnn.get_logo(n_images, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/layer2/' + logo_name, filter_width, False, base_seq)

def generate_max_dense_neuron_models(base_seq, n_tries, n_image_list, n_epochs, cost_func='max_dense_score', target=4.0) :

	num_filters = 80

	for k in range(0, num_filters) :
		for n_images in n_image_list :
			print("n_images=" + str(n_images))
			for n_try in range(0, n_tries) :
				print("n_try=" + str(n_try))
				
				train_set_y = numpy.matrix([[-target, target]])
				for i in range(0, n_images - 1) :
					train_set_y = numpy.vstack([train_set_y, numpy.matrix([[-target, target]])])


				train_set_y_t = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)

				train_set_L = numpy.zeros((n_images, 36))
				#train_set_L[:, 20] = 1
				train_set_L_t = theano.shared(numpy.asarray(train_set_L, dtype=theano.config.floatX), borrow=True)

				train_set_d = numpy.ones((n_images, 1))
				train_set_d_t = theano.shared(numpy.asarray(train_set_d, dtype=theano.config.floatX), borrow=True)
				
				
				cnn = DualCNN(
					n_images,
					base_seq,
					cost_func,
					k,#Layer filter
					train_set_y_t,
					train_set_L_t,
					train_set_d_t,
					learning_rate=0.5,
					drop=0.2,
					n_epochs=n_epochs,
					nkerns=[70, 110, 70],
					#nkerns=[50, 90, 70],
					num_features=4,
					randomized_regions=[(0, 185), (185, 185)],
					dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
					#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_small',
					name_prefix='max_class_' + cost_func
				)

				y_hat = numpy.mean(numpy.ravel(cnn.predict()))
				
				PFM = get_consensus(cnn.layer_input.outputs.eval()[0])
				for i in range(1, n_images) :
					PFM = PFM + get_consensus(cnn.layer_input.outputs.eval()[i])

				logo_name = "filter_" + str(k) + "_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + "_avgprox_" + str(round(y_hat, 4)) + ".svg"
				cnn.get_logo(n_images, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/max_class/' + logo_name, 120, False, base_seq)

def visualize_cnn(dataset='general2'):

	#Layer 2 Logos
	#base_seq = 'NNNNNNNN.................................................................................................................................................................................'
	'''base_seq = 'NNNNNNNNNNNNNNNNNNN......................................................................................................................................................................'
	#base_seq = 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN..'
	generate_max_layer2_models(base_seq, 1, [20], n_epochs=4000, cost_func='max_layer2_score', target=4.0)'''

	#Dense Layer Logos
	'''n_images = 20
	#DoubleDope
	base_seq = '.........CATTACTCGCATCCANNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAGCCAATTAAGCCTGTCGTCGTGGGTGTCGAAAATGAAATAAAACAAGTCAATTGCGTAGTTTATTCAGACGTACCCCGTGGACCTACG'
	#Simple
	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_dense_neuron_models(base_seq, 1, [n_images], n_epochs=2000, cost_func='max_dense_score', target=4.0)'''

	#Complete
	#base_seq = '....NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN.................................................................'
	#generate_max_class_models(base_seq, 1, [20], n_epochs=2000, cost_func='max_score', target_prox_ratio=1.0, name_prefix='')


	#n_images = [20]#[20, 50]#20#50#100
	n_epochs = 6000#6000#3500#2000

	#DoubleDope
	#base_seq = '.........CATTACTCGCATCCANNNNNNNNNNNNNNNNNNNNNNNNNANNAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAGCCAATTAAGCCTGTCGTCGTGGGTGTCGAAAATGAAATAAAACAAGTCAATTGCGTAGTTTATTCAGACGTACCCCGTGGACCTACG'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='doubledope')

	#Simple
	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#base_seq = 'ATCTNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNAGTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple')
	generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple')
	'''

	#TOMM5
	#base_seq = 'TGCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCTAAAANANAAAACTATTTGGGAAGTATGAAACNNNNNNNNNNNNNNNNNNNNACCCTTATCCCTGTGACGTTTGGCCTCTGACAATACTGGTATAATTGTAAATAATGTCAAACTCCGTTTTCTAGCAAGTATTAAGGGA'
	#base_seq = 'TGCTNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNCTAAAATATAAAACTATTTGGGAAGTATGAAACNNNNNNNNNNNNNNNNNNNNACCCTTATCCCTGTGACGTTTGGCCTCTGACAATACTGGTATAATTGTAAATAATGTCAAACTCCGTTTTCTAGCAAGTATTAAGGGA'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='tomm5')


	#print('' + 1)

	n_images = [1]#[20, 50]#20#50#100
	n_epochs = 7000#6500#6000#3500#2000




	'''#CSTF penalty
	#DoubleDope
	base_seq = '.........CATTACTCGCATCCANNNNNNNNNNNNNNNNNNNNNNNNNANNAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAGCCAATTAAGCCTGTCGTCGTGGGTGTCGAAAATGAAATAAAACAAGTCAATTGCGTAGTTTATTCAGACGTACCCCGTGGACCTACG'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='doubledope')
	#generate_max_class_models_single_runs(base_seq, 2, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_cstf', target=1.0, name_prefix='doubledope')
	generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_aruns_cstf', target=1.0, name_prefix='doubledope')

	#Simple
	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#base_seq = 'ATCTNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNAGTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple')
	#generate_max_class_models_single_runs(base_seq, 3, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_aruns', target=4.0, name_prefix='simple')
	#generate_max_class_models_single_runs(base_seq, 2, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_cstf', target=1.0, name_prefix='simple')
	generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_aruns_cstf', target=1.0, name_prefix='simple')
	

	print('' + 1)'''






	#DoubleDope
	'''base_seq = '.........CATTACTCGCATCCANNNNNNNNNNNNNNNNNNNNNNNNNANNAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCAGCCAATTAAGCCTGTCGTCGTGGGTGTCGAAAATGAAATAAAACAAGTCAATTGCGTAGTTTATTCAGACGTACCCCGTGGACCTACG'
	#generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='doubledope')
	#generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='doubledope')
	generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_aruns', target=1.0, name_prefix='doubledope')
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.0, name_prefix='doubledope', lib_bias=20)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.25, name_prefix='doubledope', lib_bias=20)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.5, name_prefix='doubledope', lib_bias=20)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.75, name_prefix='doubledope', lib_bias=20)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=1.0, name_prefix='doubledope', lib_bias=20)

	#Simple
	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='simple')
	#generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='simple')
	generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_aruns', target=1.0, name_prefix='simple')
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.0, name_prefix='simple', lib_bias=22)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.25, name_prefix='simple', lib_bias=22)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.5, name_prefix='simple', lib_bias=22)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.75, name_prefix='simple', lib_bias=22)
	#generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=1.0, name_prefix='simple', lib_bias=22)

	

	#TOMM5
	base_seq = 'TGCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNCTAAAANANAAAACTATTTGGGAAGTATGAAACNNNNNNNNNNNNNNNNNNNNACCCTTATCCCTGTGACGTTTGGCCTCTGACAATACTGGTATAATTGTAAATAATGTCAAACTCCGTTTTCTAGCAAGTATTAAGGGA'
	#base_seq = 'TGCTNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNCTAAAATATAAAACTATTTGGGAAGTATGAAACNNNNNNNNNNNNNNNNNNNNACCCTTATCCCTGTGACGTTTGGCCTCTGACAATACTGGTATAATTGTAAATAATGTCAAACTCCGTTTTCTAGCAAGTATTAAGGGA'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='tomm5')
	#generate_max_class_models_single_runs(base_seq, 3, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_aruns', target=4.0, name_prefix='tomm5')
	generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='tomm5')
	generate_max_class_models_single_runs(base_seq, 20, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='tomm5')
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.0, name_prefix='tomm5', lib_bias=8)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.25, name_prefix='tomm5', lib_bias=8)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.5, name_prefix='tomm5', lib_bias=8)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.75, name_prefix='tomm5', lib_bias=8)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=1.0, name_prefix='tomm5', lib_bias=8)'''



	#AAR
	base_seq = 'ATCTCTGAGCTTTNNNNNNNNNNNNNNNNNNNNNNNNNTTGCTGCAGAGAATAAAAGGACCACGTGCAATACTTAATGCCGCATGATCNNNNNNNNNNNNNNNNNNNNNNNNNGGCTCTTTTGACAGCCTTTGGCGTCTGTAGAATAAATGCTGTGGCTCCTGCTGGCTGCTGTGGTGTTCACC.'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='aar')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='aar')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='aar')
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.0, name_prefix='aar', lib_bias=30)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.25, name_prefix='aar', lib_bias=30)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.5, name_prefix='aar', lib_bias=30)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.75, name_prefix='aar', lib_bias=30)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=1.0, name_prefix='aar', lib_bias=30)

	#ATR
	base_seq = 'TGCATTTGNNNNNNNNNNNNNNNNNNNNNNNNNCAATTCTAAAGTACAACATAAATTTACGTTCTCAGCAACTGTTATTTCTCNNNNNNNNNNNNNNNNNNNNNNNNNAATATACATTCAGTTATTAAGAAATAAACTGCTTTCTTAATACATACTGTGCATTATAATTGGAGAAATAGAATAT.'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='atr')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='atr')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='atr')
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.0, name_prefix='atr', lib_bias=31)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.25, name_prefix='atr', lib_bias=31)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.5, name_prefix='atr', lib_bias=31)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.75, name_prefix='atr', lib_bias=31)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=1.0, name_prefix='atr', lib_bias=31)

	#HSP
	base_seq = 'TCTTCTGAAATCTNNNNNNNNNNNNNNNNNNNNNNNNNTTCTCTTTTATAATAAACTAATGATAACTAATGACATCCAGTGTCTCCAANNNNNNNNNNNNNNNNNNNNNNNNNCACTTCCAAATAAAAATATGTAAATGAGTGGTTAATCTTTAGTTATTTTAAGATGATTTTAGGGTTTTGCT.'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='hsp')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='hsp')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='hsp')

	#SNH
	base_seq = 'ANNNNNNNNNNNNNNNNNNNNNNNNNTTACTTGATGTTGATAACATCACAATAAATTATGGAGAAAAATACATATTNNNNNNNNNNNNNNNNNNNNNNNANTAAAGTGTTTTCTTTTAAATCAACTCTAAATAGCTCCATTCTCATAGTCACTAGTCAGACCGCTCGCGCACTACTCAGCGACC.'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='snh')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='snh')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='snh')

	#SOX
	base_seq = 'CGATCTTCTTTTTTTANNNNNNNNNNNNNNNNNNNNNNNNNATTTTGTTAATAATAAGATAATGATGAGTAACTTAACCAGCACATTTCTCNNNNNNNNNNNNNNNNNNNNNNNNNGTTTTCTGATGACATAATAAAGACAGATCATTTCAGAATCTGGCCCTTGTGCAGGGGAGGAGGGAGGC.'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='sox')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='sox')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='sox')
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.0, name_prefix='sox', lib_bias=34)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.25, name_prefix='sox', lib_bias=34)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.5, name_prefix='sox', lib_bias=34)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=0.75, name_prefix='sox', lib_bias=34)
	generate_max_class_models_single_runs(base_seq, 5, [1], n_epochs=n_epochs, cost_func='target', target=1.0, name_prefix='sox', lib_bias=34)

	#WHA
	base_seq = 'CTTGAATTTCATNNNNNNNNNNNNNNNNNNNNNNNNNGTCATTTTGTCAAATAAATTCTGAAAATCTTTGTATTGACAGTGTGTTATNNNNNNNNNNNNNNNNNNNNNNNNNAGTGCTCAATAAAAAGAATAAAGAGGAAACAGCACTGGATCTATACCTATACAAAACAAGCTACCAGCGCTC.'
	#generate_max_class_models(base_seq, 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='wha')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_softer', target=1.0, name_prefix='wha')
	generate_max_class_models_single_runs(base_seq, 10, [1], n_epochs=n_epochs, cost_func='max_score_punish_cruns_harder', target=1.0, name_prefix='wha')
	




	


def get_consensus(input_seq) :
	for i in range(input_seq.shape[0]) :
		j = numpy.argmax(input_seq[i,:])
		if numpy.max(input_seq[i,:]) > 0 :
			input_seq[i,:] = 0
			input_seq[i,j] = 1
		else :
			input_seq[i,:] = 0
	return input_seq

def translate_to_seq(X_point) :

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
		else :
			seq += "N"
	return seq

if __name__ == '__main__':
	visualize_cnn('general2')
