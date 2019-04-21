
"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy
import math

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import theano.sparse as Tsparse

from logistic_sgd_global_onesided2_cuts_vis import LogisticRegression, TrainableImage
from mlp import HiddenLayer

#import pylab as pl
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.sparse as sp
import scipy.io as spio

import weblogolib

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
#sys.path.append("/usr/local/lib/python3.5/site-packages")
import RNA

from scipy.stats import pearsonr

class ModifiedBackprop(object):

	def __init__(self, nonlinearity):
		self.nonlinearity = nonlinearity
		self.ops = {}  # memoizes an OpFromGraph instance per tensor type

	def __call__(self, x):
		# OpFromGraph is oblique to Theano optimizations, so we need to move
		# things to GPU ourselves if needed.
		'''if theano.sandbox.cuda.cuda_enabled:
			maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
		else:
			maybe_to_gpu = lambda x: x'''
		# We move the input to GPU if needed.
		#x = maybe_to_gpu(x)
		# We note the tensor type of the input variable to the nonlinearity
		# (mainly dimensionality and dtype); we need to create a fitting Op.
		tensor_type = x.type
		# If we did not create a suitable Op yet, this is the time to do so.
		if tensor_type not in self.ops:
			# For the graph, we create an input variable of the correct type:
			inp = tensor_type()
			# We pass it through the nonlinearity (and move to GPU if needed).
			outp = self.nonlinearity(inp)#maybe_to_gpu(self.nonlinearity(inp))
			# Then we fix the forward expression...
			op = theano.OpFromGraph([inp], [outp])
			# ...and replace the gradient with our own (defined in a subclass).
			op.grad = self.grad
			# Finally, we memoize the new Op
			self.ops[tensor_type] = op
		# And apply the memoized Op to the input we got.
		return self.ops[tensor_type](x)

class GuidedBackprop(ModifiedBackprop):
	def grad(self, inputs, out_grads):
		(inp,) = inputs
		(grd,) = out_grads
		dtype = inp.dtype
		return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)

class ZeilerBackprop(ModifiedBackprop):
	def grad(self, inputs, out_grads):
		(inp,) = inputs
		(grd,) = out_grads
		#return (grd * (grd > 0).astype(inp.dtype),)  # explicitly rectify
		return (self.nonlinearity(grd),)

class LeNetConvPoolLayer(object):
	"""Pool Layer of a convolutional network """

	def store_w(self, w_file, W):
		numpy.save(w_file, W)
	
	def store_b(self, b_file, b):
		numpy.save(b_file, b)
	
	def store_model(self, W, b):
		self.store_w(self.store_as_w_file, W)
		self.store_b(self.store_as_b_file, b)
	
	def load_w(self, w_file):
		return numpy.load(w_file)
	
	def load_b(self, b_file):
		return numpy.load(b_file)
	
	def __init__(self, rng, input, deactivated_filter, deactivated_output, filter_shape, image_shape, poolsize=(2, 2), stride=(1, 1), activation_fn=T.tanh, load_model = False, w_file = '', b_file = '', store_as_w_file = None, store_as_b_file = None):
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
		self.store_as_w_file = w_file
		self.store_as_b_file = b_file
		if store_as_w_file is not None and store_as_b_file is not None :
			self.store_as_w_file = store_as_w_file
			self.store_as_b_file = store_as_b_file
		
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
			subsample=stride,
			image_shape=image_shape
		)

		'''if(use_relu == True):
			activation = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		else:
			activation = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))'''
		activation = activation_fn(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

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
		
		self.output = pooled_out
		
		# store parameters of this layer
		self.params = [self.W, self.b]

def relu(x):
    return T.switch(x<0, 0, x)

class DualCNN(object):

	def get_logo(self, k, PFM, file_path='cnn_motif_analysis/fullseq_v/', seq_length=6, normalize=False, base_seq='', output_fmt='svg') :

		if normalize == True :
			for i in range(0, PFM.shape[0]) :
				if numpy.sum(PFM[i, :]) > 0 :
					PFM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])
				#PFM[i, :] *= 10000.0
			#print(PFM)
		

		#Create weblogo from API
		logo_output_format = output_fmt#"svg"
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

	def get_fold_logo(self, k, PFM, file_path='cnn_motif_analysis/fullseq_v/', seq_length=6, output_fmt='svg') :

		#Create weblogo from API
		logo_output_format = output_fmt#"svg"
		#Load data from an occurence matrix
		data = weblogolib.LogoData.from_counts('LIR', PFM[:seq_length, :])

		#Generate color scheme
		color_rules = []
		color_rules.append(weblogolib.SymbolColor("L", "green"))
		color_rules.append(weblogolib.SymbolColor("I", "blue"))
		color_rules.append(weblogolib.SymbolColor("R", "green"))
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


	def __init__(self, n_images, base_seq, cost_func, cut_pos, train_set_y, train_set_L, train_set_d, learning_rate=0.1, drop=0, n_epochs=30, nkerns=[30, 40, 50], num_features=4, randomized_regions=[(2, 37), (45, 80)], dataset='default', cell_line='default', name_prefix='default', start_seqs=None):
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
			deactivated_filter=None,
			deactivated_output=None,
			image_shape=(n_images, 1, left_random_size, num_features),
			filter_shape=(nkerns[0], 1, 8, num_features),
			poolsize=(2, 1),
			stride=(1, 1),
			activation_fn=relu
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
			deactivated_filter=None,
			deactivated_output=None,
			image_shape=(n_images, nkerns[0], 89, 1),
			filter_shape=(nkerns[1], nkerns[0], 6, 1),
			poolsize=(1, 1),
			stride=(1, 1),
			activation_fn=relu
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
			#srng,
			input=layer3_input,
			n_in=nkerns[1] * (84) * 1 + 1,#n_in=nkerns[1] * (169) * 1 + 1,
			n_out=200,
			activation=relu
			,load_model = True,
			w_file='model_store/' + dataset + '_' + cell_line + '_mlp_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_mlp_b'
		)

		layer3_output = layer3.output

		train_drop = T.lscalar()
		self.train_drop = train_drop
		if drop != 0 :
			print('Using dropout = ' + str(drop))
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = train_drop)

		layer4_input = T.concatenate([layer3_output, L_input], axis=1)
		#layer4_input = layer3.output

		# classify the values of the fully-connected sigmoidal layer
		layer4 = LogisticRegression(self, image=layer0_input, input=layer4_input, L_input=L_input, n_in=200 + 36, n_out=self.input_size + 1, load_model = True, cost_func=cost_func, cost_layer_filter=cut_pos,
			w_file='model_store/' + dataset + '_' + cell_line + '_lr_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_lr_b')

		self.layer0_left = layer0_left
		#self.layer0_right = layer0_right
		self.layer1 = layer1
		#self.layer2 = layer2
		self.layer3 = layer3
		self.output_layer = layer4

		epoch_i = T.lscalar()
		
		# the cost we minimize during training is the NLL of the model
		cost = layer4.negative_log_likelihood(y, epoch_i)
		
		# create a list of all model parameters to be fit by gradient descent
		params = layer_input.params
		#params = layer4.params + layer3.params + layer1.params + layer0_left.params# + layer0_right.params
		#params = layer3.params + layer2.params + layer0_left.params + layer0_right.params
		
		# create a list of gradients for all model parameters
		grads = T.grad(cost, params)
		
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

		cuts = numpy.zeros((n_epochs + 1, n_images, 185))

		pwms[0, :, :, :] = numpy.array(self.layer_input.outputs.eval())

		cuts[0, :, :] = numpy.array(self.predict())[:, :185]

		while (epoch < n_epochs) :
			epoch = epoch + 1

			cost_ij = train_model(epoch)

			y_hat = numpy.ravel(numpy.sum(self.predict()[:, cut_pos], axis=1))
			logodds_hat = numpy.log(y_hat / (1 - y_hat))
			logodds_series[:, epoch-1] = logodds_hat

			pwms[epoch, :, :, :] = numpy.array(self.layer_input.outputs.eval())

			cuts[epoch, :, :] = numpy.array(self.predict())[:, :185]

			if epoch % 100 == 0 :
				print('training @ epoch = ', epoch)
				print(cost_ij)
				print('logodds_hat mean = ' + str(numpy.mean(logodds_hat)))


		end_time = time.clock()

		'''numpy.save("cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/" + name_prefix + "_pwms", pwms)
		numpy.save("cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/" + name_prefix + "_cuts", cuts)

		if cost_func in ['target', 'max_score', 'max_score_GGCC'] :
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

			plt.savefig("cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/" + name_prefix + "_trainingplot.svg")
			#plt.show()
			plt.close()

			f = plt.figure()
			plt.plot(numpy.arange(185), cuts[-1, 0, :], alpha=0.7)

			plt.savefig("cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/" + name_prefix + "_endcutplot.svg")
			#plt.show()
			plt.close()'''


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def generate_max_class_models(base_seq, cut_pos, n_tries, n_image_list, n_epochs, cost_func='max_score', target=4.0, name_prefix='', start_seqs=None) :

	for n_images in n_image_list :
		print("n_images=" + str(n_images))
		f = open('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/multi_run/' + name_prefix + "_predictions_" + cost_func + '_' + str(n_images) + "_images.txt",'w')
		
		PFMs = numpy.zeros((n_tries, 185, 4))
		structures = []
		structures_ext = []
		mfes = []
		mfes_ext = []
		y_hats = []

		for n_try in range(0, n_tries) :
			print("n_try=" + str(n_try))
			f.write('n_try: ' + str(n_try) + '\n')
			
			train_set_y = numpy.zeros((1, 186))
			train_set_y[0, 75] = 1
			for i in range(0, n_images - 1) :
				train_set_y = numpy.vstack([train_set_y, numpy.zeros((1, 186))])
				train_set_y[-1, 75] = 1


			train_set_y_t = theano.shared(numpy.asarray(train_set_y, dtype=theano.config.floatX), borrow=True)

			train_set_L = numpy.zeros((n_images, 36))
			train_set_L[:, 22] = 1
			train_set_L_t = theano.shared(numpy.asarray(train_set_L, dtype=theano.config.floatX), borrow=True)

			train_set_d = numpy.zeros((n_images, 1))
			train_set_d_t = theano.shared(numpy.asarray(train_set_d, dtype=theano.config.floatX), borrow=True)
			
			
			cnn = DualCNN(
				n_images,
				base_seq,
				cost_func,
				cut_pos,#Layer filter
				train_set_y_t,
				train_set_L_t,
				train_set_d_t,
				learning_rate=0.2,
				drop=0.2,
				n_epochs=n_epochs,
				nkerns=[70, 110, 70],
				num_features=4,
				randomized_regions=[(0, 185), (185, 185)],
				dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned',#_pasaligned
				name_prefix=name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try),
				start_seqs=start_seqs
			)

			y_hat = cnn.predict()

			#PFM = get_consensus(cnn.layer_input.outputs.eval()[0])
			PFM = numpy.array(cnn.layer_input.outputs.eval()[0])

			f.write('y_hat: ' + str(y_hat[0]) + '\n')
			print(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[0])))
			f.write(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[0])) + '\n')


			end_cut_pos = cut_pos[-1]

			seq = translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[0]))
			dse_seq = seq[55:end_cut_pos]
			dse_seq_ext = seq[40:end_cut_pos]
			(structure, mfe) = RNA.fold(dse_seq)
			(structure_ext, mfe_ext) = RNA.fold(dse_seq_ext)

			PFM_fold = numpy.zeros((len(structure), 3))
			for j in range(0, len(structure)) :
				if structure[j] == '(' :
					PFM_fold[j, 0] += 1
				elif structure[j] == '.' :
					PFM_fold[j, 1] += 1
				elif structure[j] == ')' :
					PFM_fold[j, 2] += 1
			PFM_fold_ext = numpy.zeros((len(structure_ext), 3))
			for j in range(0, len(structure_ext)) :
				if structure_ext[j] == '(' :
					PFM_fold_ext[j, 0] += 1
				elif structure_ext[j] == '.' :
					PFM_fold_ext[j, 1] += 1
				elif structure_ext[j] == ')' :
					PFM_fold_ext[j, 2] += 1


			print('mfe: ' + str(mfe))
			print('mfe_ext: ' + str(mfe_ext))
			print('fold: ' + structure)
			print('foex: ' + structure_ext)
			f.write(structure + ' ' + str(mfe) + '\n')
			f.write(structure_ext + ' ' + str(mfe_ext) + '\n')

			PFMs[n_try * n_images:(n_try + 1) * n_images, :, :] = numpy.array(cnn.layer_input.outputs.eval())
			structures.append(structure)
			structures_ext.append(structure_ext)
			mfes.append(mfe)
			mfes_ext.append(mfe_ext)
			y_hats.append(y_hat)

			for i in range(1, n_images) :
				#PFM = PFM + get_consensus(cnn.layer_input.outputs.eval()[i])
				PFM = PFM + numpy.array(cnn.layer_input.outputs.eval()[i])

				f.write('y_hat: ' + str(y_hat[i]) + '\n')
				print(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[i])))
				f.write(translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[i])) + '\n')


				seq = translate_to_seq(get_consensus(cnn.layer_input.outputs.eval()[i]))
				dse_seq = seq[55:end_cut_pos]
				dse_seq_ext = seq[40:end_cut_pos]
				(structure, mfe) = RNA.fold(dse_seq)
				(structure_ext, mfe_ext) = RNA.fold(dse_seq_ext)

				for j in range(0, len(structure)) :
					if structure[j] == '(' :
						PFM_fold[j, 0] += 1
					elif structure[j] == '.' :
						PFM_fold[j, 1] += 1
					elif structure[j] == ')' :
						PFM_fold[j, 2] += 1
				for j in range(0, len(structure_ext)) :
					if structure_ext[j] == '(' :
						PFM_fold_ext[j, 0] += 1
					elif structure_ext[j] == '.' :
						PFM_fold_ext[j, 1] += 1
					elif structure_ext[j] == ')' :
						PFM_fold_ext[j, 2] += 1

				print('mfe: ' + str(mfe))
				print('mfe_ext: ' + str(mfe_ext))
				print('fold: ' + structure)
				print('foex: ' + structure_ext)
				f.write(structure + ' ' + str(mfe) + '\n')
				f.write(structure_ext + ' ' + str(mfe_ext) + '\n')

				structures.append(structure)
				structures_ext.append(structure_ext)
				mfes.append(mfe)
				mfes_ext.append(mfe_ext)


			f.write('\n')

			logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + ".png"
			cnn.get_logo(n_images, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/multi_run/' + logo_name, 120, False, base_seq, output_fmt='png')

			'''logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + ".svg"
			cnn.get_logo(n_images, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/' + logo_name, 120, False, base_seq)

			#logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + "_fold.svg"
			#cnn.get_fold_logo(n_images, PFM_fold, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/' + logo_name, PFM_fold.shape[0])

			#logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + "_fold_ext.svg"
			#cnn.get_fold_logo(n_images, PFM_fold_ext, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/' + logo_name, PFM_fold_ext.shape[0])

			logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + ".eps"
			cnn.get_logo(n_images, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/' + logo_name, 120, False, base_seq, output_fmt='eps')

			logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + "_fold.eps"
			cnn.get_fold_logo(n_images, PFM_fold, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/' + logo_name, PFM_fold.shape[0], output_fmt='eps')

			logo_name = name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_try_" + str(n_try) + "_fold_ext.eps"
			cnn.get_fold_logo(n_images, PFM_fold_ext, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/single_run/' + logo_name, PFM_fold_ext.shape[0], output_fmt='eps')
			'''

		
		structures = numpy.array(structures, dtype=object)
		structures_ext = numpy.array(structures_ext, dtype=object)
		mfes = numpy.array(mfes)
		mfes_ext = numpy.array(mfes_ext)
		mfes_ext = numpy.array(mfes_ext)
		y_hats = numpy.array(numpy.concatenate(y_hats, axis=0))

		save_pathh = 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/max_class/multi_run/'
		numpy.save(save_pathh + name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_pwm", PFMs)
		numpy.save(save_pathh + name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_cuthat", y_hats)
		#numpy.save(save_pathh + name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_structure", structures)
		#numpy.save(save_pathh + name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_structure_ext", structures_ext)
		#numpy.save(save_pathh + name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_mfe", mfes)
		#numpy.save(save_pathh + name_prefix + "_max_class_" + cost_func + '_' + str(n_images) + "_images_mfe_ext", mfes_ext)

		f.close()


def visualize_cnn():

	n_epochs = 6000#5000#2000#6000

	#Simple
	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'



	#cuts A GGCC-reward punish A-runs in DSE

	'''for n_tries in [20] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_60_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_65_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_70_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_75_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_80_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_85_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_90_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_95_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_100_GGCC_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns', target=4.0, name_prefix='simple_105_GGCC_punish_aruns')
	'''

	for n_tries in [10] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_60_punish_aruns')

		print('' + 1)

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_65_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_70_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_75_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_80_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_85_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_90_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_95_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_100_punish_aruns')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns', target=4.0, name_prefix='simple_105_punish_aruns')
	




	#Low-entropy punished runs


	#cuts A GGCC-reward

	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_60_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_65_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_70_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_75_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_80_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_85_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_90_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_95_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_100_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_105_GGCC_ent')


	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_60_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_65_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_70_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_75_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_80_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_85_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_90_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_95_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_100_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_105_ent')
	

	#cuts AT GGCC-reward

	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_60_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_65_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_70_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_75_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_80_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_85_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_90_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_95_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_100_AT_GGCC_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_ent', target=4.0, name_prefix='simple_105_AT_GGCC_ent')
		



	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_60_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_65_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_70_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_75_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_80_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_85_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_90_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_95_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_100_AT_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_ent', target=4.0, name_prefix='simple_105_AT_ent')



	#cuts A GGCC-reward punish A-runs in DSE

	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_60_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_65_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_70_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_75_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_80_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_85_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_90_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_95_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_100_GGCC_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC_punish_aruns_ent', target=4.0, name_prefix='simple_105_GGCC_punish_aruns_ent')


	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_60_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_65_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_70_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_75_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_80_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_85_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_90_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_95_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_100_punish_aruns_ent')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_punish_aruns_ent', target=4.0, name_prefix='simple_105_punish_aruns_ent')





	#No entropy penalty




	#cuts A GGCC-reward

	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_GGCC')


	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105')
	

	#cuts AT GGCC-reward

	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_AT_GGCC')
		



	for n_tries in [4] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105_AT')





	print('' + 1)


	#cuts A GGCC-reward

	for n_tries in [20] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_GGCC')


	for n_tries in [20] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105')
	

	#cuts AT GGCC-reward

	for n_tries in [20] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_AT_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_AT_GGCC')
		



	for n_tries in [20] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100_AT')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105_AT')





	print('' + 1)


	#cuts free GGCC-reward

	for n_tries in [20] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_free_GGCC')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_free_GGCC')


	for n_tries in [20] :
		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100_free')

		base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
		generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105_free')


def visualize_cnn_single():

	n_epochs = 6000#2000#6000

	#Simple
	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'

	#cuts A GGCC-reward

	n_tries = 1

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_GGCC')


	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105')

	#cuts AT GGCC-reward

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_AT_GGCC')
	
	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_AT_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_AT_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_AT_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_AT_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_AT_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_AT_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_AT_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_AT_GGCC')

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_AT_GGCC')
	
	print('' + 1)

	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100_AT')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105_AT')
	'''

	#cuts free GGCC-reward

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_60_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_65_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_70_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_75_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_80_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_85_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_90_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_95_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_100_free_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score_GGCC', target=4.0, name_prefix='simple_105_free_GGCC')


	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [59, 60], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [64, 65], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [69, 70], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [74, 75], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [79, 80], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [89, 90], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [94, 95], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [99, 100], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_100_free')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [104, 105], n_tries, [1], n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105_free')














def visualize_cnn_bak():

	n_images = [5]
	n_epochs = 2000#6000

	#Simple
	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANTAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAAGTCCTGCCCGGTCGGCTTGAGTGCGTGTGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	

	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')'''

	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [69, 70], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [79, 80], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [89, 90], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_90_GGCC')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [94, 95], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_95_GGCC')'''

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [98, 99], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_99_GGCC', start_seqs=None)

	#base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#generate_max_class_models(base_seq, [104, 105], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_105')


	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [59, 60], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [64, 65], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [69, 70], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [74, 75], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [79, 80], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')'''

	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [59, 60], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [64, 65], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [69, 70], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [74, 75], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [79, 80], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')'''

	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [59, 60], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [64, 65], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [69, 70], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [74, 75], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [79, 80], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [84, 85], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')'''

	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [60], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [65], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [70], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [75], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [80], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [85], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')'''

	'''base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [60], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_60')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [65], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_65')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [70], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_70')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [75], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_75')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [80], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_80')

	base_seq = 'ATCTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAATAAANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNATNNNNNNNNNNNNNNNNNNTGTCCTGCCCGGTCGGCTTGAGCGCGTGGGTCTCGTTTAGATGCTGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	generate_max_class_models(base_seq, [85], 1, n_images, n_epochs=n_epochs, cost_func='max_score', target=4.0, name_prefix='simple_85')'''


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
	visualize_cnn()