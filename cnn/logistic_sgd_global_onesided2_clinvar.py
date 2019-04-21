"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
				&= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

	- textbooks: "Pattern Recognition and Machine Learning" -
				 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy
import scipy.sparse as sp
import scipy.io as spio

import theano
import theano.tensor as T

class LogisticRegression(object):
	"""Multi-class Logistic Regression Class

	The logistic regression is fully described by a weight matrix :math:`W`
	and bias vector :math:`b`. Classification is done by projecting data
	points onto a set of hyperplanes, the distance to which is used to
	determine a class membership probability.
	"""
	
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

	def __init__(self, input, n_in, n_out, load_model = False, w_file = '', b_file = ''):
		""" Initialize the parameters of the logistic regression

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
					  architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
					  which the labels lie

		"""
		# start-snippet-1
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		
		self.w_file = w_file
		self.b_file = b_file
		
		if load_model == False :
			self.W = theano.shared(
				value=numpy.zeros(
					(n_in, n_out),
					dtype=theano.config.floatX
				),
				name='W',
				borrow=True
			)
			# initialize the baises b as a vector of n_out 0s
			self.b = theano.shared(
				value=numpy.zeros(
					(n_out,),
					dtype=theano.config.floatX
				),
				name='b',
				borrow=True
			)
		else :
			self.W = theano.shared(
				value=self.load_w(w_file + '.npy'),
				name='W',
				borrow=True
			)
			# initialize the baises b as a vector of n_out 0s
			self.b = theano.shared(
				value=self.load_b(b_file + '.npy'),
				name='b',
				borrow=True
			)

		# symbolic expression for computing the matrix of class-membership
		# probabilities
		# Where:
		# W is a matrix where column-k represent the separation hyper plain for
		# class-k
		# x is a matrix where row-j  represents input training sample-j
		# b is a vector where element-k represent the free parameter of hyper
		# plain-k
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		self.s_y_given_x = T.dot(input, self.W) + self.b

		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		#self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		# end-snippet-1

		# parameters of the model
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.

		.. math::

			\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
			\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
				\log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
			\ell (\theta=\{W,b\}, \mathcal{D})

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label

		Note: we use the mean instead of the sum so that
			  the learning rate is less dependent on the batch size
		"""
		# start-snippet-2
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		
		return -T.mean(T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])])
		
		#return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		# end-snippet-2

	def rsquare(self, y):
		return 1.0 - (T.dot(y[:,1] - self.p_y_given_x[:,1], y[:,1] - self.p_y_given_x[:,1]) / T.dot(y[:,1] - T.mean(y[:,1]), y[:,1] - T.mean(y[:,1])))
	
	def sse(self, y):
		return T.dot(y[:,1] - self.p_y_given_x[:,1], y[:,1] - self.p_y_given_x[:,1])
	
	def sst(self, y):
		return T.dot(y[:,1] - T.mean(y[:,1]), y[:,1] - T.mean(y[:,1]))

	def multisse(self, y, j):
		return T.dot(y[:,j] - self.p_y_given_x[:,j], y[:,j] - self.p_y_given_x[:,j])
	
	def multisst(self, y, j):
		return T.dot(y[:,j] - T.mean(y[:,j]), y[:,j] - T.mean(y[:,j]))
	
	def abs_error(self, y):
		return T.mean(abs(y[:,1] - self.p_y_given_x[:,1]))
	
	def recall(self, j=1):
		return self.p_y_given_x[:,j]

	def log_loss(self, y):
		return -T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])]


def shared_dataset(data, datatype=theano.config.floatX, borrow=True):
		""" Function that loads the dataset into shared variables"""
		
		shared_data = theano.shared(numpy.asarray(data,
											   dtype=datatype),
								 borrow=borrow)

		return shared_data

def load_output_data(dataset):#,1,2
	
	y_ref = numpy.matrix(numpy.load('snps/apa_' + dataset + '_output_ref.npy'))
	y_ref = numpy.hstack([numpy.matrix(1.0 - numpy.sum(y_ref[:, 0],axis=1)), y_ref[:, 0]])

	snptype = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snptype.npy')[:,0])
	snpregion = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snpregion.npy')[:,0])
	snpsign = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snpsign.npy')[:,0])

	'''#snpregion_prune_index = (snpregion == 2)
	snpregion_prune_index = ((snpregion == 1) | (snpregion == 4)) | (snpregion == 2)
	#snpregion_prune_index = ((snpregion == 1) | (snpregion == 5)) | (snpregion == 4)
	#snpregion_prune_index = (snpregion == 5)
	#snpregion_prune_index = (snpregion == 2) | (snpregion == 3)
	#snpregion_prune_index = ((snpregion == 1) | (snpregion == 2)) | (snpregion == 4)
	y_ref = y_ref[snpregion_prune_index,:]
	snpregion = snpregion[snpregion_prune_index]
	snptype = snptype[snpregion_prune_index]
	snpsign = snpsign[snpregion_prune_index]'''

	'''#snpsign_prune_index = (snpsign == 3) | (snpsign == 6)
	#snpsign_prune_index = (snpsign == 3) | (snpsign == 4)
	#snpsign_prune_index = (snpsign == 1) | (snpsign == 6)
	
	#snpsign_prune_index = (snpsign == 6) #| (snpsign == 5)
	#snpsign_prune_index = (snpsign == 1) | (snpsign == 6)
	#snpsign_prune_index = ((snpsign == 3) | (snpsign == 4)) | (snpsign == 5)
	#snpsign_prune_index = (((snpsign == 1) | (snpsign == 6)) | (snpsign == 3))# | (snpsign == 5)
	snpsign_prune_index = (snpsign == 6)
	y_ref = y_ref[snpsign_prune_index,:]
	snpregion = snpregion[snpsign_prune_index]
	snptype = snptype[snpsign_prune_index]
	snpsign = snpsign[snpsign_prune_index]'''


	print('Number of benign samples = ' + str(len(snpsign[snpsign == 1])))
	print('Number of pathogenic samples = ' + str(len(snpsign[snpsign == 6])))

	print('Number of likely benign samples = ' + str(len(snpsign[snpsign == 2])))
	print('Number of likely pathogenic samples = ' + str(len(snpsign[snpsign == 5])))


	ref_y = theano.shared(numpy.asarray(y_ref, dtype=theano.config.floatX), borrow=True)
	snpsign = theano.shared(numpy.asarray(snpsign, dtype=theano.config.floatX), borrow=True)
	
	rval = [ref_y, snpsign]
	return rval

def load_input_data(dataset):
	#############
	# LOAD DATA #
	#############

	print('... loading data')
	
	#X = numpy.load(dataset + '_input.npy')#_foldprob
	X = spio.loadmat('snps/apa_fullseq_' + dataset + '_input.mat')
	X_ref = sp.csr_matrix(X["ref"])
	X_var = sp.csr_matrix(X["var"])

	snptype = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snptype.npy')[:,0])
	snpregion = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snpregion.npy')[:,0])
	snpsign = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snpsign.npy')[:,0])
	apadist = numpy.ravel(numpy.load('snps/apa_' + dataset + '_apadist.npy')[:,0])
	gene = numpy.ravel(numpy.load('snps/apa_' + dataset + '_gene.npy'))
	ids = numpy.ravel(numpy.load('snps/apa_' + dataset + '_id.npy'))

	#snpregion_prune_index = (snpregion == 2)
	'''snpregion_prune_index = ((snpregion == 1) | (snpregion == 4)) | (snpregion == 2)
	#snpregion_prune_index = (snpregion == 5)
	#snpregion_prune_index = ((snpregion == 1) | (snpregion == 5)) | (snpregion == 4)
	#snpregion_prune_index = (snpregion == 2) | (snpregion == 3)
	#snpregion_prune_index = ((snpregion == 1) | (snpregion == 2)) | (snpregion == 4)
	X_ref = X_ref[snpregion_prune_index,:]
	X_var = X_var[snpregion_prune_index,:]
	snpregion = snpregion[snpregion_prune_index]
	snptype = snptype[snpregion_prune_index]
	apadist = apadist[snpregion_prune_index]
	snpsign = snpsign[snpregion_prune_index]
	gene = gene[snpregion_prune_index]
	ids = ids[snpregion_prune_index]'''

	#snpsign_prune_index = (snpsign == 3) | (snpsign == 6)
	#snpsign_prune_index = (snpsign == 3)
	#snpsign_prune_index = (snpsign == 3) | (snpsign == 4)
	#snpsign_prune_index = (snpsign == 1) | (snpsign == 6)
	
	#snpsign_prune_index = (snpsign == 6) #| (snpsign == 5)
	#snpsign_prune_index = (snpsign == 1) | (snpsign == 6)
	'''#snpsign_prune_index = (((snpsign == 1) | (snpsign == 6)) | (snpsign == 3))# | (snpsign == 5)
	#snpsign_prune_index = ((snpsign == 3) | (snpsign == 4)) | (snpsign == 5)
	snpsign_prune_index = (snpsign == 6)
	X_ref = X_ref[snpsign_prune_index,:]
	X_var = X_var[snpsign_prune_index,:]
	snpregion = snpregion[snpsign_prune_index]
	snptype = snptype[snpsign_prune_index]
	apadist = apadist[snpsign_prune_index]
	snpsign = snpsign[snpsign_prune_index]
	gene = gene[snpsign_prune_index]
	ids = ids[snpsign_prune_index]'''

	ref_x = theano.shared(X_ref, borrow=True)
	var_x = theano.shared(X_var, borrow=True)

	L_input = numpy.zeros((X_ref.shape[0], 36))
	zero_L = theano.shared(numpy.asarray(L_input, dtype=theano.config.floatX), borrow=True)
	
	rval = [ref_x, var_x, zero_L, apadist, gene, snpregion, snptype, ids]
	return rval

def translate_to_seq(x) :
	X_point = numpy.ravel(x.todense())
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
		else :
			seq += "."
	return seq

if __name__ == '__main__':
	datasets = load_data()
	#sgd_optimization_mnist()
