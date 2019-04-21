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
		self.store_w(self.store_as_w_file, W)
		self.store_b(self.store_as_b_file, b)
	
	def load_w(self, w_file):
		return numpy.load(w_file)
	
	def load_b(self, b_file):
		return numpy.load(b_file)

	def __init__(self, input, n_in, n_out, load_model = False, w_file = '', b_file = '', store_as_w_file = None, store_as_b_file = None):
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
		self.store_as_w_file = w_file
		self.store_as_b_file = b_file
		if store_as_w_file is not None and store_as_b_file is not None :
			self.store_as_w_file = store_as_w_file
			self.store_as_b_file = store_as_b_file

		
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
	
	def abs_error(self, y):
		return T.mean(abs(y[:,1] - self.p_y_given_x[:,1]))
	
	def recall(self):
		return self.p_y_given_x[:,1]

	def log_loss(self, y):
		return -T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])]


def shared_dataset(data, datatype=theano.config.floatX, borrow=True):
		""" Function that loads the dataset into shared variables"""
		
		shared_data = theano.shared(numpy.asarray(data,
											   dtype=datatype),
								 borrow=borrow)

		return shared_data

def load_output_data(dataset, L_included_list=None, constant_test_set_size_param=None, splice_site_indexes=[1], shuffle_index=None, shuffle_all_index=None, balance_data=False, balance_test_set=False, count_filter=None, misprime_index=None, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=False):#,1,2
	
	train_set_split = 0.95#0.96 #0.96 #0.95
	valid_set_split = 0.02#0.02 #0.02 #0.02

	constant_set_split = True
	constant_valid_set_split = 1000
	constant_test_set_split = 40000#80000#40000

	if constant_test_set_size_param is not None :
		constant_set_split = True
		constant_test_set_split = constant_test_set_size_param

	y = numpy.matrix(numpy.load('apa_' + dataset + '_output.npy'))#NOT .T
	
	y = numpy.hstack([numpy.matrix(1.0 - numpy.sum(y[:, splice_site_indexes],axis=1)), y[:, splice_site_indexes]])
	
	print(y.shape)
	print(y)
	
	c = numpy.ravel(numpy.load('apa_' + dataset + '_count.npy'))

	L = numpy.ravel(numpy.load('apa_' + dataset + '_libindex.npy'))

	#L_prune_index = (L != 13) & (L != 14)
	#L_prune_index = L == 5
	#L_prune_index = L >= 20
	#L_prune_index = L <= 5
	#L_prune_index = (L >= 5) & (L <= 12)
	'''y = y[L_prune_index,:]
	c = c[L_prune_index]
	L = L[L_prune_index]'''

	'''sort_index = numpy.argsort(c)
	y = y[sort_index,:]
	c = c[sort_index]
	L = L[sort_index]'''

	'''if shuffle_index is not None :
		y = y[shuffle_index,:]
		c = c[shuffle_index]'''

	#Only non-competing PASes
	Floader = numpy.load('npz_apa_seq_' + dataset + '_features.npz')
	F = sp.csr_matrix((Floader['data'], Floader['indices'], Floader['indptr']), shape = Floader['shape'], dtype=numpy.int8)
	
	F_aataaa = numpy.ravel(F[:, 0].todense())
	F_attaaa = numpy.ravel(F[:, 1].todense())
	F_denovo = numpy.ravel(F[:, 5].todense())

	F_dpas_aataaa = numpy.ravel(F[:, 2].todense())
	F_dpas_attaaa = numpy.ravel(F[:, 3].todense())

	y = y[(F_aataaa == 1) & (F_dpas_aataaa == 0),:]
	L = L[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
	c = c[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
	#y = y[F_aataaa == 1,:]
	#L = L[F_aataaa == 1]
	#c = c[F_aataaa == 1]


	'''if count_filter is not None :
		y = y[c > count_filter,:]
		L = L[c > count_filter]
		c = c[c > count_filter]'''

	if balance_all_libraries == True :

		#List of included libraries
		L_included = [20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[20]#[2, 5, 8, 11]#[22]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[20]#[2, 5, 8, 11, 20]#, 22]#[20, 22]
		if L_included_list is not None :
			L_included = L_included_list

		arranged_index_len = 0
		min_join_len = len(numpy.nonzero(L == L_included[0])[0])
		if count_filter is not None :
			min_join_len = len(numpy.nonzero((L == L_included[0]) & (c > count_filter))[0])

		for lib in L_included :
			lib_len = len(numpy.nonzero(L == lib)[0])
			if count_filter is not None :
				lib_len = len(numpy.nonzero((L == lib) & (c > count_filter))[0])

			arranged_index_len += lib_len
			if lib_len < min_join_len :
				min_join_len = lib_len

		arranged_index = numpy.zeros(arranged_index_len, dtype=numpy.int)

		arranged_remainder_index = 0
		arranged_join_index = arranged_index_len - len(L_included) * min_join_len

		for lib_i in range(0, len(L_included)) :
			lib = L_included[lib_i]

			#1. Get indexes of each Library

			if count_filter is None :
				apa_lib_index = numpy.nonzero(L == lib)[0]
			else :
				apa_lib_index = numpy.nonzero((L == lib) & (c > count_filter))[0]

			#2. Sort indexes of each library by count
			c_apa_lib = c[apa_lib_index]
			sort_index_apa_lib = numpy.argsort(c_apa_lib)
			apa_lib_index = apa_lib_index[sort_index_apa_lib]

			#3. Shuffle indexes of each library modulo 2
			even_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 0
			odd_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 1

			apa_lib_index_even = apa_lib_index[even_index_apa_lib]
			apa_lib_index_odd = apa_lib_index[odd_index_apa_lib]

			apa_lib_index = numpy.concatenate([apa_lib_index_even, apa_lib_index_odd])

			#4. Join modulo 2
			i = 0
			for j in range(len(apa_lib_index) - min_join_len, len(apa_lib_index)) :
				arranged_index[arranged_join_index + i * len(L_included) + lib_i] = apa_lib_index[j]
				i += 1

			#5. Append remainder
			for j in range(0, len(apa_lib_index) - min_join_len) :
				arranged_index[arranged_remainder_index] = apa_lib_index[j]
				arranged_remainder_index += 1

		if shuffle_index is not None :
			if constant_set_split == False :
				arranged_index[:int(train_set_split * len(arranged_index))] = arranged_index[shuffle_index]
			else :
				arranged_index[:int(len(arranged_index) - constant_valid_set_split - constant_test_set_split)] = arranged_index[shuffle_index]
		elif shuffle_all_index is not None :
			if constant_set_split == False :
				arranged_index[:] = arranged_index[shuffle_all_index]
			else :
				arranged_index[:] = arranged_index[shuffle_all_index]

		print('Arranged index:')
		print(len(arranged_index))
		print(arranged_index)

		L = L[arranged_index]
		c = c[arranged_index]
		y = y[arranged_index, :]

	y_train = None
	y_validation = None
	y_test = None
	if constant_set_split == False :
		y_train = y[:int(train_set_split * y.shape[0]),:]
		y_validation = y[y_train.shape[0]:y_train.shape[0] + int(valid_set_split * y.shape[0]),:]
		y_test = y[y_train.shape[0] + y_validation.shape[0]:,:]
	else :
		y_train = y[:y.shape[0] - constant_valid_set_split - constant_test_set_split,:]
		y_validation = y[y.shape[0] - constant_valid_set_split - constant_test_set_split:y.shape[0] - constant_test_set_split,:]
		y_test = y[y.shape[0] - constant_test_set_split:,:]

	print(y_train.shape[0])
	print(y_validation.shape[0])
	print(y_test.shape[0])

	y_test_hat = numpy.asarray(y_test[:, 1])
	print('Test set distrib [<= 0.2, 0.2 - 0.8, >= 0.8].')
	print( len(y_test_hat[y_test_hat <= 0.2]) )
	print( len(y_test_hat[(y_test_hat > 0.2) & (y_test_hat < 0.8)]) )
	print( len(y_test_hat[y_test_hat >= 0.8]) )

	print('Test set distrib [<= 0.5, > 0.5].')
	print( len(y_test_hat[y_test_hat <= 0.5]) )
	print( len(y_test_hat[y_test_hat > 0.5]) )

	#Spoof training set
	y_train = y[0:100, :]

	train_set_y = shared_dataset(y_train)
	valid_set_y = shared_dataset(y_validation)
	test_set_y = shared_dataset(y_test)

	rval = [train_set_y, valid_set_y, test_set_y]
	return rval

def permute_sparse_matrix(M, order):
	permuted_row = order[M.row]
	permuted_col = order[M.col]
	new_M = sparse.coo_matrix((M.data, (permuted_row, permuted_col)), shape=M.shape)
	return new_M

def load_input_data(dataset, L_included_list=None, constant_test_set_size_param=None, shuffle=False, shuffle_all=False, balance_data=False, balance_test_set=False, count_filter=None, misprime_filter=False, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=False):
	#############
	# LOAD DATA #
	#############

	print('... loading data')

	write_cache = False
	use_cache = False

	cache_name = ''#'20'#'20_AATAAA_pPas_No_dPas'#'20'#'22_denovo'#'22'#'20_AATAAA_pPas_No_dPas'#'20_AATAAA'#'22'#'20'#'test2percent'#''#20'#'20_22'

	train_set_split = 0.95#0.96
	valid_set_split = 0.02#0.02

	constant_set_split = True
	constant_valid_set_split = 1000
	constant_test_set_split = 40000#80000#40000

	if constant_test_set_size_param is not None :
		constant_set_split = True
		constant_test_set_split = constant_test_set_size_param

	rval = None

	if use_cache == False :
	
		#X = numpy.load(dataset + '_input.npy')#_foldprob
		#X = spio.loadmat('apa_fullseq_v_' + dataset + '_input.mat')
		#X = sp.csr_matrix(X["input"])

		loader = numpy.load('npz_apa_seq_' + dataset + '_input.npz')
		#loader = numpy.load('npz_apa_fullseq_' + dataset + '_input.npz')
		X = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)


		y = numpy.ravel(numpy.load('apa_' + dataset + '_output.npy')[:,1])
		c = numpy.ravel(numpy.load('apa_' + dataset + '_count.npy'))
		L = numpy.ravel(numpy.load('apa_' + dataset + '_libindex.npy'))

		d_input = numpy.load('apa_' + dataset + '_distalpas.npy')

		print('d_input')
		print(d_input[L <= 12,:])
		print(d_input[L == 20,:])
		print(d_input[L == 22,:])
		print(d_input[L >= 30,:])
		print(X.shape)
		print(y.shape)
		print(c.shape)
		print(L.shape)
		print(L)
		print(numpy.max(L))

		#L_prune_index = (L != 13) & (L != 14)
		#L_prune_index = (L >= 5) & (L <= 12)
		#L_prune_index = L == 5
		#L_prune_index = L >= 20
		'''X = X[L_prune_index,:]
		y = y[L_prune_index]
		c = c[L_prune_index]
		L = L[L_prune_index]'''

		'''sort_index = numpy.argsort(c)
		X = X[sort_index,:]
		y = y[sort_index]
		c = c[sort_index]
		L = L[sort_index]'''

		#Only non-competing PASes
		Floader = numpy.load('npz_apa_seq_' + dataset + '_features.npz')
		F = sp.csr_matrix((Floader['data'], Floader['indices'], Floader['indptr']), shape = Floader['shape'], dtype=numpy.int8)
	
		F_aataaa = numpy.ravel(F[:, 0].todense())
		F_attaaa = numpy.ravel(F[:, 1].todense())
		F_denovo = numpy.ravel(F[:, 5].todense())

		F_dpas_aataaa = numpy.ravel(F[:, 2].todense())
		F_dpas_attaaa = numpy.ravel(F[:, 3].todense())

		X = X[(F_aataaa == 1) & (F_dpas_aataaa == 0),:]
		L = L[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
		y = y[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
		c = c[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
		#X = X[F_aataaa == 1,:]
		#L = L[F_aataaa == 1]
		#y = y[F_aataaa == 1]
		#c = c[F_aataaa == 1]

		
		shuffle_index = None
		shuffle_all_index = None
		'''if shuffle == True :
			shuffle_index = numpy.arange(X.shape[0])
			numpy.random.shuffle(shuffle_index)
			X = X[shuffle_index,:]
			L = L[shuffle_index]
			c = c[shuffle_index]
			y = y[shuffle_index]'''

		'''if count_filter is not None :
			X = X[c > count_filter,:]
			L = L[c > count_filter]
			d_input = d_input[c > count_filter,:]
			y = y[c > count_filter]
			c = c[c > count_filter]'''

		print('Done count filtering.')
		print(X.shape)

		if balance_all_libraries == True :

			#List of included libraries
			L_included = [20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[20]#[2, 5, 8, 11]#[22]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[20]#[2, 5, 8, 11, 20]#, 22]#[20, 22]
			if L_included_list is not None :
				L_included = L_included_list

			arranged_index_len = 0
			min_join_len = len(numpy.nonzero(L == L_included[0])[0])
			if count_filter is not None :
				min_join_len = len(numpy.nonzero((L == L_included[0]) & (c > count_filter))[0])

			for lib in L_included :
				lib_len = len(numpy.nonzero(L == lib)[0])
				if count_filter is not None :
					lib_len = len(numpy.nonzero((L == lib) & (c > count_filter))[0])

				arranged_index_len += lib_len
				if lib_len < min_join_len :
					min_join_len = lib_len

			arranged_index = numpy.zeros(arranged_index_len, dtype=numpy.int)

			arranged_remainder_index = 0
			arranged_join_index = arranged_index_len - len(L_included) * min_join_len

			for lib_i in range(0, len(L_included)) :
				lib = L_included[lib_i]

				print('Arranging lib ' + str(lib))

				#1. Get indexes of each Library

				apa_lib_index = numpy.nonzero(L == lib)[0]
				if count_filter is not None :
					apa_lib_index = numpy.nonzero((L == lib) & (c > count_filter))[0]

				#2. Sort indexes of each library by count
				c_apa_lib = c[apa_lib_index]
				sort_index_apa_lib = numpy.argsort(c_apa_lib)
				apa_lib_index = apa_lib_index[sort_index_apa_lib]

				#3. Shuffle indexes of each library modulo 2
				even_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 0
				odd_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 1

				apa_lib_index_even = apa_lib_index[even_index_apa_lib]
				apa_lib_index_odd = apa_lib_index[odd_index_apa_lib]

				apa_lib_index = numpy.concatenate([apa_lib_index_even, apa_lib_index_odd])

				#4. Join modulo 2
				i = 0
				for j in range(len(apa_lib_index) - min_join_len, len(apa_lib_index)) :
					arranged_index[arranged_join_index + i * len(L_included) + lib_i] = apa_lib_index[j]
					i += 1

				#5. Append remainder
				for j in range(0, len(apa_lib_index) - min_join_len) :
					arranged_index[arranged_remainder_index] = apa_lib_index[j]
					arranged_remainder_index += 1

			if shuffle == True :
				if constant_set_split == False :
					shuffle_index = numpy.arange(len(arranged_index[:int(train_set_split * len(arranged_index))]))
					numpy.random.shuffle(shuffle_index)
					arranged_index[:int(train_set_split * len(arranged_index))] = arranged_index[shuffle_index]
				else :
					shuffle_index = numpy.arange(len(arranged_index[:int(len(arranged_index) - constant_valid_set_split - constant_test_set_split)]))
					numpy.random.shuffle(shuffle_index)
					arranged_index[:int(len(arranged_index) - constant_valid_set_split - constant_test_set_split)] = arranged_index[shuffle_index]
			elif shuffle_all == True :
				if constant_set_split == False :
					shuffle_all_index = numpy.arange(len(arranged_index))
					numpy.random.shuffle(shuffle_all_index)
					arranged_index[:] = arranged_index[shuffle_all_index]
				else :
					shuffle_all_index = numpy.arange(len(arranged_index))
					numpy.random.shuffle(shuffle_all_index)
					arranged_index[:] = arranged_index[shuffle_all_index]

			print('Arranged index:')
			print(len(arranged_index))
			print(arranged_index)

			X = X[arranged_index,:]
			L = L[arranged_index]
			d_input = d_input[arranged_index,:]
			c = c[arranged_index]
			y = y[arranged_index]


		L_input = numpy.zeros((len(y), 36))
		for i in range(0, len(y)) :
			L_input[i,int(L[i])] = 1

		X_train = None
		X_validation = None
		X_test = None
		y_train = None
		y_validation = None
		y_test = None
		c_train = None
		c_validation = None
		c_test = None
		if constant_set_split == False :
			X_train = X[:int(train_set_split * X.shape[0]),:]
			X_validation = X[X_train.shape[0]:X_train.shape[0] + int(valid_set_split * X.shape[0]),:]
			X_test = X[X_train.shape[0] + X_validation.shape[0]:,:]

			y_train = y[:int(train_set_split * y.shape[0])]
			y_validation = y[y_train.shape[0]:y_train.shape[0] + int(valid_set_split * y.shape[0])]
			y_test = y[y_train.shape[0] + y_validation.shape[0]:]

			c_train = c[:int(train_set_split * c.shape[0])]
			c_validation = c[c_train.shape[0]:c_train.shape[0] + int(valid_set_split * c.shape[0])]
			c_test = c[c_train.shape[0] + c_validation.shape[0]:]
		else :
			X_train = X[:X.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			X_validation = X[X.shape[0] - constant_valid_set_split - constant_test_set_split:X.shape[0] - constant_test_set_split,:]
			X_test = X[X.shape[0] - constant_test_set_split:,:]

			y_train = y[:y.shape[0] - constant_valid_set_split - constant_test_set_split]
			y_validation = y[y.shape[0] - constant_valid_set_split - constant_test_set_split:y.shape[0] - constant_test_set_split]
			y_test = y[y.shape[0] - constant_test_set_split:]

			c_train = c[:c.shape[0] - constant_valid_set_split - constant_test_set_split]
			c_validation = c[c.shape[0] - constant_valid_set_split - constant_test_set_split:c.shape[0] - constant_test_set_split]
			c_test = c[c.shape[0] - constant_test_set_split:]
		#Subselect of test set
		#X_test = X_test[y_test >= 0.8,:]

		#Spoof training set
		X_train = sp.csr_matrix(numpy.zeros((100, X.shape[1])))

		train_set_x = theano.shared(X_train, borrow=True)
		valid_set_x = theano.shared(X_validation, borrow=True)
		test_set_x = theano.shared(X_test, borrow=True)

		train_set_c = theano.shared(numpy.asarray(c_train, dtype=theano.config.floatX), borrow=True)
		valid_set_c = theano.shared(numpy.asarray(c_validation, dtype=theano.config.floatX), borrow=True)
		test_set_c= theano.shared(numpy.asarray(c_test, dtype=theano.config.floatX), borrow=True)

		L_train = None
		L_validation = None
		L_test = None
		if constant_set_split == False :
			L_train = L_input[:int(train_set_split * L_input.shape[0]),:]
			L_validation = L_input[L_train.shape[0]:L_train.shape[0] + int(valid_set_split * L_input.shape[0]),:]
			L_test = L_input[L_train.shape[0] + L_validation.shape[0]:,:]
		else :
			L_train = L_input[:L_input.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			L_validation = L_input[L_input.shape[0] - constant_valid_set_split - constant_test_set_split:L_input.shape[0] - constant_test_set_split,:]
			L_test = L_input[L_input.shape[0] - constant_test_set_split:,:]

		#Spoof training set
		L_train = numpy.zeros((100, 30))

		#train_set_L = theano.shared(L_train, borrow=True)
		#valid_set_L = theano.shared(L_validation, borrow=True)
		#test_set_L = theano.shared(L_test, borrow=True)

		train_set_L = theano.shared(numpy.asarray(L_train, dtype=theano.config.floatX), borrow=True)
		valid_set_L = theano.shared(numpy.asarray(L_validation, dtype=theano.config.floatX), borrow=True)
		test_set_L = theano.shared(numpy.asarray(L_test, dtype=theano.config.floatX), borrow=True)

		d_train = None
		d_validation = None
		d_test = None
		if constant_set_split == False :
			d_train = d_input[:int(train_set_split * d_input.shape[0]),:]
			d_validation = d_input[d_train.shape[0]:d_train.shape[0] + int(valid_set_split * d_input.shape[0]),:]
			d_test = d_input[d_train.shape[0] + d_validation.shape[0]:,:]
		else :
			d_train = d_input[:d_input.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			d_validation = d_input[d_input.shape[0] - constant_valid_set_split - constant_test_set_split:d_input.shape[0] - constant_test_set_split,:]
			d_test = d_input[d_input.shape[0] - constant_test_set_split:,:]

		#Spoof training set
		d_train = numpy.zeros((100, 30))

		train_set_d = theano.shared(numpy.asarray(d_train, dtype=theano.config.floatX), borrow=True)
		valid_set_d = theano.shared(numpy.asarray(d_validation, dtype=theano.config.floatX), borrow=True)
		test_set_d = theano.shared(numpy.asarray(d_test, dtype=theano.config.floatX), borrow=True)

		if write_cache == True :
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_train', data=X_train.data, indices=X_train.indices, indptr=X_train.indptr, shape=X_train.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_validation', data=X_validation.data, indices=X_validation.indices, indptr=X_validation.indptr, shape=X_validation.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_test', data=X_test.data, indices=X_test.indices, indptr=X_test.indptr, shape=X_test.shape)

			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_train', L_train)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_validation', L_validation)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_test', L_test)

			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_shuffle_index', shuffle_index)

		
		rval = [train_set_x, valid_set_x, test_set_x, shuffle_index, train_set_L, valid_set_L, test_set_L, None, train_set_d, valid_set_d, test_set_d, shuffle_all_index, train_set_c, valid_set_c, test_set_c]
	
	else :

		if cache_name != '' :
			cache_name = '_' + cache_name

		loader = numpy.load('cache' + cache_name + '_npz_apa_seq_' + dataset + '_input_train.npz')
		X_train = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)
		loader = numpy.load('cache' + cache_name + '_npz_apa_seq_' + dataset + '_input_validation.npz')
		X_validation = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)
		loader = numpy.load('cache' + cache_name + '_npz_apa_seq_' + dataset + '_input_test.npz')
		X_test = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)

		train_set_x = theano.shared(X_train, borrow=True)
		valid_set_x = theano.shared(X_validation, borrow=True)
		test_set_x = theano.shared(X_test, borrow=True)

		L_train = numpy.load('cache' + cache_name + '_apa_' + dataset + '_L_input_train.npy')
		L_validation = numpy.load('cache' + cache_name + '_apa_' + dataset + '_L_input_validation.npy')
		L_test = numpy.load('cache' + cache_name + '_apa_' + dataset + '_L_input_test.npy')

		train_set_L = theano.shared(L_train, borrow=True)
		valid_set_L = theano.shared(L_validation, borrow=True)
		test_set_L = theano.shared(L_test, borrow=True)

		shuffle_index = numpy.ravel(numpy.load('cache' + cache_name + '_apa_' + dataset + '_shuffle_index.npy'))

		rval = [train_set_x, valid_set_x, test_set_x, shuffle_index, train_set_L, valid_set_L, test_set_L, None, None]

	return rval

if __name__ == '__main__':
	datasets = load_data()
	#sgd_optimization_mnist()
