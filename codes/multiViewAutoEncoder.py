import os
import sys
import time
from numpy import *
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images



import Image

import random

import itertools


def random_combination(iterable, r):
    	"Random selection from itertools.combinations(iterable, r)"
	pool = tuple(iterable)
	n = len(pool)
	indices = sorted(random.sample(xrange(n), r))
	return tuple(pool[i] for i in indices)	





class multiViewAutoEncoder(object):





    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W1=None,
	W2=None,
        b1hid=None,
	b2hid=None,
	b1vis=None,
	b2vis=None,
        lamda=4
		):

		self.n_visible=n_visible
		self.n_hidden=n_hidden
                if not numpy_rng:
                    numpy_rng=numpy.random.RandomState(123)

		if not theano_rng:
			theano_rng=RandomStreams(numpy_rng.randint(2**30))

		if not W1:
			initial_W1=numpy.asarray(
				numpy_rng.uniform(
					low=-4*numpy.sqrt(6./ (n_hidden+n_visible)),
					high=4*numpy.sqrt(6./(n_hidden+n_visible)),
					size=(n_visible,n_hidden)),
				dtype=theano.config.floatX
			)
			W1=theano.shared(value=initial_W1,name='W1',borrow=True)
		
		if not W2:
			initial_W2=numpy.asarray(
				numpy_rng.uniform(
					low=-4*numpy.sqrt(6./ (n_hidden+n_visible)),
					high=4*numpy.sqrt(6./(n_hidden+n_visible)),
					size=(n_visible,n_hidden)),
				dtype=theano.config.floatX
			)
		
			W2=theano.shared(value=initial_W2,name='W2',borrow=True)
		
		if not b1vis:
			b1vis= theano.shared(value=numpy.zeros(
					n_visible,
					dtype=theano.config.floatX
					),
					borrow=True
				)

		if not b2vis:
			b2vis= theano.shared(value=numpy.zeros(
					n_visible,
					dtype=theano.config.floatX
					),
					borrow=True
				)

		if not b1hid:
			b1hid= theano.shared(value=numpy.zeros(
					n_hidden,
					dtype=theano.config.floatX
					),
					name='b1',
					borrow=True
				)


		if not b2hid:
			b2hid= theano.shared(value=numpy.zeros(
					n_hidden,
					dtype=theano.config.floatX
					),
					name='b2',
					borrow=True
				)

		self.W1=W1
		
		self.W2=W2

		self.b1=b1hid

		self.b2=b2hid

		self.b1_prime=b1vis

		self.b2_prime=b2vis

		self.theano_rng=theano_rng

                self.W1_prime=self.W1.T

                self.W2_prime=self.W2.T

                self.lamda=lamda
		if input is None:
			self.x=T.dmatrix(name='input')
		else:
			self.x=input
		self.params=[self.W1,self.b1,self.b1_prime,self.W2,self.b2,self.b2_prime]


	
    def get_reconstructed_input1(self,hidden):
		return T.nnet.sigmoid(T.dot(hidden,self.W1_prime)+self.b1_prime)

    def get_reconstructed_input2(self,hidden):
		return T.nnet.sigmoid(T.dot(hidden,self.W2_prime)+self.b2_prime)

    def get_hidden_values1(self,input ):
		return T.nnet.sigmoid(T.dot(input,self.W1)+self.b1)

    def get_hidden_values2(self,input ):
		return T.nnet.sigmoid(T.dot(input,self.W2)+self.b2)
		    
    #def get_covariance(self,a,b):
     #   return numpy.cov(a,b)[0][1] 
    def get_cost_updates(self,learning_rate):
		

		y1=self.get_hidden_values1(self.x)
		z1=self.get_reconstructed_input1(y1)
		
		y2=self.get_hidden_values2(self.x)
		z2=self.get_reconstructed_input2(y2)

		L1=-T.sum(self.x*T.log(z1)+(1-self.x)*T.log(1-z1),axis=1)


		L2=-T.sum(self.x*T.log(z2)+(1-self.x)*T.log(1-z2),axis=1)

		comb=itertools.combinations(range(self.n_hidden),2)		
		rand_comb=random_combination(comb,10*self.n_hidden)
		correlation=[]

		for i in rand_comb:
                        x1=y1[:,i[0]]-ones(20)*T.sum(y1[:,i[0]])/20
                        x2=y2[:,i[1]]-ones(20)*T.sum(y2[:,i[1]])/20
                        nr=T.sum(x1*x2)/(T.sqrt(x1*x1)*T.sqrt(T.sum(x2*x2)))
			correlation.append(nr)
                        
                tot_correlation =T.sum(correlation)

		L = L1+L2+(self.lamda*tot_correlation)
                cost=T.mean(L)
		
		gradients=T.grad(cost,self.params)
		updates=[]
		for param,gparam in zip(self.params,gradients):
			updates.append((param,param-learning_rate*gparam))
			return (cost,updates)







		
def testMultiviewAutoEncoders(learning_rate=.1,batch_size=20,training_epochs=2,dataset='mnist.pkl.gz',output_folder='MVAE_plots'):
	
    dataset=load_data(dataset)
    train_set_x=theano.shared(numpy.asarray(numpy.zeros((100,728)),dtype=theano.config.floatX),borrow=True)

    train_set_x,train_set_y=dataset[0]

    n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size

    index=T.lscalar()
    x=T.matrix('x')

		
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    os.chdir(output_folder)

    multiViewAE=multiViewAutoEncoder(numpy_rng=None,theano_rng=None,input=x,n_visible=28*28,n_hidden=500)
    cost,updates=multiViewAE.get_cost_updates(learning_rate=0.1)
    train_MVAE=theano.function([index],cost,updates=updates,givens={x:train_set_x[index*batch_size:(index+1)*batch_size]})

    start_time=time.clock()

    for epoch in range(training_epochs):
    	c=[]
    	for batch_index in range(n_train_batches):
    		c.append(train_da(batch_index))

    	print 'Training epoch %d , cost %f '%epoch,numpy.mean(c)

    end_time=time.clock()
    training_time=(end_time-start_time)
		
	   


    print >> sys.stderr, ('The no training time for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W1.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_view1.png')


    image = Image.fromarray(
        tile_raster_images(X=da.W2.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_view2.png')

if __name__=='__main__':
	testMultiviewAutoEncoders()

