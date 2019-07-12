# Core network definitions

import _init_paths
import caffe
from caffe import layers as L
from caffe import params as P
import os, cv2
import collections
import numpy as np


def _getLr_multiples(name, values): # values are list of numbers [w_lr_mult, w_decay_mult, b_lr_mult, b_decay_mult]
	weight_param = dict(name=name + '_w', lr_mult=values[0], decay_mult=values[1])
	bias_param = dict(name=name + '_b', lr_mult=values[2], decay_mult=values[3])
	return [weight_param, bias_param]
	

def createCoreCorrespondenceNetwork_32S(data, par):
	network = collections.OrderedDict()

	# conv 1
	network['conv1_1'] = L.Convolution(data, kernel_size=3, num_output=64, pad=100, stride=1, param=_getLr_multiples('conv1_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu1_1'] = L.ReLU(network['conv1_1'], in_place=True)
	network['conv1_2'] = L.Convolution(network['relu1_1'], kernel_size=3, num_output=64, pad=1, stride=1, param=_getLr_multiples('conv1_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))			
	network['relu1_2'] = L.ReLU(network['conv1_2'], in_place=True)
	network['pool1'] = L.Pooling(network['relu1_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	# conv 2
	network['conv2_1'] = L.Convolution(network['pool1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu2_1'] = L.ReLU(network['conv2_1'], in_place=True)
	network['conv2_2'] = L.Convolution(network['relu2_1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu2_2'] = L.ReLU(network['conv2_2'], in_place=True)
	network['pool2'] = L.Pooling(network['relu2_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	# conv 3
	network['conv3_1'] = L.Convolution(network['pool2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_1'] = L.ReLU(network['conv3_1'], in_place=True)
	network['conv3_2'] = L.Convolution(network['relu3_1'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_2'] = L.ReLU(network['conv3_2'], in_place=True)
	network['conv3_3'] = L.Convolution(network['relu3_2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_3'] = L.ReLU(network['conv3_3'], in_place=True)
	network['pool3'] = L.Pooling(network['relu3_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	# conv 4
	network['conv4_1'] = L.Convolution(network['pool3'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_1'] = L.ReLU(network['conv4_1'], in_place=True)
	network['conv4_2'] = L.Convolution(network['relu4_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_2'] = L.ReLU(network['conv4_2'], in_place=True)
	network['conv4_3'] = L.Convolution(network['relu4_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_3'] = L.ReLU(network['conv4_3'], in_place=True)
	network['pool4'] = L.Pooling(network['relu4_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)	
	# conv 5
	network['conv5_1'] = L.Convolution(network['pool4'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_1'] = L.ReLU(network['conv5_1'], in_place=True)
	network['conv5_2'] = L.Convolution(network['relu5_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_2'] = L.ReLU(network['conv5_2'], in_place=True)
	network['conv5_3'] = L.Convolution(network['relu5_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_3'] = L.ReLU(network['conv5_3'], in_place=True)
	network['pool5'] = L.Pooling(network['relu5_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	# Last two conv layers with dropout
	network['fc6'] = L.Convolution(network['pool5'],  kernel_size=7, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc6', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu6'] = L.ReLU(network['fc6'], in_place=True)
	network['drop6'] = L.Dropout(network['relu6'], dropout_ratio=0.5)
	network['fc7'] = L.Convolution(network['drop6'],  kernel_size=1, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc7', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu7'] = L.ReLU(network['fc7'], in_place=True)
	network['drop7'] = L.Dropout(network['relu7'], dropout_ratio=0.5)
	# score conv layer
	network['score_fr'] = L.Convolution(network['drop7'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_fr', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	# upsampling layer
	network['upscore'] = L.Deconvolution(network['score_fr'], convolution_param=dict(kernel_size=64, num_output=par.feat_dim, bias_term=False, stride=32, 
										weight_filler=dict(type='gaussian', std=0.01)), param=[{'lr_mult':0}])
	# crop layer to get the final score map
	network['score'] = L.Crop(network['upscore'], data, axis=2, offset=19)

	return network


def ScaleConv(name, inData, kernel_dim, nOut): # name can be dil1_2
	#smooth1 = L.Convolution(inData, kernel_size=3, num_output=nOut, pad=1, stride=1, param=_getLr_multiples(name+'_smooth1', [0,0,0,0]), 
	#					   weight_filler=dict(type='gaussian', std=1), bias_filler=dict(type='constant', value=0))
	dil1 = L.Convolution(inData, kernel_size=kernel_dim, num_output=nOut, pad=1, stride=1, dilation=1, param=_getLr_multiples(name, [1,1,2,0]),
						weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

#	smooth2 = L.Convolution(inData, kernel_size=5, num_output=nOut, pad=2, stride=1, param=_getLr_multiples(name+'_smooth2', [0,0,0,0]), 
#						   weight_filler=dict(type='gaussian', std=1), bias_filler=dict(type='constant', value=0))	
	dil2 = L.Convolution(inData, kernel_size=kernel_dim, num_output=nOut, pad=2, stride=1, dilation=2, param=_getLr_multiples(name, [1,1,2,0]),
						weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

#	smooth3 = L.Convolution(inData, kernel_size=7, num_output=nOut, pad=3, stride=1, param=_getLr_multiples(name+'_smooth3', [0,0,0,0]), 
#						   weight_filler=dict(type='gaussian', std=1), bias_filler=dict(type='constant', value=0))	
	dil3 = L.Convolution(inData, kernel_size=kernel_dim, num_output=nOut, pad=3, stride=1, dilation=3, param=_getLr_multiples(name, [1,1,2,0]),
						weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

#	smooth4 = L.Convolution(inData, kernel_size=9, num_output=nOut, pad=4, stride=1, param=_getLr_multiples(name+'_smooth4', [0,0,0,0]), 
#					   weight_filler=dict(type='gaussian', std=1), bias_filler=dict(type='constant', value=0))
	dil4 = L.Convolution(inData, kernel_size=kernel_dim, num_output=nOut, pad=4, stride=1, dilation=4, param=_getLr_multiples(name, [1,1,2,0]),
						weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))

	#scale_pool = L.Eltwise(relu_dil1, relu_dil2, relu_dil3, relu_dil4, operation=P.Eltwise.MAX)
	#return relu_dil1, relu_dil2, relu_dil3, relu_dil4, scale_pool
	scale_pool = L.Eltwise(dil1, dil2, dil3, dil4, operation=P.Eltwise.MAX)
	return dil1, dil2, dil3, dil4, scale_pool


def createCoreCorrespondenceNetwork_Scale8S(data, par):
	network = collections.OrderedDict()

	# conv 1
	network['conv1_1'] = L.Convolution(data, kernel_size=3, num_output=64, pad=85, stride=1, param=_getLr_multiples('conv1_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu1_1'] = L.ReLU(network['conv1_1'], in_place=True)	
	#network['conv1_2'] = L.Convolution(network['relu1_1'], kernel_size=3, num_output=64, pad=1, stride=1, param=_getLr_multiples('conv1_2', [1,1,2,0]), 
	#									weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))			
	#network['relu1_2'] = L.ReLU(network['conv1_2'], in_place=True)
	#network['pool1'] = L.Pooling(network['relu1_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	network["dil1_2_1"], network["dil1_2_2"], network["dil1_2_3"], network["dil1_2_4"], network["Sc_Pool1_2"] = ScaleConv("conv1_2", network['relu1_1'], 3, 64)
	network['pool1'] = L.Pooling(network['Sc_Pool1_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# conv 2
	network['conv2_1'] = L.Convolution(network['pool1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu2_1'] = L.ReLU(network['conv2_1'], in_place=True)
	#network['conv2_2'] = L.Convolution(network['relu2_1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_2', [1,1,2,0]), 
	#									weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	#network['relu2_2'] = L.ReLU(network['conv2_2'], in_place=True)
	#network['pool2'] = L.Pooling(network['relu2_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	network["dil2_2_1"], network["dil2_2_2"], network["dil2_2_3"], network["dil2_2_4"], network["Sc_Pool2_2"] = ScaleConv("conv2_2", network['relu2_1'], 3, 128)	
	network['pool2'] = L.Pooling(network['Sc_Pool2_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# conv 3
	network['conv3_1'] = L.Convolution(network['pool2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_1'] = L.ReLU(network['conv3_1'], in_place=True)
	network['conv3_2'] = L.Convolution(network['relu3_1'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_2'] = L.ReLU(network['conv3_2'], in_place=True)
	network['conv3_3'] = L.Convolution(network['relu3_2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_3'] = L.ReLU(network['conv3_3'], in_place=True)
	network['pool3'] = L.Pooling(network['relu3_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	#network["dil3_3_1"], network["dil3_3_2"], network["dil3_3_3"], network["dil3_3_4"], network["Sc_Pool3_3"] = ScaleConv("conv3_3", network['relu3_2'], 3, 256)
	#network['pool3'] = L.Pooling(network['Sc_Pool3_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# conv 4
	network['conv4_1'] = L.Convolution(network['pool3'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_1'] = L.ReLU(network['conv4_1'], in_place=True)
	network['conv4_2'] = L.Convolution(network['relu4_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_2'] = L.ReLU(network['conv4_2'], in_place=True)
	network['conv4_3'] = L.Convolution(network['relu4_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_3'] = L.ReLU(network['conv4_3'], in_place=True)
	network['pool4'] = L.Pooling(network['relu4_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)	
	#network["dil4_3_1"], network["dil4_3_2"], network["dil4_3_3"], network["dil4_3_4"], network["Sc_Pool4_3"] = ScaleConv("conv4_3", network['relu4_2'], 3, 512)
	#network['pool4'] = L.Pooling(network['Sc_Pool4_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)	
	
	# conv 5
	network['conv5_1'] = L.Convolution(network['pool4'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_1'] = L.ReLU(network['conv5_1'], in_place=True)
	network['conv5_2'] = L.Convolution(network['relu5_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_2'] = L.ReLU(network['conv5_2'], in_place=True)
	network['conv5_3'] = L.Convolution(network['relu5_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_3'] = L.ReLU(network['conv5_3'], in_place=True)
	network['pool5'] = L.Pooling(network['relu5_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	#network["dil5_3_1"], network["dil5_3_2"], network["dil5_3_3"], network["dil5_3_4"], network["Sc_Pool5_3"] = ScaleConv("conv5_3", network['relu5_2'], 3, 512)
	#network['pool5'] = L.Pooling(network['Sc_Pool5_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# Last two conv layers with dropout, upscore from fully connected layers
	network['fc6'] = L.Convolution(network['pool5'],  kernel_size=7, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc6', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu6'] = L.ReLU(network['fc6'], in_place=True)
	network['drop6'] = L.Dropout(network['relu6'], dropout_ratio=0.5)
	
	network['fc7'] = L.Convolution(network['drop6'],  kernel_size=1, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc7', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu7'] = L.ReLU(network['fc7'], in_place=True)
	network['drop7'] = L.Dropout(network['relu7'], dropout_ratio=0.5)
	
	# score conv layer
	network['score_fr'] = L.Convolution(network['drop7'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_fr', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										#weight_filler=dict(type='xavier') )
	# upsampling layer
	network['upscore2'] = L.Deconvolution(network['score_fr'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=4, num_output=par.feat_dim, bias_term=False, stride=2,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
	# Upscore from pool4
	network['score_pool4'] = L.Convolution(network['pool4'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_pool4', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										 #  weight_filler=dict(type='xavier') )
	network['score_pool4c'] = L.Crop(network['score_pool4'], network['upscore2'], axis=2, offset=5)
	network['fuse_pool4'] = L.Eltwise(network['upscore2'], network['score_pool4c'], operation=P.Eltwise.SUM)
	network['upscore_pool4'] = L.Deconvolution(network['fuse_pool4'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=4, num_output=par.feat_dim, bias_term=False, stride=2,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
	# upscore from pool3
	network['score_pool3'] = L.Convolution(network['pool3'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_pool3', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										#   weight_filler=dict(type='xavier') )
	network['score_pool3c'] = L.Crop(network['score_pool3'], network['upscore_pool4'], axis=2, offset=9)
	network['fuse_pool3'] = L.Eltwise(network['upscore_pool4'], network['score_pool3c'], operation=P.Eltwise.SUM)
	network['upscore8'] = L.Deconvolution(network['fuse_pool3'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=16, num_output=par.feat_dim, bias_term=False, stride=8,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
										  
	# crop layer to get the final score map
	#network['score'] = L.Crop(network['upscore8'], data, axis=2, offset=31)
	network['score'] = L.Crop(network['upscore8'], data, crop_param=dict(axis=2, offset=[20,28])) # axis=2, offset=21)

	return network


def createCoreCorrespondenceNetwork_Scale4S(data, par):
	network = collections.OrderedDict()

	# conv 1
	network['conv1_1'] = L.Convolution(data, kernel_size=3, num_output=64, pad=85, stride=1, param=_getLr_multiples('conv1_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu1_1'] = L.ReLU(network['conv1_1'], in_place=True)	
	#network['conv1_2'] = L.Convolution(network['relu1_1'], kernel_size=3, num_output=64, pad=1, stride=1, param=_getLr_multiples('conv1_2', [1,1,2,0]), 
	#									weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))			
	#network['relu1_2'] = L.ReLU(network['conv1_2'], in_place=True)
	#network['pool1'] = L.Pooling(network['relu1_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	network["dil1_2_1"], network["dil1_2_2"], network["dil1_2_3"], network["dil1_2_4"], network["Sc_Pool1_2"] = ScaleConv("conv1_2", network['relu1_1'], 3, 64)
	network['pool1'] = L.Pooling(network['Sc_Pool1_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# conv 2
	network['conv2_1'] = L.Convolution(network['pool1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu2_1'] = L.ReLU(network['conv2_1'], in_place=True)
	#network['conv2_2'] = L.Convolution(network['relu2_1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_2', [1,1,2,0]), 
	#									weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	#network['relu2_2'] = L.ReLU(network['conv2_2'], in_place=True)
	#network['pool2'] = L.Pooling(network['relu2_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	network["dil2_2_1"], network["dil2_2_2"], network["dil2_2_3"], network["dil2_2_4"], network["Sc_Pool2_2"] = ScaleConv("conv2_2", network['relu2_1'], 3, 128)	
	network['pool2'] = L.Pooling(network['Sc_Pool2_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# conv 3
	network['conv3_1'] = L.Convolution(network['pool2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_1'] = L.ReLU(network['conv3_1'], in_place=True)
	network['conv3_2'] = L.Convolution(network['relu3_1'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_2'] = L.ReLU(network['conv3_2'], in_place=True)
	network['conv3_3'] = L.Convolution(network['relu3_2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_3'] = L.ReLU(network['conv3_3'], in_place=True)
	network['pool3'] = L.Pooling(network['relu3_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	#network["dil3_3_1"], network["dil3_3_2"], network["dil3_3_3"], network["dil3_3_4"], network["Sc_Pool3_3"] = ScaleConv("conv3_3", network['relu3_2'], 3, 256)
	#network['pool3'] = L.Pooling(network['Sc_Pool3_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# conv 4
	network['conv4_1'] = L.Convolution(network['pool3'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_1'] = L.ReLU(network['conv4_1'], in_place=True)
	network['conv4_2'] = L.Convolution(network['relu4_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_2'] = L.ReLU(network['conv4_2'], in_place=True)
	network['conv4_3'] = L.Convolution(network['relu4_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_3'] = L.ReLU(network['conv4_3'], in_place=True)
	network['pool4'] = L.Pooling(network['relu4_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)	
	#network["dil4_3_1"], network["dil4_3_2"], network["dil4_3_3"], network["dil4_3_4"], network["Sc_Pool4_3"] = ScaleConv("conv4_3", network['relu4_2'], 3, 512)
	#network['pool4'] = L.Pooling(network['Sc_Pool4_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)	
	
	# conv 5
	network['conv5_1'] = L.Convolution(network['pool4'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_1'] = L.ReLU(network['conv5_1'], in_place=True)
	network['conv5_2'] = L.Convolution(network['relu5_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_2'] = L.ReLU(network['conv5_2'], in_place=True)
	network['conv5_3'] = L.Convolution(network['relu5_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_3'] = L.ReLU(network['conv5_3'], in_place=True)
	network['pool5'] = L.Pooling(network['relu5_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	#network["dil5_3_1"], network["dil5_3_2"], network["dil5_3_3"], network["dil5_3_4"], network["Sc_Pool5_3"] = ScaleConv("conv5_3", network['relu5_2'], 3, 512)
	#network['pool5'] = L.Pooling(network['Sc_Pool5_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# Last two conv layers with dropout, upscore from fully connected layers
	network['fc6'] = L.Convolution(network['pool5'],  kernel_size=7, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc6', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu6'] = L.ReLU(network['fc6'], in_place=True)
	network['drop6'] = L.Dropout(network['relu6'], dropout_ratio=0.5)
	
	network['fc7'] = L.Convolution(network['drop6'],  kernel_size=1, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc7', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu7'] = L.ReLU(network['fc7'], in_place=True)
	network['drop7'] = L.Dropout(network['relu7'], dropout_ratio=0.5)
	
	# score conv layer
	network['score_fr'] = L.Convolution(network['drop7'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_fr', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										#weight_filler=dict(type='xavier') )
	# upsampling layer
	network['upscore2'] = L.Deconvolution(network['score_fr'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=4, num_output=par.feat_dim, bias_term=False, stride=2,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
	# Upscore from pool4
	network['score_pool4'] = L.Convolution(network['pool4'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_pool4', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										 #  weight_filler=dict(type='xavier') )
	network['score_pool4c'] = L.Crop(network['score_pool4'], network['upscore2'], axis=2, offset=5)
	network['fuse_pool4'] = L.Eltwise(network['upscore2'], network['score_pool4c'], operation=P.Eltwise.SUM)
	network['upscore_pool4'] = L.Deconvolution(network['fuse_pool4'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=4, num_output=par.feat_dim, bias_term=False, stride=2,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
	# upscore from pool3
	network['score_pool3'] = L.Convolution(network['pool3'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_pool3', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										#   weight_filler=dict(type='xavier') )
	network['score_pool3c'] = L.Crop(network['score_pool3'], network['upscore_pool4'], axis=2, offset=9)
	network['fuse_pool3'] = L.Eltwise(network['upscore_pool4'], network['score_pool3c'], operation=P.Eltwise.SUM)
	network['upscore_pool3'] = L.Deconvolution(network['fuse_pool3'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=4, num_output=par.feat_dim, bias_term=False, stride=2,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
	# upscore from pool2
	network['score_pool2'] = L.Convolution(network['pool2'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_pool2', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))	
	network['score_pool2c'] = L.Crop(network['score_pool2'], network['upscore_pool3'], axis=2, offset=9)
	network['fuse_pool2'] = L.Eltwise(network['upscore_pool3'], network['score_pool2c'], operation=P.Eltwise.SUM)
	network['upscore8'] = L.Deconvolution(network['fuse_pool2'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=16, num_output=par.feat_dim, bias_term=False, stride=4,
										weight_filler=dict(type='gaussian', std=par.std_conv)))	
	
	# crop layer to get the final score map
	network['score'] = L.Crop(network['upscore8'], data, crop_param=dict(axis=2, offset=[26,34]))#axis=2, offset=31)
	#network['score'] = L.Crop(network['upscore_pool3'], data, axis=2, offset=21)

	return network



def createCoreCorrespondenceNetwork_8S(data, par):
	network = collections.OrderedDict()

	# conv 1
	network['conv1_1'] = L.Convolution(data, kernel_size=3, num_output=64, pad=100, stride=1, param=_getLr_multiples('conv1_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu1_1'] = L.ReLU(network['conv1_1'], in_place=True)
	network['conv1_2'] = L.Convolution(network['relu1_1'], kernel_size=3, num_output=64, pad=1, stride=1, param=_getLr_multiples('conv1_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))			
	network['relu1_2'] = L.ReLU(network['conv1_2'], in_place=True)
	network['pool1'] = L.Pooling(network['relu1_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	# conv 2
	network['conv2_1'] = L.Convolution(network['pool1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu2_1'] = L.ReLU(network['conv2_1'], in_place=True)
	network['conv2_2'] = L.Convolution(network['relu2_1'], kernel_size=3, num_output=128, pad=1, stride=1, param=_getLr_multiples('conv2_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu2_2'] = L.ReLU(network['conv2_2'], in_place=True)
	network['pool2'] = L.Pooling(network['relu2_2'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	# conv 3
	network['conv3_1'] = L.Convolution(network['pool2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_1'] = L.ReLU(network['conv3_1'], in_place=True)
	network['conv3_2'] = L.Convolution(network['relu3_1'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_2'] = L.ReLU(network['conv3_2'], in_place=True)
	network['conv3_3'] = L.Convolution(network['relu3_2'], kernel_size=3, num_output=256, pad=1, stride=1, param=_getLr_multiples('conv3_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu3_3'] = L.ReLU(network['conv3_3'], in_place=True)
	network['pool3'] = L.Pooling(network['relu3_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	# conv 4
	network['conv4_1'] = L.Convolution(network['pool3'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_1'] = L.ReLU(network['conv4_1'], in_place=True)
	network['conv4_2'] = L.Convolution(network['relu4_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_2'] = L.ReLU(network['conv4_2'], in_place=True)
	network['conv4_3'] = L.Convolution(network['relu4_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv4_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu4_3'] = L.ReLU(network['conv4_3'], in_place=True)
	network['pool4'] = L.Pooling(network['relu4_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)	
	# conv 5
	network['conv5_1'] = L.Convolution(network['pool4'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_1', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_1'] = L.ReLU(network['conv5_1'], in_place=True)
	network['conv5_2'] = L.Convolution(network['relu5_1'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_2', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_2'] = L.ReLU(network['conv5_2'], in_place=True)
	network['conv5_3'] = L.Convolution(network['relu5_2'], kernel_size=3, num_output=512, pad=1, stride=1, param=_getLr_multiples('conv5_3', [1,1,2,0]), 
										weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu5_3'] = L.ReLU(network['conv5_3'], in_place=True)
	network['pool5'] = L.Pooling(network['relu5_3'], kernel_size=2, stride=2, pool=P.Pooling.MAX)
	
	# Last two conv layers with dropout, upscore from fully connected layers
	network['fc6'] = L.Convolution(network['pool5'],  kernel_size=7, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc6', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu6'] = L.ReLU(network['fc6'], in_place=True)
	network['drop6'] = L.Dropout(network['relu6'], dropout_ratio=0.5)
	network['fc7'] = L.Convolution(network['drop6'],  kernel_size=1, num_output=4096, pad=0, stride=1, param=_getLr_multiples('fc7', [1,1,2,0]), 
								  weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0))
	network['relu7'] = L.ReLU(network['fc7'], in_place=True)
	network['drop7'] = L.Dropout(network['relu7'], dropout_ratio=0.5)
	# score conv layer
	network['score_fr'] = L.Convolution(network['drop7'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_fr', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										#weight_filler=dict(type='xavier') )
	# upsampling layer
	network['upscore2'] = L.Deconvolution(network['score_fr'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=4, num_output=par.feat_dim, bias_term=False, stride=2,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
	# Upscore from pool4
	network['score_pool4'] = L.Convolution(network['pool4'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_pool4', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										 #  weight_filler=dict(type='xavier') )
	network['score_pool4c'] = L.Crop(network['score_pool4'], network['upscore2'], axis=2, offset=5)
	network['fuse_pool4'] = L.Eltwise(network['upscore2'], network['score_pool4c'], operation=P.Eltwise.SUM)
	network['upscore_pool4'] = L.Deconvolution(network['fuse_pool4'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=4, num_output=par.feat_dim, bias_term=False, stride=2,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
	# upscore from pool3
	network['score_pool3'] = L.Convolution(network['pool3'], kernel_size=1, num_output=par.feat_dim, pad=0, param=_getLr_multiples('score_pool3', [1,1,2,0]), 
									   weight_filler=dict(type='gaussian', std=par.std_conv), bias_filler=dict(type='constant', value=0))
										#   weight_filler=dict(type='xavier') )
	network['score_pool3c'] = L.Crop(network['score_pool3'], network['upscore_pool4'], axis=2, offset=9)
	network['fuse_pool3'] = L.Eltwise(network['upscore_pool4'], network['score_pool3c'], operation=P.Eltwise.SUM)
	network['upscore8'] = L.Deconvolution(network['fuse_pool3'], param=[{'lr_mult':0}], convolution_param=dict(kernel_size=16, num_output=par.feat_dim, bias_term=False, stride=8,
										weight_filler=dict(type='gaussian', std=par.std_conv)))
										#weight_filler=dict(type='xavier') ) )
										  
	# crop layer to get the final score map
	network['score'] = L.Crop(network['upscore8'], data, axis=2, offset=31)

	return network



#def createCoreNetwork_DeepLab




































