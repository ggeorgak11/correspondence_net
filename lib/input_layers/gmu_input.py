
import caffe
import numpy as np
from gmu_input_data import get_blobs, get_im_dims

class GmuInputLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		blob_list = ['data_1', 'data_2', 'depth_1', 'depth_2', 'RT_1', 'RT_2', 'intrinsic', 'boxes']
		dim = get_im_dims()
		boxes_dim = [1,1,11,4] # temporary dim
		im_dim, depth_dim, RT_dim, intr_dim = [1, 3, dim[1], dim[0]], [1, 1, dim[1], dim[0]], [1, 1, 1, 12], [1,1,1,4] #[1, 3, 1080, 1920], [1, 1, 1080, 1920], [1, 1, 1, 12], [1,1,1,4]
		dim_list = [ im_dim, im_dim, depth_dim, depth_dim, RT_dim, RT_dim, intr_dim, boxes_dim ]
		self.blob_order={}
		for i in range(len(blob_list)):
			self.blob_order[blob_list[i]] = i
			top[i].reshape(dim_list[i][0], dim_list[i][1], dim_list[i][2], dim_list[i][3])	
		print 'InputLayer: blob list:', self.blob_order
		assert len(top)==len(self.blob_order), "Number of input blobs must be the same as the number of top blobs!"
	
	def reshape(self, bottom, top):
		# reshape is done in setup so as not to do it every time the layer is called
		pass
	
	def forward(self, bottom, top):
		blobs = get_blobs()
		for blob_name, blob in blobs.iteritems():
			#print blob_name, blob.shape
			top_ind = self.blob_order[blob_name]
			# Reshape net's input blobs
			top[top_ind].reshape(*(blob.shape))
			# Copy data into net's input blobs
			top[top_ind].data[...] = blob.astype(np.float32, copy=False)
		
	
	def backward(self, top, propagate_down, bottom):
		# this layer does not backpropagate gradients
		pass
	
	
	


