
import caffe
import numpy as np
import yaml
import cv2
import time
import matplotlib.pyplot as plt
from functools import reduce
import visualizations as vis

'''
Find correspondences between a pair of images and create a training set of positive and negative pairs

Bottom blobs:
0: scores_1
1: scores_2
2: RT_1
3: RT_2
4: intrinsic
5: depth_1
6: depth_2

Top blobs:
0: feat_1
1: feat_2
2: labels
'''

class CorrespondenceLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		# get the layer params defined in the prototxt
		layer_params = yaml.load(self.param_str)
		self.nPositives = layer_params['nPositives']
		self.nNegPerPos = layer_params['nNegPerPos']
		self.objSampling = layer_params['objSampling']
		self.nSamples = self.nPositives + self.nPositives*self.nNegPerPos
		self.feat_dim = bottom[0].channels
		#self.debug = False

	
	def reshape(self, bottom, top):
		# necessary reshapes for the top blobs
		top[0].reshape(self.nSamples, bottom[0].channels, 1, 1)
		top[1].reshape(self.nSamples, bottom[0].channels, 1, 1)
		top[2].reshape(self.nSamples, 1, 1, 1)

		
	def forward(self, bottom, top):
		# unpack the scores
		scores_1 = bottom[0].data
		scores_1 = scores_1.transpose((2,3,1,0))
		scores_1 = scores_1[:,:,:,0]
		scores_2 = bottom[1].data
		scores_2 = scores_2.transpose((2,3,1,0))
		scores_2 = scores_2[:,:,:,0]		
		# unpack the intrinsic parameters and poses
		RT_1 = bottom[2].data
		RT_2 = bottom[3].data
		intr = bottom[4].data
		intr = intr[0,0,0,:]
		RT_1, RT_2 = RT_1[0,0,0,:], RT_2[0,0,0,:]
		R1, T1 = RT_1[:9].reshape(3,3), RT_1[9:]
		R2, T2 = RT_2[:9].reshape(3,3), RT_2[9:]
		# unpack the depth blobs
		depth_1 = bottom[5].data
		depth_2 = bottom[6].data		
		depth_1 = depth_1.transpose((2,3,0,1))
		depth_1 = depth_1[:,:,0,0]
		depth_1 = depth_1.astype(np.int, copy=False)
		im_size = depth_1.shape # (480,640)
		depth_2 = depth_2.transpose((2,3,0,1))
		depth_2 = depth_2[:,:,0,0]
		depth_2 = depth_2.astype(np.int, copy=False)		
		# unpack the gt boxes 
		boxes = bottom[7].data # gt boxes for image1
		boxes = boxes[0,0,:,:].astype(np.int, copy=False)

		# get all coordinates that do not have a 0 depth value and are inside the buffer
		#start_time=time.time() ###
		inds = np.where(depth_1 > 0)
		coords = np.zeros((len(inds[0]), 2), dtype=np.int)
		coords[:,0] = inds[1] # inds[1] is x (width coordinate)
		coords[:,1] = inds[0]
		#coords = coords[0::10,:] # subsample coords
		#print("---2 %s seconds ---" % (time.time() - start_time)) ###
		
		#start_time=time.time() ###
		points3D = self.get3Dpoints(coords, depth_1, intr, R1, T1) # project the points on img2 frame to find correspondences
		#print("---3 %s seconds ---" % (time.time() - start_time)) ###
		
		#start_time=time.time() ### 
		points2D_im2, valid_inds = self.world2img(points3D, intr, R2, T2, depth_1.shape) # points2D_im2[i,0] is width, points2D_im2[i,1] is height
		#print("---4 %s seconds ---" % (time.time() - start_time)) ###
		
		# keep the points from im1 for which there was a valid coord in im2
		points2D_im1 = coords[valid_inds,:]	# points2D_im1 and points2D_im2 contain the corresponding coordinates from the two images (positive samples)
		
		# plot to verify correspondences
		#self.verify_plot(points2D_im1, points2D_im2)
		
		if self.objSampling==1:
			# get the indices of the points that are inside the gt bboxes, in order to sample positives only from the objects
			box_inds = self.get_boxObject_inds(boxes, points2D_im1)
			#self.verify_points(points2D_im1[box_inds[0::100],:]) # uncomment to verify the sampled points
			#self.verify_plot(points2D_im1[box_inds[0::100], :], points2D_im2[box_inds[0::100], :])
			feats1, feats2, labels = self.sample_pairs(points2D_im1, points2D_im2, scores_1, scores_2, box_inds)
		else:
			#start_time=time.time() ### 
			feats1, feats2, labels = self.sample_pairs(points2D_im1, points2D_im2, scores_1, scores_2)
			#print("--- Sample pairs %s seconds ---" % (time.time() - start_time))
			
		# plot to verify the positive sampled pairs (or negative by switching labels in the function)
		#self.verify_samples(labels)
		
		# pass the top blobs
		feats1 = feats1.reshape(feats1.shape[0], feats1.shape[1], 1, 1)
		top[0].data[...] = feats1
		feats2 = feats2.reshape(feats2.shape[0], feats2.shape[1], 1, 1)
		top[1].data[...] = feats2
		labels = labels.reshape(labels.shape[0], 1, 1, 1)
		top[2].data[...] = labels
		#print feats1[0,0,0,0]
		#print feats2
		#print labels
		
	
	def backward(self, top, propagate_down, bottom):
		#print "Correspondence backwards!"
		# For now just backpropagate the gradients for the first branch
		# create a blob the size of bottom[0] and fill the gradients at the appropriate locations
		#start_time=time.time() ###
		grad_map_1 = np.zeros((bottom[0].height, bottom[0].width, bottom[0].channels, 1), dtype=np.float32)
		grad_map_2 = np.zeros((bottom[1].height, bottom[1].width, bottom[1].channels, 1), dtype=np.float32)
		labels = top[2].data
		'''
		feat_1_diff = top[0].diff[:,:,0,0]
		feat_2_diff = top[1].diff[:,:,0,0]
		for i in range(self.nSamples):
			# For each position of a gradient map we accumulate the gradient from a positive pair and the gradient from nNegPerPos pairs
			# So we need to normalize the gradients from negative examples
			if labels[i]==0:
				gradient_1 = feat_1_diff[i,:] / float(self.nNegPerPos)
				gradient_2 = feat_2_diff[i,:] / float(self.nNegPerPos)
			else:
				gradient_1  = feat_1_diff[i,:]
				gradient_2  = feat_2_diff[i,:]
			x1,y1 = self.sampled_points_1[i,0], self.sampled_points_1[i,1] 
			grad_map_1[y1,x1,:,0] = gradient_1 + grad_map_1[y1,x1,:,0]				
			x2,y2 = self.sampled_points_2[i,0], self.sampled_points_2[i,1] 
			grad_map_2[y2,x2,:,0] = gradient_2 + grad_map_2[y2,x2,:,0]
		'''
		idx_neg = np.where(labels==0)[0]
		feat_1_diff = top[0].diff[:,:,0,0]
		feat_2_diff = top[1].diff[:,:,0,0]
		feat_1_diff[idx_neg, :] = feat_1_diff[idx_neg, :] / float(self.nNegPerPos)
		feat_2_diff[idx_neg, :] = feat_2_diff[idx_neg, :] / float(self.nNegPerPos)
		x1,y1 = self.sampled_points_1[:,0], self.sampled_points_1[:,1]
		x2,y2 = self.sampled_points_2[:,0], self.sampled_points_2[:,1] 
		for i in range(self.nSamples):
			grad_map_1[y1[i],x1[i],:,0] = feat_1_diff[i,:] + grad_map_1[y1[i],x1[i],:,0]
			grad_map_2[y2[i],x2[i],:,0] = feat_2_diff[i,:] + grad_map_2[y2[i],x2[i],:,0]
						
		grad_map_1 = grad_map_1.transpose((3,2,0,1))
		grad_map_2 = grad_map_2.transpose((3,2,0,1))
		bottom[0].diff[...] = grad_map_1
		bottom[1].diff[...] = grad_map_2
		#print("---6 %s seconds ---" % (time.time() - start_time)) ###

		
	def sample_pairs(self, points2D_im1, points2D_im2, scores_1, scores_2, box_inds=None):
		# * In the future consider reshaping the vectors with the size of valid_inds, in order to accomodate pairs that do not have a large overlap
		feats1 = np.zeros((self.nSamples, self.feat_dim), dtype=np.float32)
		feats2 = np.zeros((self.nSamples, self.feat_dim), dtype=np.float32)
		labels = np.ones((self.nSamples, 1), dtype=np.float32)
		self.sampled_points_1 = np.zeros((self.nSamples, 2), dtype=np.int) # will be needed when accumulating the gradients for backprop
		self.sampled_points_2 = np.zeros((self.nSamples, 2), dtype=np.int)
		
		# get the amount of available positive samples
		if box_inds is not None:
			pos_avail = len(box_inds)
		else:
			pos_avail = len(points2D_im1)
		
		if pos_avail < self.nPositives: #self.nSamples:	
			# In the case when there are not enough correspondence between the two frames,
			# then we pass positive features with zero values so that the loss and gradients are 0
			# We also pass a single negative so that nNeg>0 and loss will not return nan
			feats1[0,:] = np.ones((self.feat_dim), dtype=np.float32)*999
			labels[0] = 0
			print "pos_avail smaller than npositives!"
		else:
			
			if box_inds is not None: # sample the positive inds only from the box_inds
				np.random.shuffle(box_inds)
				pos_samples_ind = box_inds[:self.nPositives]
			else:
				a = np.arange(points2D_im1.shape[0]) # sample from all correspondences
				np.random.shuffle(a)
				pos_samples_ind = a[:self.nPositives]				
			
			# get the positive pairs
			self.sampled_points_1[:self.nPositives, :] = points2D_im1[pos_samples_ind,:]
			self.sampled_points_2[:self.nPositives, :] = points2D_im2[pos_samples_ind,:]
			feats1[:self.nPositives, :] = scores_1[ points2D_im1[pos_samples_ind,1] , points2D_im1[pos_samples_ind,0], : ]
			feats2[:self.nPositives, :] = scores_2[ points2D_im2[pos_samples_ind,1] , points2D_im2[pos_samples_ind,0], : ]
			#labels[:self.nPositives] = 1 # no need, already ones
			
			# get the negative pairs
			neg_samples_ind = np.random.randint(points2D_im2.shape[0], size=self.nNegPerPos*self.nPositives)
			neg_samples_posind = np.repeat(pos_samples_ind, self.nNegPerPos, axis=0) # repeat each element in pos_samples_ind nNegPerPos times
			self.sampled_points_1[self.nPositives:, :] = points2D_im1[neg_samples_posind,:]
			self.sampled_points_2[self.nPositives:, :] = points2D_im2[neg_samples_ind,:]
			feats1[self.nPositives:,:] = scores_1[ points2D_im1[neg_samples_posind,1] , points2D_im1[neg_samples_posind,0], : ]
			feats2[self.nPositives:,:] = scores_2[ points2D_im2[neg_samples_ind,1] , points2D_im2[neg_samples_ind,0], : ]
			labels[self.nPositives:] = 0
			
		return feats1, feats2, labels
		
		
	def get_boxObject_inds(self, boxes, points2D_im1):
		inds_inboxes = []
		for i in range(boxes.shape[0]):
			box = boxes[i,:]
			top, left, bottom, right = box[0], box[1], box[2], box[3]
			inds1 = np.where(points2D_im1[:,0] > left)
			inds2 = np.where(points2D_im1[:,0] < right)
			inds3 = np.where(points2D_im1[:,1] > top)
			inds4 = np.where(points2D_im1[:,1] < bottom)
			idx = reduce(np.intersect1d, (inds1, inds2, inds3, inds4))
			idx = idx.tolist()
			inds_inboxes = idx + inds_inboxes
		inds_inboxes = np.asarray(inds_inboxes, dtype=np.int)
		inds_inboxes = np.unique(inds_inboxes)
		return inds_inboxes
	
	
	def get3Dpoints(self, points2D, depth, intr, R, T):
		fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
		depth = depth/1000.0
		if len(intr)>4: # check if a scale exists and use it on depth
			scale = intr[4]
			depth = depth*scale
		Rc2w = np.linalg.inv(R)
		Tc2w = np.dot(-Rc2w, np.transpose(T))
		Tc2w = Tc2w.reshape(3,1)
		#points3D = np.zeros((len(points2D), 3), dtype=np.float32)
		z = depth[points2D[:,1], points2D[:,0]]
		local3D = np.zeros((points2D.shape[0], 3), dtype=np.float32)
		a = points2D[:,0]-cx
		b = points2D[:,1]-cy
		q1 = -a[:,np.newaxis]*z[:,np.newaxis] / fx
		q2 = b[:,np.newaxis]*z[:,np.newaxis] / fy
		local3D[:,0] = q1.reshape(q1.shape[0])
		local3D[:,1] = q2.reshape(q2.shape[0])
		local3D[:,2] = z
		#local3D[:,0] = -(points2D[:,0]-cx)*z[:,np.newaxis] / fx
		#local3D[:,1] = (points2D[:,1]-cy)*z[:,np.newaxis] / fy
		#local3D[:,2] = z
		#print local3D.shape
		points3D = np.dot(Rc2w, local3D.T) + Tc2w	
		return points3D.T	

	
	def world2img(self, points3D, intr, R, T, im_dim):
		fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
		#points2D = np.zeros((len(points3D), 2), dtype=np.int)
		points2D = []
		valid_inds = []
		count=0
		local3D = np.dot(R, points3D.T) + T.reshape(3,1)
		a = -local3D[0,:]*fx
		b = local3D[1,:]*fy
		q1 = a[:,np.newaxis] / local3D[2,:][:,np.newaxis] + cx
		q2 = b[:,np.newaxis] / local3D[2,:][:,np.newaxis] + cy
		x_proj = q1.reshape(q1.shape[0])
		y_proj = q2.reshape(q2.shape[0])
		x_proj = np.round(x_proj)
		y_proj = np.round(y_proj)		
		# keep only coordinates in the image frame
		inds_1 = np.where(x_proj>=1)
		inds_2 = np.where(y_proj>=1)
		inds_3 = np.where(x_proj<im_dim[1]-1)
		inds_4 = np.where(y_proj<im_dim[0]-1)
		idx = reduce(np.intersect1d, (inds_1, inds_2, inds_3, inds_4))
		x_proj = x_proj[idx]
		y_proj = y_proj[idx]
		x_proj = x_proj.astype(np.int)
		y_proj = y_proj.astype(np.int)
		points2D = np.zeros((x_proj.shape[0], 2), dtype=int)
		points2D[:,0] = x_proj
		points2D[:,1] = y_proj
		return points2D, idx	
	
	def verify_plot(self, points2D_im1, points2D_im2):
		dataroot = "/home/george/GMU_Kitchens/"
		scene_path = dataroot + "gmu_scene_008/"
		img1 = cv2.imread(scene_path + "Images/rgb_209.png") # these are the images currently loaded by the input layer
		img2 = cv2.imread(scene_path + "Images/rgb_229.png")
		im_dim = (640,480,3)
		img1 = cv2.resize(img1, im_dim[:2]) # (960, 540)
		img2 = cv2.resize(img2, im_dim[:2])
		#vis.plot_points(img1, points2D_im1[0::1000], "points im1")
		#vis.plot_points(img2, points2D_im2[0::1000], "points im2")
		vis.plot_points(img1, points2D_im1, "points im1")
		vis.plot_points(img2, points2D_im2, "points im2")		
		plt.show()
			
	def verify_points(self, points):
		dataroot = "/home/george/GMU_Kitchens/"
		scene_path = dataroot + "gmu_scene_008/"
		img1 = cv2.imread(scene_path + "Images/rgb_209.png") # these are the images currently loaded by the input layer
		#img2 = cv2.imread(scene_path + "Images/rgb_229.png")
		im_dim = (640,480,3)
		img1 = cv2.resize(img1, im_dim[:2])
		#img2 = cv2.resize(img2, im_dim[:2])
		vis.plot_points(img1, points, "points im1")
		#vis.plot_points(img2, [], "im2")
		plt.show()	
	
	def verify_samples(self, labels):
		pos_inds = np.where(labels==1)[0]
		points1 = self.sampled_points_1[pos_inds, :]
		points2 = self.sampled_points_2[pos_inds, :]
		self.verify_plot(points1[0::100], points2[0::100])
	
	
	
# unused debugging
#print "inds from depth", len(inds[0])
#print "coords", coords.shape
#print "points3D", points3D.shape
#print "valid inds", len(valid_inds)
#print "points2D_im1", points2D_im1.shape
#print "points2D_im2", points2D_im2.shape
#raise Exception("Not enough valid points for sampling! Please ensure sufficient overlap between the pair of images!")	
	
	
'''
def get3Dpoints_slow(self, points2D, depth, intr, R, T):
	fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
	depth = depth/1000.0
	if len(intr)>4: # check if a scale exists and use it on depth
		scale = intr[4]
		depth = depth*scale
	Rc2w = np.linalg.inv(R)
	Tc2w = np.dot(-Rc2w, np.transpose(T))
	points3D = np.zeros((len(points2D), 3), dtype=np.float32)
	for i in range(len(points2D)):
		x = points2D[i][0]
		y = points2D[i][1]
		z = depth[y, x] # assume that all points sent have z>0
		local3D = np.zeros((3), dtype=np.float32)
		local3D[0] = -(x-cx)*z / fx
		local3D[1] = (y-cy)*z / fy
		local3D[2] = z
		#print "local3D:", local3D
		w3D = np.dot(Rc2w, local3D) + Tc2w
		#w3D = np.dot(R, local3D) + T
		#print "w3D:", w3D
		points3D[i,:] = w3D
	return points3D
'''	
	
	
'''
def world2img_slow(self, points3D, intr, R, T, im_dim):
	fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
	#points2D = np.zeros((len(points3D), 2), dtype=np.int)
	points2D = []
	valid_inds = []
	count=0
	for i in range(len(points3D)):
		p = points3D[i,:]
		local3D = np.dot(R, p) + T
		X,Y,Z = local3D[0],local3D[1],local3D[2]
		x_proj = round(-X*fx/Z + cx)
		y_proj = round(Y*fy/Z + cy)
		# keep only coordinates in the image frame
		#if x_proj>=0+buffer and y_proj>=0+buffer and x_proj<im_dim[1]-buffer and y_proj<im_dim[0]-buffer:
		if x_proj>=1 and y_proj>=1 and x_proj<im_dim[1]-1 and y_proj<im_dim[0]-1:
			tmp = []
			tmp.append(int(x_proj))
			tmp.append(int(y_proj))
			points2D.append(tmp)
			valid_inds.append(i)
			#points2D[count,0] = int(x_proj)
			#points2D[count,1] = int(y_proj)
			#count+=1
	points2D = np.array(points2D)
	return points2D, valid_inds
'''

'''
def sample_pairs_slow(self, points2D_im1, points2D_im2, scores_1, scores_2, box_inds=None):
	# * In the future consider reshaping the vectors with the size of valid_inds, in order to accomodate pairs that do not have a large overlap
	feats1 = np.zeros((self.nSamples, self.feat_dim), dtype=np.float32)
	feats2 = np.zeros((self.nSamples, self.feat_dim), dtype=np.float32)
	labels = np.ones((self.nSamples, 1), dtype=np.float32)
	self.sampled_points_1 = np.zeros((self.nSamples, 2), dtype=np.int) # will be needed when accumulating the gradients for backprop
	self.sampled_points_2 = np.zeros((self.nSamples, 2), dtype=np.int)

	# get the amount of available positive samples
	if box_inds is not None:
		pos_avail = len(box_inds)
	else:
		pos_avail = len(points2D_im1)

	if pos_avail < self.nPositives: #self.nSamples:	
		# In the case when there are not enough correspondence between the two frames,
		# then we pass positive features with zero values so that the loss and gradients are 0
		# We also pass a single negative so that nNeg>0 and loss will not return nan
		feats1[0,:] = np.ones((self.feat_dim), dtype=np.float32)*999
		labels[0] = 0
		print "pos_avail smaller than npositives!"
	else:

		if box_inds is not None: # sample the positive inds only from the box_inds
			np.random.shuffle(box_inds)
			pos_samples_ind = box_inds[:self.nPositives]
		else:
			a = np.arange(points2D_im1.shape[0]) # sample from all correspondences
			np.random.shuffle(a)
			pos_samples_ind = a[:self.nPositives]				

		i=0
		pos_i=0
		#start_time=time.time() ###
		# maybe choose the negative samples beforehand and figure out a way to vectorize this
		while i<self.nSamples:
			id = pos_samples_ind[pos_i] # id of the randomly chosen sample
			# store the positive
			self.sampled_points_1[i,:] = points2D_im1[id,:]
			self.sampled_points_2[i,:] = points2D_im2[id,:]
			feats1[i,:] = scores_1[ points2D_im1[id,1], points2D_im1[id,0], :]
			feats2[i,:] = scores_2[ points2D_im2[id,1], points2D_im2[id,0], :]
			labels[i] = 1
			i+=1		
			# get the pool of possible negatives, remove the current positive correspondence from points2D_im2
			neg_pool = np.delete(points2D_im2, id, 0)
			#print "neg_pool", neg_pool.shape
			rand_inds = np.random.randint(neg_pool.shape[0], size=self.nNegPerPos)
			for j in range(len(rand_inds)):
				# store the negative, choose randomly from pos_samples_ind that is not the same as id
				neg_id = rand_inds[j]
				self.sampled_points_1[i,:] = points2D_im1[id,:]
				self.sampled_points_2[i,:] = points2D_im2[neg_id,:]
				feats1[i,:] = scores_1[ points2D_im1[id,1], points2D_im1[id,0], :]
				feats2[i,:] = scores_2[ points2D_im2[neg_id,1], points2D_im2[neg_id,0], :]
				labels[i] = 0
				i+=1
			pos_i+=1
		#print("---5 %s seconds ---" % (time.time() - start_time)) ###
	return feats1, feats2, labels
'''
	
# unused snippet	
#buffer = 0 #20
# find the intersection of all coords that are inside the buffer
#inds_1 = np.where(inds[1]>=buffer)
#inds_2 = np.where(inds[0]>=buffer)
#inds_3 = np.where(inds[1]<im_size[1]-buffer)
#inds_4 = np.where(inds[0]<im_size[0]-buffer)
#idx = reduce(np.intersect1d, (inds_1, inds_2, inds_3, inds_4))
#coords[:,0] = inds[1][idx] # inds[1] is x (width coordinate)
#coords[:,1] = inds[0][idx]
	
	
	
	
	
	
	
	
	
	