
import os
import glob
import numpy as np



class CorrespondenceParams(object):
	def __init__(self):
		self.root_dir = '/home/george/gmu_correspondences/'
		self.caffe_dir = self.root_dir + 'caffe-master/'
		self.model_dir = self.caffe_dir + "models/correspondence_net/"
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		self.viewpoint_train_prototxt_file = self.model_dir + "train.prototxt"
		self.viewpoint_solver_prototxt_file = self.model_dir + "solver.prototxt"
		self.viewpoint_deploy_prototxt_file = self.model_dir + "deploy.prototxt"
		
		self.pretrained_model = self.caffe_dir + "data/ilsvrc12/VGG/VGG_ILSVRC_16_layers.caffemodel"
		#self.pretrained_model = self.caffe_dir + "data/fcn8s-heavy-pascal.caffemodel"
		self.pretrained_deploy = self.caffe_dir + 'data/ilsvrc12/VGG/VGG_ILSVRC_16_layers_deploy.prototxt'
		#self.pretrained_deploy = self.caffe_dir + "data/deploy_fcn8s_heavy_pascal.prototxt"
		
		# caffe model save path
		self.exp_ind = 10 # experiment id
		self.train_save_path = self.root_dir + "output/train_net_"+str(self.exp_ind)+"/"
		if not os.path.exists(self.train_save_path):
			os.makedirs(self.train_save_path)
		
		# caffe model to use during testing
		self.exp_test = 7
		self.test_iter = 20000
		self.test_model_path = self.root_dir + "output/train_net_"+str(self.exp_test)+"/"
		if not os.path.exists(self.test_model_path):
			raise Exception("Test model does not exist!")
		
		self.cb_ind = "_wrgbd" #"" #"_patch" #"_2"
		self.psize = 1 #3 # patch size
		
		self.phase = "TEST_CODEBOOK_wrgbd" # "TRAIN" "TEST" "TEST_CODEBOOK" "TEST_CODEBOOK_wrgbd" "CODEBOOK" "CODEBOOK_wrgbd" "TEST_SCENE" "TEST_OBJECT" "DEBUG" 
		self.architect = "8S" # "32S" "8S" "Scale8S" "Scale4S"
		self.objSampling = 0 # 1 0 # enable sampling only from the objects for correspondences 
		self.std_conv = 0.001
		self.interp_scale = 0.1 #1 #0.1
		
		# caffe hyperparameters for solver
		self.momentum = 0.9
		self.weight_decay = 0.0005 #0.0004
		self.lr_policy = "step" #'inv'
		self.stepsize = 20000 # 10000
		self.base_lr = 0.001 #0.001 #0.0006
		self.gamma = 0.1 #0.0001
		self.power = 0.75
		self.solver_mode = 'GPU' # 'GPU' # 'CPU'
		self.display = 10 #10
		self.average_loss = 1
		self.debug_info = False
		#self.force_backward = True
		self.max_iter = 20000 # 20000 #1000
		self.snapshot = 0 # disable caffe snapshot
		self.viewpoint_currentIteration_max_iter=None
		self.snapshot_prefix = self.root_dir + 'output/test_model/train' # dummy prefix
		self.viewpoint_log_file = self.root_dir + 'log_test.txt'
		self.gpu_id = 1 # 0 is actually TITAN X PASCAL, 1 is GeForce GTX TITAN (shown reversed in nvidia-smi)
		self.train_snapshot_iters = 2000
		self.verbose_iters = 100 # after how many iterations to plot info about training (losses, dist)
		
		self.nPositives = 3000
		self.nNegPerPos = 32
		self.feat_dim = 12 #108 #32 #12 #16 # 3
		self.margin = 2 # 2 # 1
		self.contr_legacy = False # keep to false
		self.loss_weight_contr = 1 # 0.01
		
		self.PIXEL_MEANS_bgr = np.array((116.190, 97.203, 92.318), dtype=np.float32) # from FCN input layer
		self.dataroot = "/home/george/GMU_Kitchens/"
		self.BigBIRD_root = "/home/george/BigBIRD/"
		self.wrgbd_root = "/home/george/WRGB-D/"
		self.im_dim = (640,480,3) #(320,240,3) #(426,320,3) #(640,480,3)
		
		# testing parameters
		self.second_ratio_thresh = 0.9 # 0.9 # 0.5
		self.ransac_max_iter = 1000 #2000 #500
		self.inlier_thresh = 5 #10 # pixel difference
		self.max_inlier_thresh = 99999 # inf to get as many inliers as possible
		self.save_dir = self.root_dir+"examples/"  #"examples/obj-to-scene_1/"
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)
		
		# define the fold for training on the GMU_kitchens
		self.fold = 1
		if self.fold==1:
			self.train_scene_set = [1,3,6,7,8,9]
			self.test_scene_set = [2,4,5]
		elif self.fold==2:
			self.train_scene_set = [1,2,4,5,7,8]
			self.test_scene_set = [3,6,9]
		elif self.fold==3:	
			self.train_scene_set = [2,3,4,5,6,9] 
			self.test_scene_set = [1,7,8]
		else:
			raise Exception("Unknown fold!")
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		