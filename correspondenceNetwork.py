
import _init_paths
import caffe
from caffe import layers as L
from caffe import params as P
import os, cv2
import collections
import numpy as np
from correspondenceParams import CorrespondenceParams
import coreNetworks
import surgery # from FCN git
import utils
import gmu_input_data
import bigbird_input_data
import wrgbd_input_data as wrgbd_inp
import visualizations as vis
import matplotlib.pyplot as plt
import time
from pyflann import *
import matching as match
import cPickle as pickle
import scipy
import glob
import scipy.io

class CorrespondenceNetwork():
	
	def __init__(self, params):
		self.params = params

	def createNetwork(self):
		n = caffe.NetSpec()
		
		# data layer
		n.data_1, n.data_2, n.depth_1, n.depth_2, n.RT_1, n.RT_2, n.intrinsic, n.boxes = L.Python(name='data', ntop=8, python_param={'module': 'gmu_input', 'layer': 'GmuInputLayer'})
	
		subnet_names = ['_1', '_2']
		data_branches = [n.data_1, n.data_2]
		for i in range(len(subnet_names)):
			name = subnet_names[i]
			#network = self.createCoreCorrespondenceNetwork(data_branches[i])
			if self.params.architect == "32S":
				network = coreNetworks.createCoreCorrespondenceNetwork_32S(data_branches[i], self.params)
			elif self.params.architect == "8S":
				network = coreNetworks.createCoreCorrespondenceNetwork_8S(data_branches[i], self.params)
			elif self.params.architect == "Scale8S":
				network = coreNetworks.createCoreCorrespondenceNetwork_Scale8S(data_branches[i], self.params)
			elif self.params.architect == "Scale4S":
				network = coreNetworks.createCoreCorrespondenceNetwork_Scale4S(data_branches[i], self.params)
			else:
				raise Exception("Unkown architecture!")
				
			for layer in network.keys():
				n.__setattr__(layer + name, network[layer])
	
		# crop layers for both branches
		#n.score_1 = L.Crop(n.upscore_1, n.data_1, axis=2, offset=19)
		#n.score_2 = L.Crop(n.upscore_2, n.data_2, axis=2, offset=19)
		# Correspondence layer
		n.feat_1, n.feat_2, n.labels = L.Python(n.score_1, n.score_2, n.RT_1, n.RT_2, n.intrinsic, n.depth_1, n.depth_2, n.boxes, ntop=3, name='correspondence',
												python_param={'module':'correspondence_layer', 'layer':'CorrespondenceLayer', 
												'param_str': "{\'nPositives\':"+str(self.params.nPositives)+", \'nNegPerPos\':"+str(self.params.nNegPerPos)+
															", \'objSampling\':"+str(self.params.objSampling)+"}"})
		# loss
		n.contr_loss = L.ContrastiveLoss(n.feat_1, n.feat_2, n.labels, contrastive_loss_param=dict(margin=self.params.margin, 
								legacy_version=self.params.contr_legacy), loss_weight=self.params.loss_weight_contr)#, legacy_version=2))
	
		with open(self.params.viewpoint_train_prototxt_file, 'w') as f:
			f.write(str(n.to_proto()))
		return f.name

	def generateSolverFile(self):
		f=open(self.params.viewpoint_solver_prototxt_file, 'w') 
		line = '{}: {}'.format('base_lr', self.params.base_lr)
		f.write(line + '\n')
		line = '{}: {}'.format('display', self.params.display)
		f.write(line + '\n')
		line = '{}: {}'.format('average_loss', self.params.average_loss)
		f.write(line + '\n')
		#line = '{}: {}'.format('debug_info', self.params.debug_info)
		#f.write(line + '\n')		
		line = '{}: {}'.format('gamma', self.params.gamma)
		f.write(line + '\n')
		line = '{}: "{}"'.format('lr_policy', self.params.lr_policy)
		f.write(line + '\n')
		line = '{}: {}'.format('max_iter', self.params.max_iter)
		f.write(line + '\n')
		line = '{}: "{}"'.format('net', os.path.normpath(self.params.viewpoint_train_prototxt_file).replace('\\', '\\\\'))
		f.write(line + '\n')
		line = '{}: {}'.format('power', self.params.power)
		f.write(line + '\n')
		line = '{}: {}'.format('stepsize', self.params.stepsize)
		f.write(line + '\n')		
		line = '{}: {}'.format('snapshot', self.params.train_snapshot_iters)
		f.write(line + '\n')
		line = '{}: "{}"'.format('snapshot_prefix', os.path.normpath(self.params.snapshot_prefix).replace('\\', '\\\\'))
		f.write(line + '\n')
		line = '{}: {}'.format('solver_mode', self.params.solver_mode)
		f.write(line + '\n')
		line = '{}: {}'.format('weight_decay', self.params.weight_decay)
		f.write(line + '\n')	
	
	
	def generateDeployPrototxtFile(self):
		"""
		Generates the deploy protoxt file
		"""
		n = caffe.NetSpec()
		im_dim = self.params.im_dim
		n.data = L.Input(shape=[dict(dim=[1, 3, im_dim[1], im_dim[0]])])
		#n.im_info = L.Input(shape=[dict(dim=[1, 3])])
		#n.depth = L.Input(shape=[dict(dim=[1, 1, im_dim[1], im_dim[0]])])
		#network = self.createCoreCorrespondenceNetwork(n.data)
		if self.params.architect == "32S":
			network = coreNetworks.createCoreCorrespondenceNetwork_32S(n.data, self.params)
		elif self.params.architect == "8S":
			network = coreNetworks.createCoreCorrespondenceNetwork_8S(n.data, self.params)
		elif self.params.architect == "Scale8S":
			network = coreNetworks.createCoreCorrespondenceNetwork_Scale8S(n.data, self.params)
		elif self.params.architect == "Scale4S":
			network = coreNetworks.createCoreCorrespondenceNetwork_Scale4S(n.data, self.params)
		else:
			raise Exception("Unkown architecture!")
			
		for layer in network.keys():
			n.__setattr__(layer + "_1", network[layer])
		basic_str = str(n.to_proto())
		with open(self.params.viewpoint_deploy_prototxt_file, 'w') as f:
			f.write(basic_str)		
	
	
	def load_pretrained_weights(self, solver):
		# Load the weights from the specified pretrained network
		# Needs to be done manually because the layer names are different
		base_net = caffe.Net(self.params.pretrained_deploy, self.params.pretrained_model, caffe.TEST)
		base_layers = base_net.params.keys()
		print base_layers
		print "Manually loading pretrained weights from: " + self.params.pretrained_model
		base_layers = [x for x in base_layers if not x.startswith("fc")] # ignore fc layers			
		for i in range(len(base_layers)):
			b_layer = base_layers[i]
			W = base_net.params[b_layer][0].data[...]
			b = base_net.params[b_layer][1].data[...]
			l1 = b_layer + "_1"
			l2 = b_layer + "_2"
			# init their weights
			solver.net.params[l1][0].data[...] = W
			solver.net.params[l1][1].data[...] = b
			solver.net.params[l2][0].data[...] = W
			solver.net.params[l2][1].data[...] = b
			print "Copying to layers " + l1 + ", " + l2
		print "Load successful!"
		return solver
	
	
	def load_pretrained_weights2(self, solver):
		base_net = caffe.Net(self.params.pretrained_deploy, self.params.pretrained_model, caffe.TEST)
		surgery.transplant(solver.net, base_net, "_1")
		surgery.transplant(solver.net, base_net, "_2")
		# surgeries
		interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
		surgery.interp(solver.net, interp_layers, self.params.interp_scale)
		if self.params.architect=="Scale8S":
			surgery.load_dil_weights(solver.net, base_net)	
		del base_net
		return solver

	
	'''
	def load_pretrained_weights_8S(self, solver):
		solver.net.copy_from(self.params.pretrained_model)
		print "Copied from base net"
		# surgeries
		interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
		surgery.interp(solver.net, interp_layers)
		return solver
	'''

	def deploy_net(self, deploy_proto, caffemodel, img):
		net = caffe.Net(deploy_proto, caffemodel, caffe.TEST)
		blob_im = gmu_input_data.prepare_im_blob(img, self.params.im_dim)
		net.blobs['data'].reshape(*blob_im.shape)
		net.blobs['data'].data[...] = blob_im
		net.forward()
		score_blob = net.blobs['score_1'].data
		score_blob = score_blob.transpose((2,3,1,0))
		score = score_blob[:,:,:,0]
		return score	
	
	def deploy_net_wrgbd(self, deploy_proto, caffemodel, img):
		net = caffe.Net(deploy_proto, caffemodel, caffe.TEST)
		blob_im = wrgbd_inp.prepare_im_blob(img)
		net.blobs['data'].reshape(*blob_im.shape)
		net.blobs['data'].data[...] = blob_im
		net.forward()
		score_blob = net.blobs['score_1'].data
		score_blob = score_blob.transpose((2,3,1,0))
		score = score_blob[:,:,:,0]
		return score

	
def find_matches(score_1, score_2, x, y, par):
	# randomly choose k points from score_1 and find their matches in score_2
	#sample_k=2000 # initial sampling of points from the image
	#x = np.random.randint(par.im_dim[0], size=sample_k)
	#y = np.random.randint(par.im_dim[1], size=sample_k)
	init_matches = np.zeros((len(x),4), dtype=np.int) # x1, y1, x2, y2
	match_dist = np.zeros((len(x)), dtype=np.float32)
	for p in range(len(x)):
		f = score_1[y[p], x[p], :]
		sub = score_2-f
		dist = np.linalg.norm(sub, axis=2)
		feat_min_dist = np.min(dist)
		match_coords = np.where(dist == feat_min_dist)
		#print match_coords[0][0], match_coords[1][0]
		init_matches[p,0], init_matches[p,1] = x[p], y[p]
		init_matches[p,2], init_matches[p,3] = match_coords[1][0], match_coords[0][0]
		match_dist[p] = feat_min_dist
	#print init_matches, match_dist
	# keep the best k matches
	#k=50
	idx = np.argsort(match_dist)
	matches = init_matches[idx,:]
	matches_dist = match_dist[idx]
	#matches = matches[:k, :]
	#matches_dist = matches_dist[:k]
	return matches, matches_dist



	

if __name__ == '__main__':
	par = CorrespondenceParams()
	corrNet = CorrespondenceNetwork(par)
	
	if par.solver_mode=="GPU":
		caffe.set_mode_gpu()
		caffe.set_device(par.gpu_id)
		#caffe.set_device([0,1])
	else:
		caffe.set_mode_cpu()
	
	
	if par.phase=="TRAIN":
		corrNet.generateSolverFile()
		corrNet_name = corrNet.createNetwork()
		# Initialize solver
		solver = caffe.SGDSolver(par.viewpoint_solver_prototxt_file)	
		solver = corrNet.load_pretrained_weights2(solver)	
		#solver = corrNet.load_pretrained_weights_8S(solver)	
		
		loss_contr=np.zeros((par.max_iter+1), dtype=float)
		utils.save_params(par) # store the training parameters for this session
		while solver.iter < par.max_iter:
			solver.step(1)		
			#print "upscore2 params", solver.net.params['upscore2_1'][0].data[0,0,:,:] 
			# take snapshot
			if solver.iter % par.train_snapshot_iters == 0:
				filename = par.train_save_path + "corrNetwork_" + str(solver.iter) + ".caffemodel"
				solver.net.save(filename)
			# store contrastive loss	
			loss_contr[solver.iter] = solver.net.blobs['contr_loss'].data	
			# visualize the contrastive loss and save the pair distances	
			if solver.iter % par.verbose_iters == 0:
				vis.plot_loss(loss_contr, solver.iter, 1, par.train_save_path, "contr")
				utils.save_pairs_dist(par.train_save_path, solver.net, solver.iter)
		# save all loss values in a txt file		
		utils.save_loss(loss_contr[1:], par.train_save_path, "contr")
	
	
	elif par.phase=="TEST_SCENE":
		corrNet.generateDeployPrototxtFile() # generate the deploy prototxt
		caffemodel = par.test_model_path + "corrNetwork_" + str(par.test_iter) + ".caffemodel"
		print "Using caffemodel: "+caffemodel
		flann = FLANN()
		#scene_id = 2 # 4, 6
		#nfiles = 718 # 704 # 441
		scene_id = [2,4,6]
		nfiles = [718,704,441]
		for sc in range(len(scene_id)):
			putative_list, precision_list, recall_list = [], [], []
			for i in range(0,nfiles[sc],5):
				img1, fr_name1, _ = gmu_input_data.pick_im(scene_id[sc], i)
				img2, fr_name2, _ = gmu_input_data.pick_im(scene_id[sc], i+10)
				# get the feature maps
				score_1 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, img1)
				score_2 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, img2)
				print "\nMatching between", fr_name1, "and", fr_name2, "..." 
				# Keypoint Sampling
				#x,y = utils.grid_sampling(par, 7) # grid step
				# sample from the edges
				im1_edge = utils.apply_canny(img1)
				x, y = utils.edge_sampling(im1_edge)			
				nFeatures = x.shape[0]
				print "nFeatures:", nFeatures

				# Feature Matching
				#init_matches, match_dist = match.get_matches_brute(par, x, y, score_1, score_2)
				init_matches, match_dist = match.get_matches_flann(flann, par, x, y, score_1, score_2)
				nPutative = init_matches.shape[0]
				#print "nPutative", nPutative

				# Geometric verification			
				inlier_list, M = match.geometric_verification(par, init_matches[:,:2], init_matches[:,2:])
				nInliers = len(inlier_list)
				nCorrespondences = match.find_correspondences(par, x, y, M) # get the available correspondences to calculate the recall later	

				putative_match_ratio = nPutative / float(nFeatures)
				precision = nInliers / float(nPutative)
				recall = nInliers / float(nCorrespondences)
				print "putative_match_ratio:", putative_match_ratio
				print "precision:", precision
				print "recall:", recall
				putative_list.append(putative_match_ratio)
				precision_list.append(precision)
				recall_list.append(recall)
				'''
				start_time=time.time() ###
				title = str(scene_id[sc])+" "+fr_name1+" "+fr_name2+"_"+par.architect+"_"+str(par.exp_test)
				vis.plot_matches(img1, img2, init_matches, title, par)			
				matches = init_matches[inlier_list, :]
				title = str(scene_id[sc])+" "+fr_name1+" "+fr_name2+"_"+par.architect+"_"+str(par.exp_test)+"_geom"
				vis.plot_matches(img1, img2, matches, title, par)
				print("---Saving the images took %s seconds ---" % (time.time() - start_time)) ###			
				'''
			# get the results over the scene
			f = open(par.save_dir+"res_"+str(scene_id[sc])+".txt", 'w')
			f.write("putative_match_ratio: "+str(sum(putative_list) / float(len(putative_list)))+"\n")
			f.write("precision: "+str(sum(precision_list) / float(len(precision_list)))+"\n")
			f.write("recall: "+str(sum(recall_list) / float(len(recall_list)))+"\n")
			f.close()
				
				
	elif par.phase=="TEST":
		# generate the deploy prototxt
		corrNet.generateDeployPrototxtFile()
		caffemodel = par.test_model_path + "corrNetwork_" + str(par.test_iter) + ".caffemodel"
		print "Using caffemodel: "+caffemodel
		flann = FLANN()
		#img1, img2 = gmu_input_data.sample_rnd_pair()
		scene_id = 2 #6 #7 #2
		fr_name1, fr_name2 = "rgb_403.png", "rgb_413.png" #"rgb_687.png", "rgb_697.png" #"rgb_106.png", "rgb_126.png"
		img1 = gmu_input_data.pick_im_file(scene_id, fr_name1)
		img2 = gmu_input_data.pick_im_file(scene_id, fr_name2)
		
		# get the feature maps
		score_1 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, img1)
		score_2 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, img2)
		#print "score_maps_diff", sum(sum(sum(score_1-score_2)))
		test_mode = 0
		
		if test_mode==0:
			# Keypoint Sampling
			#x,y = utils.grid_sampling(par, 7) # grid step
			# sample from the edges
			im1_edge = utils.apply_canny(img1)
			x, y = utils.edge_sampling(im1_edge)			
			nFeatures = x.shape[0]
			print "nFeatures:", nFeatures
			
			# Feature Matching
			#init_matches, match_dist = get_matches_brute(par, x, y, score_1, score_2)
			init_matches, match_dist = match.get_matches_flann(flann, par, x, y, score_1, score_2)
			nPutative = init_matches.shape[0]
			print "nPutative", nPutative
			
			# Geometric verification			
			inlier_list, M = match.geometric_verification(par, init_matches[:,:2], init_matches[:,2:])
			nInliers = len(inlier_list)
			nCorrespondences = match.find_correspondences(par, x, y, M) # get the available correspondences to calculate the recall later	
			
			putative_match_ratio = nPutative / float(nFeatures)
			precision = nInliers / float(nPutative)
			recall = nInliers / float(nCorrespondences)
			print "putative_match_ratio:", putative_match_ratio
			print "precision:", precision
			print "recall:", recall
			
			
			#vis.plot_points(img1, matches[:,:2], str(scene_id)+" "+fr_name1+" points")
			#vis.plot_points(img2, matches[:,2:], fr_name2+" matches")
			title = str(scene_id)+" "+fr_name1+" "+fr_name2+"_"+par.architect+"_"+str(par.exp_test)+"flann" #"scene-to-scene_12"
			#vis.plot_matches(img1, img2, matches, str(scene_id)+" "+fr_name1+" "+fr_name2, par)
			vis.plot_matches(img1, img2, init_matches, title, par)
						
			matches = init_matches[inlier_list, :]
			title = str(scene_id)+" "+fr_name1+" "+fr_name2+"_"+par.architect+"_"+str(par.exp_test)+"_flann_geom"
			vis.plot_matches(img1, img2, matches, title, par)
			plt.show()
			
		elif test_mode==1:
			# choose a particular point and check its match
			x, y = [147], [227] #[220], [230] #[267], [230] #[543], [225]
			matches = np.zeros((len(x), 4), dtype=np.int)
			for i in range(len(x)):
				f = score_1[y[i], x[i], :]
				dist = np.linalg.norm(score_2-f, axis=2)
				# normalize dist from 0..1
				#dist = (dist-np.min(dist)) / (np.max(dist)-np.min(dist))
				feat_min_dist = np.min(dist)
				print feat_min_dist
				#print dist[240,240]
				#print dist[260,510]
				#print dist[75,525]
				#print dist[50,50]
				#print np.max(dist)
				#cv2.imshow("dist", dist)
				#print dist.shape
				
				match_coords = np.where(dist == feat_min_dist)			
				matches[i,:] = x[i], y[i], match_coords[1][0], match_coords[0][0]
			print matches
			vis.plot_points(img1, matches[:,:2], str(scene_id)+" "+fr_name1+" points_"+str(x)+"-"+str(y) ) #, par)
			vis.plot_points(img2, matches[:,2:], fr_name2+" matches_"+str(x)+"-"+str(y) ) #, par)
			vis.plot_dist(dist, img2, fr_name2+" dist_map_"+str(x)+"-"+str(y), par)
			plt.show()
			#cv2.waitKey();	
	
	
	elif par.phase=="CODEBOOK_wrgbd":
		# collect features from the wrgb-d object images
		corrNet.generateDeployPrototxtFile()
		caffemodel = par.test_model_path + "corrNetwork_" + str(par.test_iter) + ".caffemodel"
		print "Using caffemodel: "+caffemodel
		object_list = sorted(glob.glob(par.wrgbd_root+"rgbd-dataset/*"))	
		label_list = [1,2,3,4,7] # confirm soda can 7
		cams = [1,2,4] # number of images per cam differs
		nFeaturesPerImage = 1000 #1000 #100
		nDesc = len(object_list)*len(cams)*nFeaturesPerImage*100 # tentative 30 images per cam
		feature_bank = np.zeros((nDesc,par.feat_dim*par.psize*par.psize), dtype=np.float32)
		label_bank = np.zeros((nDesc,1), dtype=np.int)
		coords_bank = np.zeros((nDesc, 2), dtype=np.int)
		img_id_bank = []
		feat_count=0
		for i in range(len(object_list)):
			objstr = object_list[i].split("/")[-1]
			print objstr
			inst_list = sorted(glob.glob(object_list[i]+"/*"))
			for j in range(len(inst_list)):
				inst_path = inst_list[j]
				print inst_path
				for c in range(len(cams)):
					objcam = cams[c]
					# count the number of images for that cam
					cam_imgs = sorted(glob.glob(inst_path+"/"+objstr+"_"+str(j+1)+"_"+str(objcam)+"*"))
					nImgs = len(cam_imgs)/4.0 # folder includes rgb, depth, loc, mask
					for k in range(1, int(nImgs+1), 18):
						obj_im, obj_mask = wrgbd_inp.load_obj_img(objstr, objcam, j, k, inst_path)
						score_1 = corrNet.deploy_net_wrgbd(par.viewpoint_deploy_prototxt_file, caffemodel, obj_im)
						#print score_1.shape
						y,x = np.where(obj_mask==255)[0], np.where(obj_mask==255)[1]
						inds_all = np.random.permutation(y.shape[0])
						inds = inds_all[:nFeaturesPerImage]
						y = y[inds]
						x = x[inds]
						feats = utils.get_patch_feats(score_1, x, y, par.psize)					
						#raise Exception("aa")
						for ii in range(len(feats)):
							feature_bank[feat_count, :] = feats[ii,:]
							label_bank[feat_count,0] = label_list[i]
							coords_bank[feat_count,0], coords_bank[feat_count,1] = x[ii], y[ii] 
							pathToImg = inst_path + "/" + objstr + "_" + str(j+1)+"_"+str(objcam) + "_" + str(k) + "_crop.png"
							img_id_bank.append(pathToImg) 
							feat_count+=1
		feature_bank = feature_bank[:feat_count, :]
		label_bank = label_bank[:feat_count, 0]
		coords_bank = coords_bank[:feat_count, 0]
		img_id_bank = np.asarray(img_id_bank)
		pickle.dump(feature_bank, open(par.test_model_path + "feature_bank"+par.cb_ind+".p", "wb"))
		pickle.dump(label_bank, open(par.test_model_path + "label_bank"+par.cb_ind+".p", "wb"))	
		pickle.dump(coords_bank, open(par.test_model_path + "coords_bank"+par.cb_ind+".p", "wb"))
		pickle.dump(img_id_bank, open(par.test_model_path + "img_id_bank"+par.cb_ind+".p", "wb"))
		
		
	elif par.phase=="TEST_CODEBOOK_wrgbd":
		# Load the features from the object images
		feature_bank = pickle.load(open(par.test_model_path + "feature_bank"+par.cb_ind+".p", "rb"))
		label_bank = pickle.load(open(par.test_model_path + "label_bank"+par.cb_ind+".p", "rb"))
		coords_bank = pickle.load(open(par.test_model_path + "coords_bank"+par.cb_ind+".p", "rb"))
		img_id_bank = pickle.load(open(par.test_model_path + "img_id_bank"+par.cb_ind+".p", "rb"))			
		flann = FLANN()
		# build the flann kd trees
		flann_params = flann.build_index(feature_bank, target_precision=1, algorithm='kdtree', trees=10, checks=128)		
		
		corrNet.generateDeployPrototxtFile() # generate the deploy prototxt
		caffemodel = par.test_model_path + "corrNetwork_" + str(par.test_iter) + ".caffemodel"
		print "Using caffemodel: "+caffemodel

		object_list = sorted(glob.glob(par.wrgbd_root+"rgbd-dataset/*"))
		recog_accuracy=0
		test_samples=0
		#conf_matrix = np.zeros((len(object_list)+1, len(object_list)+1), dtype=np.int)		
		scenes_path = par.wrgbd_root + "rgbd-scenes-v2/imgs/"
		annot_path = par.wrgbd_root + "rgbd-scenes-v2/annotation/"
		# test on the odd numbered scenes
		test_scenes = range(1, 14, 2)
		x, y = utils.grid_sampling(par, 1) # get the image mesh
		for sc in test_scenes:
			if sc < 10:
				sc_path = scenes_path + "scene_0" + str(sc) + "/"
				bboxes = scipy.io.loadmat(annot_path + "scene_0" + str(sc) + "_info.mat") # load the bboxes annotations
			else:
				sc_path = scenes_path + "scene_" + str(sc) + "/"
				bboxes = scipy.io.loadmat(annot_path + "scene_" + str(sc) + "_info.mat") # load the bboxes annotations	
			bboxes = bboxes['bboxes'][0] # top bottom left right
			# get nfiles for the sc scene
			#img_files = sorted(glob.glob(sc_path+"*"))
			#nfiles = int(len(img_files)/2.0) # folder includes rgb and depth
			for i in range(0,len(bboxes),5):
				if bboxes[i].shape[0]==0:
					continue				
				# im_file_path = img_file[i]
				if i < 10:
					file_num = sc_path + "0000" + str(i+1) 
				elif i < 100:
					file_num = sc_path + "000" + str(i+1)
				elif i < 1000:
					file_num = sc_path + "00" + str(i+1)
				else:
					sc_path + "0" + str(i+1)
				im_file_path = file_num + "-color.png"
				img = cv2.imread(im_file_path)
				score_1 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, img)
				print file_num
				# get the gt bboxes
				fr_boxes = bboxes[i][0]
				boxes, labelsGT = utils.get_bboxes_wrgbd(fr_boxes) # returned as top left bottom right
				for j in range(boxes.shape[0]):
					box_x, box_y = match.get_box_coords(boxes[j,:], x, y)
					print "Points sampled from box:", box_x.shape[0]
					#box_feat = score_1[box_y, box_x, :]
					box_feat = utils.get_patch_feats(score_1, box_x, box_y, par.psize)
					# match the box feat to the feature bank
					result, dists, valid_inds = match.get_box_matches_flann(flann, flann_params, par, box_feat)
					box_x = box_x[valid_inds]
					box_y = box_y[valid_inds]
					ver_labels_box = label_bank[result]
					
					if len(ver_labels_box) > 0:
						h = scipy.stats.itemfreq(ver_labels_box)
						pred_label = int(h[np.argmax(h[:,1]), 0])
						print pred_label
						print labelsGT[j,0]
						if pred_label==labelsGT[j,0]:
							recog_accuracy+=1
						#conf_matrix[labelsGT[j,0], pred_label]+=1 # update the confusion matrix
						
					test_samples+=1							
				#raise Exception("aa")

		print recog_accuracy
		print test_samples
		print recog_accuracy / float(test_samples)
		#print conf_matrix					
					
					
	
	elif par.phase=="CODEBOOK":
		# collect features from the bigBIRD object images
		corrNet.generateDeployPrototxtFile()
		caffemodel = par.test_model_path + "corrNetwork_" + str(par.test_iter) + ".caffemodel"
		print "Using caffemodel: "+caffemodel
		f = open(par.BigBIRD_root + "object_list.txt", 'r')
		object_list = f.readlines()
		degrees = range(0,357,60)
		cams = [1,2,3]
		nFeaturesPerImage = 1000 #1000 #100
		nDesc = len(object_list)*len(degrees)*len(cams)*nFeaturesPerImage
		feature_bank = np.zeros((nDesc,par.feat_dim*par.psize*par.psize), dtype=np.float32)
		label_bank = np.zeros((nDesc,1), dtype=np.int)
		coords_bank = np.zeros((nDesc, 2), dtype=np.int)
		img_id_bank = []
		feat_count=0
		for i in range(len(object_list)):
			objstr = object_list[i]
			objstr = objstr[:-1]
			for c in range(len(cams)):
				objcam = cams[c]
				for ind in degrees:
					obj_im, obj_mask = bigbird_input_data.load_obj_img(objstr, objcam, ind)
					score_1 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, obj_im)
					y,x = np.where(obj_mask==0)[0], np.where(obj_mask==0)[1]
					# randomly select nFeaturesPerImage features
					#obj_im_edge = utils.apply_canny(obj_im)
					#cv2.imshow("im1", obj_im_edge/255.0)
					#cv2.waitKey();					
					
					inds_all = np.random.permutation(y.shape[0])
					inds = inds_all[:nFeaturesPerImage]
					y = y[inds]
					x = x[inds]
					feats = utils.get_patch_feats(score_1, x, y, par.psize)
					#feats = score_1[y, x, :]
					for j in range(len(feats)):
						feature_bank[feat_count, :] = feats[j,:]
						label_bank[feat_count,0] = i+1
						coords_bank[feat_count,0], coords_bank[feat_count,1] = x[j], y[j] 
						pathToImg = par.BigBIRD_root + objstr + "/NP" + str(objcam) + "_" + str(ind) + ".jpg"
						img_id_bank.append(pathToImg) # remember the image needs resizing later
						feat_count+=1
		feature_bank = feature_bank[:feat_count, :]
		label_bank = label_bank[:feat_count, 0]
		coords_bank = coords_bank[:feat_count, 0]
		img_id_bank = np.asarray(img_id_bank)
		pickle.dump(feature_bank, open(par.test_model_path + "feature_bank"+par.cb_ind+".p", "wb"))
		pickle.dump(label_bank, open(par.test_model_path + "label_bank"+par.cb_ind+".p", "wb"))	
		pickle.dump(coords_bank, open(par.test_model_path + "coords_bank"+par.cb_ind+".p", "wb"))
		pickle.dump(img_id_bank, open(par.test_model_path + "img_id_bank"+par.cb_ind+".p", "wb"))
		
	
	elif par.phase=="TEST_CODEBOOK":
		# Load the features from the object images
		feature_bank = pickle.load(open(par.test_model_path + "feature_bank"+par.cb_ind+".p", "rb"))
		label_bank = pickle.load(open(par.test_model_path + "label_bank"+par.cb_ind+".p", "rb"))
		coords_bank = pickle.load(open(par.test_model_path + "coords_bank"+par.cb_ind+".p", "rb"))
		img_id_bank = pickle.load(open(par.test_model_path + "img_id_bank"+par.cb_ind+".p", "rb"))		
		#feature_bank = feature_bank.astype(np.float32, copy=False)
		#coords_bank = coords_bank[:feature_bank.shape[0], :]
		#img_id_bank = np.asarray(img_id_bank)
		
		#for i in range(11):
		#	print i+1, np.where(label_bank==i+1)[0].shape
		#raise Exception("AA")
		
		f = open(par.BigBIRD_root + "object_list.txt", 'r')
		object_list = f.readlines()
		
		corrNet.generateDeployPrototxtFile() # generate the deploy prototxt
		caffemodel = par.test_model_path + "corrNetwork_" + str(par.test_iter) + ".caffemodel"
		print "Using caffemodel: "+caffemodel
		flann = FLANN()
		# build the flann kd trees
		flann_params = flann.build_index(feature_bank, target_precision=1, algorithm='kdtree', trees=10, checks=128)
		scene_id = [2]#,4,6]
		nfiles = [728]#,714,451]
		recog_accuracy=0
		test_samples=0
		conf_matrix = np.zeros((len(object_list)+1, len(object_list)+1), dtype=np.int)
		#conf_matrix[0,1:] = np.asarray(range(1,len(object_list)))
		#conf_matrix[1:,0] = np.asarray(range(1,len(object_list)))
		#print conf_matrix
		# choose how to decide the label for each box. 
		# "naive" is simply counting the labels of the matches
		# "withGeom" applies geometric verification on each object image of the codebook
		test_flag = "naive" #"withGeom" # "naive"
		x, y = utils.grid_sampling(par, 1) # get the image mesh
		for sc in range(len(scene_id)):
			for i in range(0,nfiles[sc],5):		
				img1, fr_name1, im_init_dim = gmu_input_data.pick_im(scene_id[sc], i)	
				ratio_x = float(par.im_dim[0]) / float(im_init_dim[1])
				ratio_y = float(par.im_dim[1]) / float(im_init_dim[0])
				boxes, labelsGT = gmu_input_data.load_frame_gtBB(scene_id[sc], i)
				boxes = gmu_input_data.rescale_boxes(boxes, ratio_x, ratio_y)				
				score_1 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, img1)
				# get the coordinates that belong in each box, and extract only those features
				for j in range(len(boxes)):
					box_x, box_y = match.get_box_coords(boxes[j,:], x, y)
					print "Points sampled from box:", box_x.shape[0]
					#box_feat = score_1[box_y, box_x, :]
					box_feat = utils.get_patch_feats(score_1, box_x, box_y, par.psize)
					# match the box feat to the feature bank
					result, dists, valid_inds = match.get_box_matches_flann(flann, flann_params, par, box_feat)
					box_x = box_x[valid_inds]
					box_y = box_y[valid_inds]
					#result, dists, single_inds = match.keep_single_match(result, dists)
					#box_x = box_x[single_inds]
					#box_y = box_y[single_inds]
					
					if test_flag=="withGeom": # geometric verification for each image with more than 4 matches. All else matches should be discarded
						par.save_dir = par.root_dir+"examples/obj-to-scene_1/" + fr_name1 + "/box_" + str(j) + "/" 
						if not os.path.exists(par.save_dir):
							os.makedirs(par.save_dir)
						matched_imgs = img_id_bank[result]
						matched_coords = coords_bank[result, :]
						retrieved_labels = label_bank[result]		
						ver_labels_box=[]
						img_uniq = np.unique(matched_imgs) #list(set(matched_imgs))
						for m in range(len(img_uniq)):
							inds = np.where(img_uniq[m]==matched_imgs)[0] # indices to the result
							if inds.shape[0]<4: # ignore images with less than 4 matches
								continue
							print img_uniq[m], inds.shape
							obj_im_label = retrieved_labels[inds][0] # coming from the same obj image, they should all be the same
							curr_matches = np.zeros((inds.shape[0],4), dtype=np.int) # x1, y1, x2, y2
							curr_matches[:,0], curr_matches[:,1] = matched_coords[inds,0], matched_coords[inds,1] # obj_coords
							curr_matches[:,2], curr_matches[:,3] = box_x[inds], box_y[inds] # scene_coords
							# do geometric verification
							inlier_list, _ = match.geometric_verification(par, curr_matches[:,:2], curr_matches[:,2:])
							print img_uniq[m], len(inlier_list)
							print ""
							geom_matches = curr_matches[inlier_list, :]
							# keep the verified labels
							ver_labels = [obj_im_label]*len(inlier_list)
							ver_labels_box = ver_labels_box + ver_labels
							
							# visualize the match
							obj_im = cv2.imread(img_uniq[m])
							obj_im = cv2.resize(obj_im, par.im_dim[:2])
							obj_im_name = img_uniq[m].split("/")[-1]
							obj_im_name = obj_im_name.split(".")[0]
							title = object_list[obj_im_label-1] + "-" + obj_im_name
							vis.plot_matches(obj_im, img1, curr_matches, title, par)
							#vis.plot_matches(obj_im, img1, geom_matches, title, par)
							
					else:
						# naive approach just count the labels from the putative matches
						ver_labels_box = label_bank[result]
					
					if len(ver_labels_box) > 0:
						h = scipy.stats.itemfreq(ver_labels_box)
						#print h
						pred_label = int(h[np.argmax(h[:,1]), 0])
						print pred_label
						print labelsGT[j,0]
						if pred_label==labelsGT[j,0]:
							recog_accuracy+=1
						conf_matrix[labelsGT[j,0], pred_label]+=1 # update the confusion matrix
						
					test_samples+=1
						
					
					#raise Exception("***")
		print recog_accuracy
		print test_samples
		print recog_accuracy / float(test_samples)
		print conf_matrix
		
	
	elif par.phase=="TEST_OBJECT":
		# do matching between a cropped object image and a scene
		# generate the deploy prototxt
		corrNet.generateDeployPrototxtFile()
		test_iter = 12000 #20000 # 2000
		caffemodel = par.test_model_path + "corrNetwork_" + str(test_iter) + ".caffemodel"
		print "Using caffemodel: "+caffemodel	
		# load the object image
		obj_im, obj_mask = bigbird_input_data.load_obj_img("honey_bunches_of_oats_honey_roasted", 1, 0) #(objstr, objcam, objind) #2, 27
		#obj_im2, obj_mask2 = bigbird_input_data.load_obj_img("honey_bunches_of_oats_honey_roasted", 1, 234) #(objstr, objcam, objind) #2, 27
		# pick an image from the scenes
		scene_id = 2 #7
		fr_name1 = 'rgb_555.png' #"rgb_336.png"
		img1 = gmu_input_data.pick_im_file(scene_id, fr_name1)
		#img1, fr_name1 = gmu_input_data.pick_im(4, 489) # (scene_id, idx1)
		# get the feature maps
		score_1 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, obj_im)
		score_2 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, img1)	
		#score_2 = corrNet.deploy_net(par.viewpoint_deploy_prototxt_file, caffemodel, obj_im2)	
		print fr_name1
		# use the mask to select only the points in the object
		y,x = np.where(obj_mask==0)[0], np.where(obj_mask==0)[1]
		y=y[0::50]
		x=x[0::50]
		# get the sorted list of matches based on distances
		matches, matches_dist = find_matches(score_1, score_2, x, y, par)
		#k=100
		#matches = matches[:k,:]
		#matches_dist = matches_dist[:k]
		#print matches
		#print matches_dist
		#title = "obj-to-obj_2"
		title = title = str(scene_id)+" obj "+fr_name1+"_32S_obj" #"obj-to-scene_5"
		vis.plot_matches(obj_im, img1, matches, title, par)
		plt.show()
		
		
	
	elif par.phase=="DEBUG":		
		print "Debug mode!"
		
		corrNet.generateSolverFile()
		corrNet_name = corrNet.createNetwork()
		# Initialize solver
		solver = caffe.SGDSolver(par.viewpoint_solver_prototxt_file)	
		solver = corrNet.load_pretrained_weights2(solver)	
		
		solver.net.forward()
		
		print solver.net.blobs.keys()
		print "conv1_1_1", solver.net.blobs['conv1_1_1'].data.shape
		print "pool1", solver.net.blobs['pool1_1'].data.shape
		print "dil1_2_1_1", solver.net.blobs['dil1_2_1_1'].data.shape
		print "dil1_2_2_1", solver.net.blobs['dil1_2_2_1'].data.shape
		print "dil1_2_3_1", solver.net.blobs['dil1_2_3_1'].data.shape
		print "dil1_2_4_1", solver.net.blobs['dil1_2_4_1'].data.shape
		print "Sc_Pool1_2_1", solver.net.blobs['Sc_Pool1_2_1'].data.shape
		print "params"
		print "dil1_2_1_1", solver.net.params['dil1_2_1_1'][0].data[0,0,0,0]
		print "dil1_2_2_1", solver.net.params['dil1_2_2_1'][0].data[0,0,0,0]
		print "dil1_2_3_1", solver.net.params['dil1_2_3_1'][0].data[0,0,0,0]
		print "dil1_2_4_1", solver.net.params['dil1_2_4_1'][0].data[0,0,0,0]
		
		print "pool3", solver.net.blobs['pool3_1'].data.shape
		print "pool4", solver.net.blobs['pool4_1'].data.shape
		print "pool5", solver.net.blobs['pool5_1'].data.shape
		print "fc6", solver.net.blobs['fc6_1'].data.shape
		print "fc7", solver.net.blobs['fc7_1'].data.shape
		print "score_fr", solver.net.blobs['score_fr_1'].data.shape
		print "upscore2", solver.net.blobs['upscore2_1'].data.shape
		
		print "score_pool4", solver.net.blobs['score_pool4_1'].data.shape
		print "score_pool4c", solver.net.blobs['score_pool4c_1'].data.shape
		print "fuse_pool4", solver.net.blobs['fuse_pool4_1'].data.shape
		print "upscore_pool4", solver.net.blobs['upscore_pool4_1'].data.shape
		
		print "score_pool3", solver.net.blobs['score_pool3_1'].data.shape
		print "score_pool3c", solver.net.blobs['score_pool3c_1'].data.shape
		print "fuse_pool3", solver.net.blobs['fuse_pool3_1'].data.shape
		#print "upscore_pool3", solver.net.blobs['upscore_pool3_1'].data.shape
		
		#print "score_pool2", solver.net.blobs['score_pool2_1'].data.shape
		#print "score_pool2c", solver.net.blobs['score_pool2c_1'].data.shape
		#print "fuse_pool2", solver.net.blobs['fuse_pool2_1'].data.shape
		
		print "upscore8", solver.net.blobs['upscore8_1'].data.shape
		print "score", solver.net.blobs['score_1'].data.shape
		
		'''
		print "params"
		print "fc_6 params", solver.net.params['fc6_1'][0].data[0,0,0,0]
		print "fc_7 params", solver.net.params['fc7_1'][0].data[0,0,0,0]
		print "score_fr params", solver.net.params['score_fr_1'][0].data[0,0,0,0] #.shape
		print "score_pool4 params", solver.net.params['score_pool4_1'][0].data[0,0,0,0] #.shape
		print "score_pool3 params", solver.net.params['score_pool3_1'][0].data[0,0,0,0] #.shape
		print "upscore2 params", solver.net.params['upscore2_1'][0].data[0,0,0,0] #.shape
		print "upscore_pool4 params", solver.net.params['upscore_pool4_1'][0].data[0,0,0,0] #.shape
		print "upscore8 params", solver.net.params['upscore8_1'][0].data[0,0,0,0] #.shape
		'''
		
		'''
		print solver.net.blobs.keys()
		print "pool3", solver.net.blobs['pool3_1'].data.shape
		print "pool4", solver.net.blobs['pool4_1'].data.shape
		print "conv5_3", solver.net.blobs['conv5_3_1'].data.shape
		print "pool5", solver.net.blobs['pool5_1'].data.shape
		print "fc6", solver.net.blobs['fc6_1'].data.shape
		print "fc7", solver.net.blobs['fc7_1'].data.shape
		print "score_fr", solver.net.blobs['score_fr_1'].data.shape
		print "upscore", solver.net.blobs['upscore_1'].data.shape
		print "score", solver.net.blobs['score_1'].data.shape
		'''
		
		# check the dimensionalities of the output blobs
		# need to change the intrinsic parameters now that the size of the images is different
		# verify the input data
		# write the correspondence layer

		#print solver.net.blobs.keys()
		#print solver.net.blobs['score_fr_1'].data.shape
		#print solver.net.blobs['upscore_1'].data.shape
		#print solver.net.blobs['score_1'].data.shape

		#score_blob_1 = solver.net.blobs['score_1'].data
		#score_blob_1 = score_blob_1.transpose((2,3,1,0))
		#score_blob_1 = score_blob_1[:,:,:,0]
		#print score_blob_1.shape
		#im = score_blob_1[:,:,:3]
		#print im.shape
		#print np.unique(im)
		#cv2.imshow("depth", im/255.0) # imshow expects values from 0...1
		#cv2.imshow("d", im*10000000)
		#cv2.waitKey();	

		# check params values
		#print solver.net.params.keys()
		#print solver.net.params['upscore_1'][0].data # W
		#print solver.net.params['score_1'][0].data # W
	

	
	else:
		raise Exception("Unknown mode!")
	
	
	
	
	
	
	
	
	
	
	
	
	
