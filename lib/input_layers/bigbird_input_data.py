
import numpy as np
import h5py
import cv2
import sys
import os
#sys.path.insert(0, '/home/george/gmu_correspondences/')
from correspondenceParams import CorrespondenceParams
params = CorrespondenceParams()


def load_obj_img(objstr, cam, ind):
	object_img_path = params.BigBIRD_root + objstr + "/"
	obj_im = cv2.imread(object_img_path + "NP" + str(cam) + "_" + str(ind) + ".jpg")
	obj_im = cv2.resize(obj_im, params.im_dim[:2])
	# check if a refined masks exists first, if not load the one provided by BigBIRD
	ref_mask_path = params.BigBIRD_root + "refined_masks/" + objstr + "/NP" + str(cam) + "_" + str(ind) + "_mask_refined.png"
	if os.path.isfile(ref_mask_path):
		obj_mask = cv2.imread(ref_mask_path)
		obj_mask = 255 - obj_mask
	else:
		object_mask_path = params.BigBIRD_root + objstr + "/masks/"
		obj_mask = cv2.imread(object_mask_path + "NP" + str(cam) + "_" + str(ind) + "_mask.pbm")
	obj_mask = obj_mask[:,:,0]
	obj_mask = cv2.resize(obj_mask, params.im_dim[:2])
	# need to threshold the mask because interpolation changes its values
	obj_mask[np.where(obj_mask>=128)[0], np.where(obj_mask>=128)[1]] = 255
	obj_mask[np.where(obj_mask<128)[0], np.where(obj_mask<128)[1]] = 0
	#print np.unique(obj_mask)
	#cv2.imshow("objim", obj_im/255.0) # imshow expects values from 0...1
	#cv2.imshow("mask", obj_mask)
	#cv2.waitKey();
	return obj_im, obj_mask

def load_depth_img(objstr, cam, ind):
	object_img_path = params.BigBIRD_root + objstr + "/"
	f = h5py.File(object_img_path + "NP" + str(cam) + "_" + str(ind) + ".h5")
	depth = np.asarray(f['depth'], dtype=np.float32)
	depth = cv2.resize(depth, params.im_dim[:2])
	#print np.unique(depth)
	#cv2.imshow("depth", depth)
	#cv2.waitKey();
	return depth

def load_pose(objstr, cam , ind):
	object_pose_path = params.BigBIRD_root + objstr + "/poses/"
	f = h5py.File(object_pose_path + "NP" + str(cam) + "_" + str(ind) + "_pose.h5")
	# keys are 'H_table_from_reference_camera', 'board_frame_offset'
	ref = np.asarray(f['H_table_from_reference_camera'], dtype=np.float32)
	offset = np.asarray(f['board_frame_offset'], dtype=np.float32)
	print ref.shape
	print ref
	print offset
	
	
def load_intrinsic(objstr):
	object_calib_path = params.BigBIRD_root + objstr + "/"
	data = h5py.File(object_calib_path + "calibration.h5")
	keys = data.keys()
	#print np.asarray(f['H_NP1_from_NP5'], dtype=np.float32)
	f = open("calibration.txt", 'w')
	for i in range(len(keys)):
		f.write(keys[i]+"\n")
		d = np.asarray(data[keys[i]])
		f.write(str(d)+"\n")
		f.write("\n")
		
	
def sample_pair_blobs():
	# for now stick to the cereal box
	objstr = "honey_bunches_of_oats_honey_roasted"
	cam_seq_train = [1, 3, 5]
	ind_seq = range(0,330,3)
	cam = cam_seq_train[np.random.randint(len(cam_seq_train), size=1)[0]]
	ind1 = ind_seq[np.random.randint(len(ind_seq), size=1)[0]]
	ind2 = ind1+30 # +30 degrees azimuth difference
	obj_im1, obj_mask1 = load_obj_img(objstr, cam, ind1)
	obj_im2, obj_mask2 = load_obj_img(objstr, cam, ind2) 
	depth1 = load_depth_img(objstr, cam, ind1)
	depth2 = load_depth_img(objstr, cam, ind2)
	# in order for the correspondence layer to sample only from the objects, use the mask to set depth=0
	# to any points outside the mask. ** Would this cause the network weights to change only in the center of the image? 
	depth1[np.where(obj_mask1==255)[0], np.where(obj_mask1==255)[1]] = 0
	depth2[np.where(obj_mask2==255)[0], np.where(obj_mask2==255)[1]] = 0
	
	load_pose(objstr, cam=5 , ind=ind1)
	load_intrinsic(objstr) # figure out how to project points from image to image
	
	# get the blobs
	
	# Need aggresive data augmentation 
	
	
#sample_pair_blobs()




























