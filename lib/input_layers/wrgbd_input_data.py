
import numpy as np
import cv2
import sys
import os
#sys.path.insert(0, '/home/george/gmu_correspondences/')
from correspondenceParams import CorrespondenceParams
params = CorrespondenceParams()


def load_obj_img(objstr, cam, inst, ind, inst_path): #objstr, inst, 
	obj_img_path = inst_path + "/" + objstr + "_" + str(inst+1)+"_"+str(cam) + "_" + str(ind) + "_crop.png"
	#params.wrgbd_root + "rgbd-dataset/" + objstr + "/" + objstr + "_" + str(inst) + "/"
	obj_im = cv2.imread(obj_img_path)
	obj_mask_path = inst_path + "/" + objstr + "_" + str(inst+1)+"_"+str(cam) + "_" + str(ind) + "_maskcrop.png"
	obj_mask = cv2.imread(obj_mask_path)
	
	#obj_im = cv2.resize(obj_im, params.im_dim[:2])
	# check if a refined masks exists first, if not load the one provided by BigBIRD
	'''
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
	'''
	#print np.unique(obj_mask)
	#cv2.imshow("objim", obj_im/255.0) # imshow expects values from 0...1
	#cv2.imshow("mask", obj_mask)
	#cv2.waitKey();
	return obj_im, obj_mask





def prepare_im_blob(im):
	img_dim = im.shape
	im = im.astype(np.float32, copy=False)
	im = im[:,:,::-1] # switch RGB to BGR, following the FCN input layer
	im -= params.PIXEL_MEANS_bgr
	blob = np.zeros((1, img_dim[0], img_dim[1], img_dim[2]), dtype=np.float32)
	blob[0, :, :, :] = im	
	# channels need to be swapped to (nChannels, height, width)
	channel_swap = (0, 3, 1, 2)
	blob = blob.transpose(channel_swap)
	return blob	












