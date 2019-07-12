import scipy.io
import numpy as np
import cv2, os
import PIL.Image
import sys
sys.path.insert(0, '/home/george/gmu_correspondences/')
from correspondenceParams import CorrespondenceParams
params = CorrespondenceParams()

# Utility functions for reading the data from the GMU Kitchens dataset

def get_blobs():
	blobs={}
	# get the blobs ready for caffe
	blob_im1, blob_im2, blob_depth1, blob_depth2, blob_RT1, blob_RT2, blob_intrinsic, blob_boxes = sample_pair_blobs()
	# keep a single scale for the images for now
	blobs['data_1'] = blob_im1
	blobs['data_2'] = blob_im2
	blobs['depth_1'] = blob_depth1
	blobs['depth_2'] = blob_depth2
	blobs['RT_1'] = blob_RT1
	blobs['RT_2'] = blob_RT2
	blobs['intrinsic'] = blob_intrinsic
	blobs['boxes'] = blob_boxes
	return blobs	

def get_im_dims():
	return params.im_dim

def load_frame_gtBB(scene_idx, idx):
	path = params.dataroot + "gmu-kitchens_info/scene_annotation/bboxes/scene_" + str(scene_idx) + "_annotated_bboxes.mat"
	bboxes = scipy.io.loadmat(path)
	fr_boxes = bboxes['bboxes'][0, idx]
	boxes = np.zeros((fr_boxes.shape[1], 4), dtype=np.int) # shape the array as nboxes x 4
	labels = np.zeros((fr_boxes.shape[1], 1), dtype=np.int)
	for i in range(fr_boxes.shape[1]):
		info = fr_boxes[0,i]
		# info[0][0][0][0] # category
		labels[i,0] = info[1][0][0] # label
		boxes[i,0] = info[2][0][0] # top
		boxes[i,1] = info[3][0][0] # left
		boxes[i,2] = info[4][0][0] # bottom
		boxes[i,3] = info[5][0][0] # right
	return boxes, labels


def load_pose_struct(scene_idx):
	path = params.dataroot + "gmu-kitchens_info/scene_pose_info/scene_"+str(scene_idx)+"_reconstruct_info_frame_sort.mat"
	poses = scipy.io.loadmat(path)
	frames = poses['frames'][0,:]
	# frames [idx][0] : ignore
	# frames [idx][1] : intrinsics [f, k1, k2] which we ignore
	# frames [idx][2] : Rw2c
	# frames [idx][3] : Tw2c
	# frames [idx][4] : image name
	return frames

def get_frame_info(idx, frames):
	if len(frames[idx]) < 5: # certain scenes frames struct are missing the first element 
		R = frames[idx][1]
		T = frames[idx][2][0]
		frame_name = frames[idx][3][0]
	else:
		R = frames[idx][2]
		T = frames[idx][3][0]
		frame_name = frames[idx][4][0]
	return frame_name, R, T
	
def save_frame_info_training(scene_id, fr_name1, fr_name2):
	filepath = params.train_save_path + "img_sequence.txt"
	if not os.path.isfile(filepath):
		f = open(filepath, 'w')
	else:
		f = open(filepath, 'a')
	line = str(scene_id) + " " + fr_name1 + " " + fr_name2 + "\n"
	f.write(line)

	
def load_rgb_imgs(scene_path, fr_name1, fr_name2):	
	# load the images
	img1 = cv2.imread(scene_path + "Images/" + fr_name1)
	img2 = cv2.imread(scene_path + "Images/" + fr_name2)
	init_shape = img1.shape
	img1 = cv2.resize(img1, params.im_dim[:2]) # (960, 540)
	img2 = cv2.resize(img2, params.im_dim[:2])	
	return img1, img2, init_shape


def load_rgb_im(scene_path, fr_name1):
	# load a single rgb image
	img1 = cv2.imread(scene_path + "Images/" + fr_name1)
	init_shape = img1.shape
	img1 = cv2.resize(img1, params.im_dim[:2])
	return img1, init_shape


def load_depth_imgs(scene_path, fr_name1, fr_name2):
	depth_name1 = 'depth' + fr_name1[3:]
	depth_name2 = 'depth' + fr_name2[3:]
	depth_data1 = PIL.Image.open(scene_path + "Depths/" + depth_name1)
	depth1 = np.array(depth_data1, dtype=np.float32)
	#cv2.imshow("d", depth1)
	depth_data2 = PIL.Image.open(scene_path + "Depths/" + depth_name2)   
	depth2 = np.array(depth_data2, dtype=np.float32)
	depth1 = cv2.resize(depth1, params.im_dim[:2])
	depth2 = cv2.resize(depth2, params.im_dim[:2])
	return depth1, depth2


def pick_im(scene_id, idx):
	frames = load_pose_struct(scene_id)
	fr_name1, R1, T1 = get_frame_info(idx, frames)
	scene_path = params.dataroot + "gmu_scene_00" + str(scene_id) + "/"
	im, init_shape = load_rgb_im(scene_path, fr_name1)
	return im, fr_name1, init_shape


def pick_im_file(scene_id, fr_name):
	scene_path = params.dataroot + "gmu_scene_00" + str(scene_id) + "/"
	im, _ = load_rgb_im(scene_path, fr_name)
	return im

def sample_rnd_pair():
	# read a random test pair from the list of test scenes
	#scene_id = par.test_scene_set[np.random.randint(len(par.test_scene_set), size=1)[0]]
	scene_id = params.train_scene_set[np.random.randint(len(params.train_scene_set), size=1)[0]]
	#scene_id = 4
	frames = load_pose_struct(scene_id)
	idx1 = np.random.randint(len(frames)-10, size=1)[0]
	idx2 = idx1+10
	#idx1 = 350
	#idx2 = 360
	fr_name1, R1, T1 = get_frame_info(idx1, frames)
	fr_name2, R2, T2 = get_frame_info(idx2, frames)
	#print scene_id, fr_name1, fr_name2
	scene_path = params.dataroot + "gmu_scene_00" + str(scene_id) + "/"
	img1, img2, _ = load_rgb_imgs(scene_path, fr_name1, fr_name2)
	#cv2.imshow("im1", img1/255.0) # imshow expects values from 0...1
	#cv2.imshow("im2", img2/255.0)
	#cv2.waitKey();	
	return img1, img2

	
def sample_pair_blobs():
	# randomly select a pair from one of the training scenes
	scene_id = params.train_scene_set[np.random.randint(len(params.train_scene_set), size=1)[0]] #1 # random training scene selection
	#scene_id = 8
	frames = load_pose_struct(scene_id)
	idx1 = np.random.randint(len(frames)-10, size=1)[0]
	idx2 = idx1+10
	#idx1 = 206
	#idx2 = 226
	fr_name1, R1, T1 = get_frame_info(idx1, frames)
	fr_name2, R2, T2 = get_frame_info(idx2, frames)
	#print fr_name1, fr_name2
	# load the gtBBoxes for the first frame, for sampling purposes in the correspondence layer
	boxes, _ = load_frame_gtBB(scene_id, idx1)
	#print fr_name1, fr_name2
	save_frame_info_training(scene_id, fr_name1, fr_name2)
	# load the scales needed for projecting the points
	f = open(params.dataroot+"scales.txt", 'r')
	scale = float(f.readlines()[scene_id-1])
	
	# load the images and depths
	scene_path = params.dataroot + "gmu_scene_00" + str(scene_id) + "/"
	img1, img2, im_init_dim = load_rgb_imgs(scene_path, fr_name1, fr_name2)	
	depth1, depth2 = load_depth_imgs(scene_path, fr_name1, fr_name2)
	#print np.unique(depth1)
	#cv2.imshow("depth", img1/255.0) # imshow expects values from 0...1
	#cv2.imshow("d", depth1)
	#cv2.waitKey();	

	# prepare the blobs
	blob_im1 = prepare_im_blob(img1, params.im_dim)
	blob_im2 = prepare_im_blob(img2, params.im_dim)
	blob_depth1 = prepare_depth_blob(depth1, params.im_dim)
	blob_depth2 = prepare_depth_blob(depth2, params.im_dim)
	blob_RT1 = prepare_pose_blob(R1, T1)
	blob_RT2 = prepare_pose_blob(R2, T2)
	ratio_x = float(params.im_dim[0]) / float(im_init_dim[1])
	ratio_y = float(params.im_dim[1]) / float(im_init_dim[0])
	#print "ratios", ratio_x, ratio_y
	blob_intrinsic = prepare_intrinsic_blob(ratio_x, ratio_y, scale)
	boxes = rescale_boxes(boxes, ratio_x, ratio_y)
	blob_boxes = prepare_boxes_blob(boxes)
	return blob_im1, blob_im2, blob_depth1, blob_depth2, blob_RT1, blob_RT2, blob_intrinsic, blob_boxes


def rescale_boxes(boxes, ratio_x, ratio_y):
	for i in range(boxes.shape[0]):
		boxes[i,0] = int(boxes[i,0]*ratio_y)
		boxes[i,1] = int(boxes[i,1]*ratio_x)
		boxes[i,2] = int(boxes[i,2]*ratio_y)
		boxes[i,3] = int(boxes[i,3]*ratio_x)
	return boxes

	
def prepare_boxes_blob(boxes):
	blob = np.zeros((1, boxes.shape[0], boxes.shape[1], 1), dtype=np.float32)
	blob[0,:,:,0] = boxes
	channel_swap = (0, 3, 1, 2)
	blob = blob.transpose(channel_swap)
	return blob	
	
def prepare_im_blob(im, img_dim):
	im = im.astype(np.float32, copy=False)
	im = im[:,:,::-1] # switch RGB to BGR, following the FCN input layer
	im -= params.PIXEL_MEANS_bgr
	blob = np.zeros((1, img_dim[1], img_dim[0], img_dim[2]), dtype=np.float32)
	blob[0, :, :, :] = im	
	# channels need to be swapped to (nChannels, height, width)
	channel_swap = (0, 3, 1, 2)
	blob = blob.transpose(channel_swap)
	return blob	
	
def prepare_depth_blob(depth, img_dim):
	blob = np.zeros((1, img_dim[1], img_dim[0], 1), dtype=np.uint16)
	blob[0, 0:depth.shape[0], 0:depth.shape[1], 0] = depth
	# channels need to be swapped to (nChannels, height, width)
	channel_swap = (0, 3, 1, 2)
	blob = blob.transpose(channel_swap)
	return blob
	
def prepare_pose_blob(R, T):
	R = np.reshape(R, (R.shape[0]*R.shape[1],1)) # 9 x 1
	T = np.reshape(T, (T.shape[0], 1)) # 3 x 1
	RT = np.concatenate((R.T, T.T), axis=1)
	blob = np.zeros((1, RT.shape[0], RT.shape[1], 1), dtype=np.float32) # 1 x 1 x 12 x 1
	blob[0, :, :, 0] = RT
	# channels need to be swapped to (nChannels, height, width)
	channel_swap = (0, 3, 1, 2)
	blob = blob.transpose(channel_swap)
	return blob

def prepare_intrinsic_blob(ratio_x, ratio_y, scale):
	# set the intrinsics as constant [fx,fy,cx,cy], from GMU_Kitchens documentation
	intr = np.array((1.0477637710998533e+03, 1.0511749325842486e+03, 9.5926120509632392e+02, 5.2911546499433564e+02, scale), dtype=np.float32)
	# because we had to resize the images, we need to rescale the intrinsic params
	intr[0] = intr[0]*ratio_x
	intr[1] = intr[1]*ratio_y
	intr[2] = intr[2]*ratio_x
	intr[3] = intr[3]*ratio_y
	intr = np.reshape(intr, (intr.shape[0], 1))
	intr = intr.transpose((1, 0)) # make 1 x 4
	blob = np.zeros((1, intr.shape[0], intr.shape[1], 1), dtype=np.float32)
	blob[0, :, :, 0] = intr
	channel_swap = (0, 3, 1, 2)
	blob = blob.transpose(channel_swap)
	return blob	
	
	
#sample_pair_blobs()	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	