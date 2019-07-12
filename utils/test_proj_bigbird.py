import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py
import sys
sys.path.insert(0, '/home/george/gmu_correspondences/')
from correspondenceParams import CorrespondenceParams
params = CorrespondenceParams()


def load_obj_img(objstr, cam , ind, im_dim):
	object_img_path = params.BigBIRD_root + objstr + "/"
	obj_im = cv2.imread(object_img_path + "NP" + str(cam) + "_" + str(ind) + ".jpg")
	im_orig_dim = obj_im.shape
	obj_im = cv2.resize(obj_im, im_dim[:2])
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
	return obj_im, obj_mask, im_orig_dim

def load_depth_img(objstr, cam, ind):
	object_img_path = params.BigBIRD_root + objstr + "/"
	f = h5py.File(object_img_path + "NP" + str(cam) + "_" + str(ind) + ".h5")
	depth = np.asarray(f['depth'], dtype=np.float32)
	depth = depth / 10000.0
	#print depth.shape
	#depth = cv2.resize(depth, params.im_dim[:2])
	#print np.unique(depth)
	#cv2.imshow("depth", depth)
	#cv2.waitKey();
	return depth

def load_pose(objstr, ind):
	object_pose_path = params.BigBIRD_root + objstr + "/poses/"
	f = h5py.File(object_pose_path + "NP5_" + str(ind) + "_pose.h5")
	# keys are 'H_table_from_reference_camera', 'board_frame_offset'
	ref = np.asarray(f['H_table_from_reference_camera'], dtype=np.float32)
	offset = np.asarray(f['board_frame_offset'], dtype=np.float32)
	offset = offset.reshape(1,3)
	ref_R = ref[:3,:3]
	ref_T = ref[:3, 3]
	ref_T = ref_T.reshape(3,1)
	return ref_R, ref_T, offset

def load_cam_transform(objstr, cam):
	# get the transformation of cam from the reference cam
	object_calib_path = params.BigBIRD_root + objstr + "/"
	data = h5py.File(object_calib_path + "calibration.h5")
	key = 'H_NP' + str(cam) + '_from_NP5'
	M = np.asarray(data[key], dtype=np.float32)
	R = M[:3, :3]
	T = M[:3, 3]
	T = T.reshape(3,1)
	return R, T
	
def invert_transform(R, T):
	Rinv = np.linalg.inv(R)
	Tinv = np.dot(-Rinv, T)
	#Tinv = Tinv.reshape(3,1)
	return Rinv, Tinv	
	
	
def load_intrinsic(objstr, cam):
	object_calib_path = params.BigBIRD_root + objstr + "/"
	data = h5py.File(object_calib_path + "calibration.h5")
	#keys = data.keys()
	key = 'NP' + str(cam) + "_rgb_K"
	#key = 'NP' + str(cam) + "_depth_K"
	K = np.asarray(data[key], dtype=np.float32)
	#print np.asarray(f['H_NP1_from_NP5'], dtype=np.float32)
	intr = np.zeros((4), dtype=np.float32)
	intr[0], intr[1], intr[2], intr[3] = K[0,0], K[1,1], K[0,2], K[1,2]
	return intr 

def getCam3D(points2D, depth, intr):
	fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
	cam3D = np.zeros((len(points2D), 3), dtype=np.float32)
	for i in range(len(points2D)):
		x = points2D[i,0]
		y = points2D[i,1]
		z = depth[y,x]
		local3D = np.zeros((3), dtype=np.float32)
		local3D[0] = -(x-cx)*z / fx
		local3D[1] = (y-cy)*z / fy
		local3D[2] = z
		cam3D[i,:] = local3D
	return cam3D

def camToImg(cam3D, intr):
	fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
	points2D = np.zeros((len(cam3D), 2), dtype=np.int)
	for i in range(len(cam3D)):
		X,Y,Z = cam3D[i,0],cam3D[i,1],cam3D[i,2]
		x_proj = round(-X*fx/Z + cx)
		y_proj = round(Y*fy/Z + cy)
		points2D[i,0] = int(x_proj)
		points2D[i,1] = int(y_proj)
	return points2D
	
def applyTransform(cam3D, R, T):
	w3D = np.dot(R, cam3D.T) + T
	return w3D.T

def scale_intrinsic(intr, im_dim, im_orig_dim):
	ratio_x = float(im_dim[0]) / float(im_orig_dim[1])
	ratio_y = float(im_dim[1]) / float(im_orig_dim[0])
	intr[0] = intr[0]*ratio_x
	intr[1] = intr[1]*ratio_y
	intr[2] = intr[2]*ratio_x
	intr[3] = intr[3]*ratio_y
	return intr

def plot_points(im, points, title):
    """Draw detected bounding boxes."""
    color_str = "orange"
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in range(len(points)):
		plt.plot([points[i,0]], [points[i,1]], marker='*', markersize=15, color=color_str)
		ax.text(points[i,0]+5, points[i,1]-10, str(i), fontsize=15, color=color_str)
    ax.set_title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.savefig(img_out_dir + title + ".png")
    #plt.close()		
		

		
objstr = "honey_bunches_of_oats_honey_roasted"

#object_calib_path = params.BigBIRD_root + objstr + "/poses/"
#data = h5py.File(object_calib_path + "turntable.h5")
#keys = data.keys()
#print np.asarray(data['center'], dtype=np.float32)
#print np.asarray(data['normal'], dtype=np.float32)

cam1=1
cam2=3
ind1 = 60
ind2 = 60
im_dim = (640,480,3)
obj_im1, obj_mask1, im_orig_dim = load_obj_img(objstr, cam1, ind1, im_dim)
obj_im2, obj_mask2, im_orig_dim = load_obj_img(objstr, cam2, ind2, im_dim) 
depth1 = load_depth_img(objstr, cam1, ind1)
depth2 = load_depth_img(objstr, cam2, ind2)

points = np.array([[305,195], [370,325]], dtype=np.int)
plot_points(obj_im1, points, "im1 points")
plot_points(obj_im2, points, "im2 points")


# project 2D points in camera1 coordinate frame
intr1 = load_intrinsic(objstr, cam1)
intr1 = scale_intrinsic(intr1, im_dim, im_orig_dim)
cam_1_3D = getCam3D(points, depth1, intr1)
print cam_1_3D

# transform the points in the reference camera coordinate frame
R1, T1 = load_cam_transform(objstr, cam1)
R1_inv, T1_inv = invert_transform(R1, T1) # pose is given from reference to cam1 so we need to invert it
w3D = applyTransform(cam_1_3D, R1_inv, T1_inv)
print w3D
#ref_R, ref_T, offset = load_pose(objstr, ind2) # we might need to apply transform from reference camera to table
#w3D = applyTransform(w3D, ref_R, ref_T)
#ratio_x = float(im_dim[0]) / float(im_orig_dim[1])
#ratio_y = float(im_dim[1]) / float(im_orig_dim[0])
#offset[0,0] = offset[0,0]*ratio_x
#offset[0,1] = offset[0,1]*ratio_y
#print offset
#w3D = w3D - offset


# transform the points from the reference to the camera2 coordinate frame
R2, T2 = load_cam_transform(objstr, cam2)
cam_2_3D = applyTransform(w3D, R2, T2)
print cam_2_3D


# project points in camera2 coordinate frame on the image
intr2 = load_intrinsic(objstr, cam2)
intr2 = scale_intrinsic(intr2, im_dim, im_orig_dim)
points2D = camToImg(cam_2_3D, intr2)
print points2D

plot_points(obj_im2, points2D, "im2 projected points")

plt.show()	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		