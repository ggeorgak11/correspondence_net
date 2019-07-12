
# utility functions regarding matching
import numpy as np
import time
import cv2
from functools import reduce


def get_matches_brute(par, x, y, score_1, score_2):
	sample_k = x.shape[0]
	init_matches = np.zeros((sample_k,4), dtype=np.int) # x1, y1, x2, y2
	match_dist = np.zeros((sample_k), dtype=np.float32)
	mc=0	
	start_time=time.time() ###
	for p in range(len(x)):
		f = score_1[y[p], x[p], :]
		#c=score_2-f # 480 640 12
		dist = np.linalg.norm(score_2-f, axis=2)
		dist_sorted = np.sort(dist, axis=None) # sort the flatten matrix
		sec_ratio = dist_sorted[0]/dist_sorted[1] # check the second ratio criterion
		if sec_ratio >= par.second_ratio_thresh:
			continue
		match_coords = np.where(dist == dist_sorted[0])
		init_matches[mc,0], init_matches[mc,1] = x[p], y[p]
		init_matches[mc,2], init_matches[mc,3] = match_coords[1][0], match_coords[0][0]
		match_dist[mc] = dist_sorted[0]
		mc+=1
	print sample_k-mc, "matches discarded due to second ratio criterion!"
	init_matches = init_matches[:mc, :] # remove the zero entries in the matrix
	match_dist = match_dist[:mc]
	print("---Brute force matching took %s seconds ---" % (time.time() - start_time)) ###
	return init_matches, match_dist
	

def get_matches_flann(flann, par, x, y, score_1, score_2):
	# try flann kd-trees for matching
	test_feat = score_1[y,x,:]
	base_feat = score_2.reshape(score_2.shape[0]*score_2.shape[1], score_2.shape[2])
	# 'algorithm' 'kdtree' 'trees' '10'
	start_time=time.time() ###
	params = flann.build_index(base_feat, target_precision=1, algorithm='kdtree', trees=10, checks=128)
	print("---FLANN build took %s seconds ---" % (time.time() - start_time)) ###
	start_time=time.time() ###
	result, dists = flann.nn_index(test_feat, 2, checks=params["checks"])
	print("---FLANN search took %s seconds ---" % (time.time() - start_time)) ###
	# check the second ratio criterion
	sec_ratio = dists[:,0] / dists[:,1]
	valid_inds = np.where(sec_ratio < par.second_ratio_thresh)[0]
	print result.shape[0]-valid_inds.shape[0], "matches discarded due to second ratio criterion!"
	result = result[valid_inds, 0]
	dists = dists[valid_inds, 0]
	x = x[valid_inds]
	y = y[valid_inds]
	match_x = result % par.im_dim[0]
	match_y = result // par.im_dim[0]
	init_matches = np.zeros((valid_inds.shape[0],4), dtype=np.int) # x1, y1, x2, y2
	init_matches[:,0], init_matches[:,1], init_matches[:,2], init_matches[:,3] = x, y, match_x, match_y 
	return init_matches, dists


def get_box_matches_flann(flann, flann_params, par, box_feat):
	result, dists = flann.nn_index(box_feat, 2, checks=flann_params["checks"])
	# check the second ratio criterion
	sec_ratio = dists[:,0] / dists[:,1]
	valid_inds = np.where(sec_ratio < par.second_ratio_thresh)[0]
	print result.shape[0]-valid_inds.shape[0], "matches discarded due to second ratio criterion!"
	result = result[valid_inds, 0]
	dists = dists[valid_inds, 0]
	return result, dists, valid_inds


def keep_single_match(result, dists):
	# keep a single match to the result
	single_inds = []
	res_uniq = np.unique(result)	
	for i in range(len(res_uniq)):
		u_inds = np.where(result==res_uniq[i])[0]
		#print u_inds
		dist_tmp = dists[u_inds]
		#print dist_tmp
		min_dist_ind = np.argmin(dist_tmp)
		#print min_dist_ind
		min_ind = u_inds[min_dist_ind]
		single_inds.append(min_ind)
	single_inds = np.asarray(single_inds)
	result = result[single_inds]
	dists = dists[single_inds]
	return result, dists, single_inds


def find_correspondences(par, x, y, M): # find how many of the initial sampling points have correspondences in the paired image
	coords = np.zeros((x.shape[0], 2), dtype=np.int)
	coords[:,0], coords[:,1] = x, y
	proj_points = transform_points(M, coords)
	inds1 = np.where(proj_points[:,0]>=0)
	inds2 = np.where(proj_points[:,1]>=0)
	inds3 = np.where(proj_points[:,0]<par.im_dim[0])
	inds4 = np.where(proj_points[:,1]<par.im_dim[1])
	idx = reduce(np.intersect1d, (inds1, inds2, inds3, inds4))
	return idx.shape[0]
	
	
def transform_points(M, im_pts): # project the points using the found transform M
	pts1_h = np.zeros((3,im_pts.shape[0]), dtype=np.int)
	pts1_h[0:2, :] = im_pts.T
	pts1_h[2, :] = 1	
	proj_points = np.dot(M, pts1_h)
	proj_points[2,:] = proj_points[2,:] + 0.00001
	proj_points[0,:] = proj_points[0,:] / proj_points[2,:]
	proj_points[1,:] = proj_points[1,:] / proj_points[2,:] 
	proj_points = proj_points[0:2, :].T	
	return proj_points
	
	
def geometric_verification(par, im1_points, im2_points):
	start_time=time.time() ###
	im1_points = im1_points.astype(np.float32, copy=False)
	im2_points = im2_points.astype(np.float32, copy=False)
	max_iters = par.ransac_max_iter
	max_inliers = -1
	nSub = 4 #int(im1_points.shape[0]/3) # subsample a third of the points to find the transformation
	for i in range(max_iters):
		rand_inds = np.random.permutation(im1_points.shape[0])
		sel = rand_inds[:nSub]
		pts1 = im1_points[sel,:]
		pts2 = im2_points[sel,:]
		M = cv2.getPerspectiveTransform(pts1,pts2)
		proj_points = transform_points(M, im1_points)
		dist2D = np.linalg.norm(im2_points-proj_points, axis=1)
		inlier_list = np.where(dist2D<=par.inlier_thresh)[0]
		nInliers = inlier_list.shape[0]		
		if nInliers >= par.max_inlier_thresh:
			print "Found", nInliers, "nInliers after", i+1, "iterations!"
			return inlier_list, M
		if nInliers > max_inliers:
			max_inliers = nInliers
			best_inlist = inlier_list
			best_M = M
	print "Found", max_inliers, "nInliers after", max_iters, "iterations!"
	print("---Geometric verification took %s seconds ---" % (time.time() - start_time)) ###
	return best_inlist, best_M



def get_box_coords(box, x, y):
	top, left, bottom, right = box[0], box[1], box[2], box[3]
	inds1 = np.where(x > left)
	inds2 = np.where(x < right)
	inds3 = np.where(y > top)
	inds4 = np.where(y < bottom)
	idx = reduce(np.intersect1d, (inds1, inds2, inds3, inds4))	
	return x[idx], y[idx]





























