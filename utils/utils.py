
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import gmu_input_data
import itertools


def apply_canny(image, sigma=0.33):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	v = np.median(blurred)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(blurred, lower, upper)
	return edged

def edge_sampling(edge_im):
	x = np.where(edge_im==255)[1]
	y = np.where(edge_im==255)[0]
	x = x[0::10]
	y = y[0::10]	
	return x, y


def grid_sampling(par, grid_step):
	a = range(0,par.im_dim[1],grid_step)
	b = range(0,par.im_dim[0],grid_step)
	c = list(itertools.product(a, b))
	q = map(list, zip(*c))
	coords = np.asarray(q)
	x = coords[1,:]
	y = coords[0,:]
	return x, y


def get_patch_feats(score_1, x, y, psize):
	if psize==1:
		feats = score_1[y, x, :]
	else:
		# sample a patch instead of a single point 
		feats = np.zeros((x.shape[0], psize*psize*score_1.shape[2]), dtype=np.float32)
		x_diff = np.array([-1,0,1,-1,0,1,-1,0,1])
		y_diff = np.array([-1,-1,-1,0,0,0,1,1,1])
		x = x.reshape(x.shape[0],1)
		y = y.reshape(y.shape[0],1)
		x_diff = x_diff.reshape(x_diff.shape[0],1)
		y_diff = y_diff.reshape(y_diff.shape[0],1)
		x_all = (x.T + x_diff).T
		y_all = (y.T + y_diff).T
		# bound the coords
		x_all[np.where(x_all<0)[0]] = 0
		x_all[np.where(x_all>score_1.shape[1]-1)[0]] = score_1.shape[1]-1
		y_all[np.where(y_all<0)[0]] = 0
		y_all[np.where(y_all>score_1.shape[0]-1)[0]] = score_1.shape[0]-1
		for i in range(x_all.shape[0]):
			feat_rect = score_1[y_all[i,:], x_all[i,:]]
			feat1 = feat_rect.reshape(feat_rect.shape[0]*feat_rect.shape[1])
			feats[i,:] = feat1
	return feats

def get_bboxes_wrgbd(fr_boxes):
	boxes = np.zeros((fr_boxes.shape[0], 4), dtype=np.int) # shape the array as nboxes x 4
	labels = np.zeros((fr_boxes.shape[0], 1), dtype=np.int)		
	for i in range(fr_boxes.shape[0]):
		# cat = fr_boxes[j][0] 
		labels[i,0] = fr_boxes[i][1]
		boxes[i,0] = fr_boxes[i][2] # top
		boxes[i,1] = fr_boxes[i][4] # left
		boxes[i,2] = fr_boxes[i][3] # bottom
		boxes[i,3] = fr_boxes[i][5] # right
	return boxes, labels	

def save_loss(loss, model_path, loss_name):
	filename = model_path + loss_name + "_loss.txt"
	f = open(filename, 'w')
	for i in range(len(loss)):
		f.write(str(loss[i])+"\n")
	f.close()
	

def save_params(par):
	att_tmp = dir(par)
	att=[]
	for i in att_tmp:
		if not i.startswith('__'):
			att.append(i)
	param_str = ""
	for i in range(len(att)):
		param_str += att[i] + ": " + str(par.__getattribute__(att[i])) + "\n"
	filename = par.train_save_path + "parameters.txt"
	f = open(filename, 'w')
	f.write(param_str)
	f.close()
	
	
def save_pairs_dist(model_path, sol, iter):
	filename_pairs_dist = model_path + "pairs_dist.txt"
	filename_avg_pairs_dist = model_path + "pairs_avg_dist.txt"
	if os.path.isfile(filename_pairs_dist): 
		f=open(filename_pairs_dist, 'a')
	else:
		f=open(filename_pairs_dist, 'w')
	if os.path.isfile(filename_avg_pairs_dist): 
		f2=open(filename_avg_pairs_dist, 'a')
	else:
		f2=open(filename_avg_pairs_dist, 'w')
	pair1 = sol.blobs['feat_1'].data
	pair2 = sol.blobs['feat_2'].data	
	labels = sol.blobs['labels'].data
	f.write("Iteration "+str(iter)+" pairs feature distances:\n")
	f2.write("Iteration "+str(iter)+" avg pairs feature distances:\n")
	avg_pos = 0
	avg_neg = 0
	nPos = len(np.where(labels == 1)[0])
	nNeg = len(np.where(labels == 0)[0])
	for i in range(len(pair1)):
		dist = np.sqrt(np.sum((pair1[i,:]-pair2[i,:])**2))
		f.write(str(dist)+", "+str(labels[i])+"\n")
		# save also the avg pair dist for positives and negatives
		if labels[i]:
			avg_pos = avg_pos + dist
		else:
			avg_neg = avg_neg + dist
	if nPos>0:		
		avg_pos = avg_pos / float(nPos)
	if nNeg>0:
		avg_neg = avg_neg / float(nNeg)
	f2.write("Pos:"+str(avg_pos)+" nPos:"+str(nPos)+"\n")
	f2.write("Neg:"+str(avg_neg)+" nNeg:"+str(nNeg)+"\n")
	f.write("\n")
	f.close()
	f2.write("\n")
	f2.close()
	
	
	