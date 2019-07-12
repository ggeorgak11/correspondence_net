
import scipy.io
import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt


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


def get_frame_info(idx, frames):
	R = frames[idx][2]
	T = frames[idx][3][0]
	k1, k2 = frames[idx][1][0,1], frames[idx][1][0,2]
	frame_name = frames[idx][4][0]
	return frame_name, R, T, k1, k2

def load_pose_struct(scene_idx, dataroot):
	path = dataroot + "gmu-kitchens_info/scene_pose_info/scene_"+str(scene_idx)+"_reconstruct_info_frame_sort.mat"
	poses = scipy.io.loadmat(path)
	frames = poses['frames'][0,:]
	# frames [idx][0] : ignore
	# frames [idx][1] : intrinsics [f, k1, k2] which we ignore
	# frames [idx][2] : Rw2c
	# frames [idx][3] : Tw2c
	# frames [idx][4] : image name
	return frames

def get3Dpoints(points2D, depth, intr, R, T):
	fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
	depth = depth/1000.0
	if len(intr)>4: # check if a scale exists and use it on depth
		scale = intr[4]
		depth = depth*scale
	Rc2w = np.linalg.inv(R)
	Tc2w = np.dot(-Rc2w, np.transpose(T))
	points3D = np.zeros((len(points2D), 3), dtype=np.float32)
	for i in range(len(points2D)):
		x = points2D[i][0] # need to verify x,y
		y = points2D[i][1]
		z = depth[y, x]
		#print z
		#if z==0:
		#	continue
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


def world2img(points3D, intr, R, T, k1, k2, buffer):
	fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
	points2D = np.zeros((len(points3D), 2), dtype=np.int)
	for i in range(len(points3D)):
		p = points3D[i,:]
		local3D = np.dot(R, p) + T
		X,Y,Z = local3D[0],local3D[1],local3D[2]
		x_proj = round(-X*fx/Z + cx)
		y_proj = round(Y*fy/Z + cy)
		#print "Point", i, "reprojected local", local3D
		'''
		# In case you want to use the distortion parameters follow this code
		# But the distortion parameters are very very small (rloc~=1.00001) so their effect is negligible
		loc = local3D/local3D[2]
		rloc = 1.0 + k1*np.power(np.linalg.norm(loc),2) + k2*np.power(np.linalg.norm(loc),4)
		x_proj = round(-loc[0]*fx*rloc + cx)
		y_proj = round(loc[1]*fy*rloc + cy)
		'''
		points2D[i,0] = int(x_proj)
		points2D[i,1] = int(y_proj)
	return points2D


def world2img_layer(points3D, intr, R, T, im_dim, buffer):
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
		if x_proj>=0+buffer and y_proj>=0+buffer and x_proj<im_dim[1]-buffer and y_proj<im_dim[0]-buffer:
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



# Load two images and try to project a point
scene_id=3
dataroot = "/home/george/GMU_Kitchens/"
f = open(dataroot+"scales.txt", 'r')
scale = float(f.readlines()[scene_id-1])

idx1 = 612
idx2 = 622
frames = load_pose_struct(scene_id, dataroot)
fr_name1, R1, T1, k1_1, k1_2 = get_frame_info(idx1, frames)
fr_name2, R2, T2, k2_1, k2_2 = get_frame_info(idx2, frames)
intr = np.array((1.0477637710998533e+03, 1.0511749325842486e+03, 9.5926120509632392e+02, 5.2911546499433564e+02, scale), dtype=np.float32)

print fr_name1
print fr_name2

# load the images and depths
scene_path = dataroot + "gmu_scene_00" + str(scene_id) + "/"
img1 = cv2.imread(scene_path + "Images/" + fr_name1)
im_curr_dim = img1.shape
img2 = cv2.imread(scene_path + "Images/" + fr_name2)

depth_name1 = 'depth' + fr_name1[3:]
depth_name2 = 'depth' + fr_name2[3:]
depth_data1 = PIL.Image.open(scene_path + "Depths/" + depth_name1)
depth1 = np.array(depth_data1, dtype=np.float32)
depth_data2 = PIL.Image.open(scene_path + "Depths/" + depth_name2)   
depth2 = np.array(depth_data2, dtype=np.float32)

# case of resizing the images, need to rescale the intrinsics
im_dim = (640,480,3)
img1 = cv2.resize(img1, im_dim[:2]) # (960, 540)
img2 = cv2.resize(img2, im_dim[:2])
depth1 = cv2.resize(depth1, im_dim[:2])
depth2 = cv2.resize(depth2, im_dim[:2])
print img1.shape, depth1.shape
ratio_x = float(im_dim[0]) / float(im_curr_dim[1])
ratio_y = float(im_dim[1]) / float(im_curr_dim[0])
print ratio_x, ratio_y
intr[0] = intr[0]*ratio_x
intr[1] = intr[1]*ratio_y
intr[2] = intr[2]*ratio_x
intr[3] = intr[3]*ratio_y

depth1 = depth1.astype(np.int, copy=False)
depth2 = depth2.astype(np.int, copy=False)

#cv2.imshow("img1", img1/255.0) # imshow expects values from 0...1
#cv2.imshow("depth1", depth1)
#cv2.imshow('img2', img2/255.0)
#cv2.imshow('depth2', depth2)
#cv2.waitKey();	

flag=2

if flag==1:
	# dummy set of example points
	points = np.array([[592,215], [600,140], [465,145]], dtype=np.int)
	#points = np.array([[965,280], [500,500], [700,700], [825,180]], dtype=np.int)
	plot_points(img1, points, "im1_points")
	points3D = get3Dpoints(points, depth1, intr, R1, T1)
	#print points3D
	points2D = world2img(points3D, intr, R2, T2, k2_1, k2_2)
	print points2D
	plot_points(img2, points2D, "reprojected_points")
else:
	print np.unique(depth1)
	# replicate correspondence layer
	buffer = 20
	inds = np.where(depth1 > 0)
	#print inds
	im_size = depth1.shape
	# find the intersection of all coords that are inside the buffer
	inds_1 = np.where(inds[1]>=buffer)
	inds_2 = np.where(inds[0]>=buffer)
	inds_3 = np.where(inds[1]<im_size[1]-buffer)
	inds_4 = np.where(inds[0]<im_size[0]-buffer)
	idx = reduce(np.intersect1d, (inds_1, inds_2, inds_3, inds_4))
	#print idx
	coords = np.zeros((len(idx), 2), dtype=np.int)
	coords[:,0] = inds[1][idx] # inds[1] is x (width coordinate)
	coords[:,1] = inds[0][idx]
	coords = coords[0::10,:] # subsample coords
	print R1, T1
	print intr
	points3D = get3Dpoints(coords, depth1, intr, R1, T1)
	print points3D
	points2D_im2, valid_inds = world2img_layer(points3D, intr, R2, T2, im_size, buffer)
	points2D_im1 = coords[valid_inds,:]
	#plot_points(img1, points2D_im1[1::500], "points im1")
	#plot_points(img2, points2D_im2[1::500], "points im2")
	print "inds from depth", len(inds)
	print "points3D", points3D.shape
	print "points2D_im1", points2D_im1.shape
	print "points2D_im2", points2D_im2.shape
	print "valid inds", len(valid_inds)
	print "coords", coords.shape	
	

#inds = np.where(depth1 > 0)
#coords = np.zeros((len(inds[0]), 2), dtype=np.int)
#coords[:,0] = inds[1]
#coords[:,1] = inds[0]
#print coords
#plot_points(img1, coords[1::500], "non zero")
	
	
		
plt.show()		
	
	
	
	
	
	
	
	
	
	
	
	




