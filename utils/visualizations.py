
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_matches(img1, img2, matches, title, par):
	cmap = plt.cm.get_cmap("hsv", len(matches)) # get distinct colors for the lines
	img1 = img1[:, :, (2, 1, 0)]
	img2 = img2[:, :, (2, 1, 0)]
	plot_img = np.concatenate((img1, img2), axis=1)
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.imshow(plot_img)
	for i in range(len(matches)):
		x1,y1,x2,y2 = matches[i,0], matches[i,1], matches[i,2], matches[i,3]
		# plot the points
		plt.plot([x1], [y1], marker='*', markersize=15, color="orange")
		plt.plot([x2+img1.shape[1]], [y2], marker='*', markersize=15, color="orange")
		# plot the line
		x,y = [x1, x2+img1.shape[1]], [y1, y2]
		plt.plot(x,y, color=cmap(i))		
	ax.set_title(title)
	plt.axis('off')
	plt.tight_layout()
	plt.draw()
	plt.show
	#plt.savefig(par.root_dir + "examples/" + title + ".png")
	plt.savefig(par.save_dir + title + ".png")
	#plt.clf()
	
	
def plot_points(im, points, title): #, par):
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
	#plt.savefig(par.root_dir + "examples/" + title + ".png")
	
	
def plot_loss(loss, iter, step, model_path, loss_name):
	x = range(1,iter, step)
	#print "x ", x
	#print "loss ", loss[1:iter:step]
	plt.plot(x, loss[1:iter:step], 'b')
	plt.axis([0, iter, 0, max(loss[1:iter:step])])
	plt.ylabel(loss_name + ' Loss')
	plt.xlabel('Iter')
	#plt.show()
	plt.savefig(model_path+loss_name+"_"+str(iter)+"_loss.png")
	plt.clf()
	
	
def plot_dist(dist, img, title, par):
	# show img2 overlaid with the matching distance over the feature map
	fig = plt.figure(frameon=False)
	im1 = plt.imshow(img)
	im2 = plt.imshow(dist, alpha=0.3)
	plt.savefig(par.root_dir + "examples/" + title + ".png")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	