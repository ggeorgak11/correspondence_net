from __future__ import division
import caffe
import numpy as np

def transplant(new_net, net, suffix):
    """
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.
    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.
    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat

			
def load_dil_weights(new_net, net):
	# Load the weights of the dilation convs
	# Usually its from conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
	dil_layers = [k for k in new_net.params.keys() if 'dil' in k]
	#print dil_layers
	for l in dil_layers:
		# find for l where to get the weights from
		b_layer = "conv" + l[3:6]
		new_net.params[l][0].data[...] = net.params[b_layer][0].data[...] # w
		new_net.params[l][1].data[...] = net.params[b_layer][1].data[...] # b	 
		print "copying", b_layer, " -> ", l 	
			
			
def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers, scale):
    """
    Set weights of each layer in layers to bilinear kernels for interpolation.
    """
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        #print filt
        net.params[l][0].data[range(m), range(k), :, :] = filt*scale
        print 'interpolated weights for', l, 'with scale', scale

def expand_score(new_net, new_layer, net, layer):
    """
    Transplant an old score layer's parameters, with k < k' classes, into a new
    score layer with k classes s.t. the first k' are the old classes.
    """
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data