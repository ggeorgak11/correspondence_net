
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
		

this_dir = osp.dirname(__file__)

# Add root to PYTHONPATH
lib_path = osp.join(this_dir)
add_path(lib_path)
	
# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, 'caffe-master', 'python')
add_path(caffe_path)

# Add utils to PYTHONPATH
utils_path = osp.join(this_dir, 'utils')
add_path(utils_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

# Add input layers to path
input_layer_path = osp.join(this_dir, 'lib', 'input_layers')
add_path(input_layer_path)

# Add correspondence layer to path
correspondence_layer_path = osp.join(this_dir, 'lib', 'correspondence_layer')
add_path(correspondence_layer_path)


