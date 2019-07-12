# correspondence_net

Python code for learning dense correspondences between frames. The code is using the [master](https://github.com/BVLC/caffe) branch of Caffe.

We provide a correspondence_layer which uses the camera poses of the frames in order to sample training pairs and pass to a contrastive loss. During backpropagation, the gradients are passed accordingly to the appropriate locations in the feature maps.

We provide input layers for training on the following datasets:
1) GMU-Kitchens [Link](https://cs.gmu.edu/~robot/gmu-kitchens.html)
2) BigBIRD [Link](http://rll.berkeley.edu/bigbird/)
3) WRGB-D [Link](https://rgbd-dataset.cs.washington.edu/)

The main python file is the correspondenceNetwork.py. Please see correspondenceParams.py for more details on setting the appropriate paths and data dependencies. 
