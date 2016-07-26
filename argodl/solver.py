import caffe
import os.path

def loadNet(prototxt,test=False):
  # Test if the file exists:
  if not os.path.isfile(prototxt):
      raise ValueError("The prototxt file you specified does not exist.")
  if test:
    net = caffe.Net(prototxt, caffe.TEST)
  else:
    net = caffe.Net(prototxt, caffe.TRAIN)
  return net


