import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_gpu()
caffe.set_device(1)

import numpy

import tempfile
import os
from matplotlib import pyplot as plt

# We need the input network:
net_file = '/home/coradam/deeplearning/alex_net_parallel/ana.prototxt'
weights = '/home/coradam/deeplearning/alex_net_parallel/alex_argoneut_iter_80000.caffemodel.h5'

key='probt'

testNet = caffe.Net(net_file,weights, caffe.TEST)

n_events = 1000
batch_size=10

print_steps = n_events / (10*batch_size)

output = numpy.zeros(n_events)

for i in xrange(n_events/batch_size):

    testNet.forward()
    # print testNet.blobs[key].data
    for j in xrange(batch_size):
      output[i*batch_size + j] = testNet.blobs[key].data[j][1]



    if i % print_steps == 0:
      print "On iteration {} of {}.".format(i*batch_size, n_events)

# print testNet.blobs['loss3/top-1'].data
# print testNet.blobs['loss3/top-5'].data
# print testNet.blobs['loss3/loss3'].data
    # labels  = net.blobs["label"].data
    # softmax = net.blobs["probt"].data
    # acc     = net.blobs['accuracy'].data

bins = numpy.arange(0,1.01,0.05)

# make a histogram of the output:
# 
score_hist, bin_edges = numpy.histogram(output,bins)

plot_bins = bin_edges[:-1]

# Figure out efficiency at certain cuts:

cut_vals = [0.5,0.7,0.8,0.9]

cut_eff = []

for val in cut_vals:
  tempbins = [0,val,1.01]
  data, junk = numpy.histogram(output,tempbins)
  cut_eff.append(1.0*data[1]/(data[0]+data[1]))


print cut_eff

plt.bar(plot_bins,score_hist,width=0.05,label=r"$\nu_e$ Classifier")

plt.grid(True)
plt.legend()
plt.show()
plt.savefig("/home/coradam/deeplearning/alex_net_parallel/figures/classifier_nue_scores.pdf")
plt.savefig("/home/coradam/deeplearning/alex_net_parallel/figures/classifier_nue_scores.png")