import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_gpu()
caffe.set_device(0)

import numpy

import tempfile
import os
from matplotlib import pyplot as plt

# We need the input network:
net_file = '/home/coradam/deeplearning/nova_net/nova_ana.prototxt'
weights = '/home/coradam/deeplearning/nova_net/nova_argoneut_iter_135000.caffemodel'

key='probt'

testNet = caffe.Net(net_file,weights, caffe.TEST)

n_events = 100
batch_size=10

print_steps = n_events / (10*batch_size)

output = numpy.zeros(n_events)

for i in xrange(n_events/batch_size):

    testNet.forward()
    print testNet.blobs[key].data
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

bins = numpy.arange(0,1.01,0.025)

# make a histogram of the output:
# 
score_hist, bin_edges = numpy.histogram(output,bins)

plot_bins = bin_edges[:-1]

# Figure out efficiency at certain cuts:

cut_vals = [0.5,0.7,0.9]
colors = ['r','g','black']
cut_eff = []

for val in cut_vals:
  tempbins = [0,val,1.01]
  data, junk = numpy.histogram(output,tempbins)
  cut_eff.append(1.0*data[1]/(data[0]+data[1]))


print cut_eff

fig,ax = plt.subplots(figsize=(12,9))

plt.bar(plot_bins,score_hist,width=0.025,label=r"$\nu_e$ CC")

for val, eff,col in zip(cut_vals,cut_eff,colors):
  plt.axvline(val,ls = '--', linewidth=5, color = col, label="{}% efficiency".format(int(100*eff)))

plt.title("Nova Network",fontsize=30)

# Make the ticks bigger
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)


plt.xlabel("Classifier Score",fontsize=30)

plt.grid(True)
plt.legend(loc=2,fontsize=25)
plt.savefig("/home/coradam/deeplearning/nova_net/figures/classifier_nue_scores.pdf",format='pdf')
plt.savefig("/home/coradam/deeplearning/nova_net/figures/classifier_nue_scores.png",format='png')
plt.show()
