import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_gpu()
caffe.set_device(1)

import numpy
import glob
import tempfile
import os
from matplotlib import pyplot as plt
from array import array

import ROOT


output_directories = dict()
output_directories.update({'nu_data' : '/home/coradam/neutrino_supera/alex_net/'})
output_directories.update({'anu_data' : '/home/coradam/antineutrino_supera/alex_net/'})
output_directories.update({'anu_sim' : '/home/coradam/anu_sim_supera/alex_net/'})

input_directories = dict()
input_directories.update({'nu_data' : '/home/coradam/neutrino_supera/'})
input_directories.update({'anu_data' : '/home/coradam/antineutrino_supera/'})
input_directories.update({'anu_sim' : '/home/coradam/anu_sim_supera/'})

file_prefix = dict()
file_prefix.update({'nu_data' : 'R'})
file_prefix.update({'anu_data' : 'R'})
file_prefix.update({'anu_sim' : 'genie_anu_mc_'})




# We need the input network:
net_file = '/home/coradam/deeplearning/nova_net/nova_ana.prototxt'
weights = '/home/coradam/deeplearning/nova_net/nova_argoneut_iter_6000.caffemodel'

key='probt'


def writeCfgFile(inputFile):
  base_file = "/home/coradam/deeplearning/cfg/scale_downsample_data.cfg"
  out_file = "/home/coradam/deeplearning/cfg/process_nova.cfg"

  # Read in the base file:
  with open(base_file,"r") as _in:

    lines = _in.readlines()
    print lines[6]

    lines[6] = "  InputFiles:   [\"" + inputFile +  "\"] # list comma-separated files (if multiple)\n"

    with open(out_file, "w+") as _out:
      for line in lines:
        _out.write(line)

  return


def processFile(inputFile):

  # # Initialize the neural network:
  testNet = caffe.Net(net_file,weights, caffe.TEST)

  # We need an output root file for this:
  _out_name = inputFile.rstrip(".root") + "_nova_cnn.root"

  name = "image2d_tpc"
  ch = ROOT.TChain("%s_tree" % name)
  ch.AddFile(inputFile)
  n_entries = ch.GetEntries()

  br = None
  ch.GetEntry(0)
  exec('br = ch.%s_branch' % name)


  outfile = ROOT.TFile("NeutrinoModeNovaNet_6000.root","RECREATE")
  # outfile = ROOT.TFile(output_directories[data_type] + os.path.basename(_out_name),"RECREATE")
  outTree = ROOT.TTree('alex_net_cnn_tree', 'results tree' )


  run_ar = array( 'i', [ 0 ] )
  event_ar = array( 'i', [ 0 ] )
  nue_score_ar = array('f',[0.0])
  bkg_score_ar = array('f',[0.0])

  outTree.Branch( 'Run', run_ar, 'Run/I' )
  outTree.Branch( 'Event', event_ar, 'Event/I' )
  outTree.Branch( 'NueScore', nue_score_ar, 'NueScore/F' )
  outTree.Branch( 'BkgScore', bkg_score_ar, 'BkgScore/F' )

  _ten_percent = 100
  _run_print_interval = 1000

  for entry in xrange(n_entries):
    # print entry
    if entry % _ten_percent == 0:
        print "On Entry {} of {}".format(entry,n_entries)

    testNet.forward()
    ch.GetEntry(entry)
    nue_score = testNet.blobs[key].data[0][1]
    # print nue_score

    if nue_score > 0.7:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)

    # if entry % _run_print_interval == 0:
    #     print "Run {}, Event {}, nue score: {:.3}".format(br.run(), br.event(), nue_score)

    if br.run() == 620:
      if br.event() == 3756:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)

    if br.run() == 622:
      if abs( br.event() - 2738) < 5:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)
        
    if br.run() == 627:
      if br.event() == 895:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)
        
    if br.run() == 629:
      if br.event() == 25898 or br.event() == 32812:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)
    if br.run() == 634:
      if br.event() == 25212:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)
    if br.run() == 635:
      if br.event() == 25757:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)
    if br.run() == 650:
      if br.event() == 11366:
        print "Entry {}, Run {}, Event {}, nue score: {:.3}".format(entry, br.run(), br.event(), nue_score)


    if entry > 3000:
      break

    run_ar[0] = br.run()
    event_ar[0] = br.event()
    nue_score_ar[0] = testNet.blobs[key].data[0][1]
    bkg_score_ar[0] = testNet.blobs[key].data[0][0]
    outTree.Fill()


  outfile.Write()
  outfile.Close()

  return

  

def main():

  # types = ['nu_data','anu_sim','anu_data']
  # # types = ['anu_data']

  # for _type in types:

  #   # Get the list of files to use:
  #   files = glob.glob(input_directories[_type] + file_prefix[_type] + "*")

  #   print "Processing {} files for type {}".format(len(files),_type)

  files = ["/home/coradam/neutrino_supera/R622_E2001-E4000_supera.root"]

  for _file in files:
    print "Processing file {}".format(_file)
    writeCfgFile(_file)

    processFile(_file)

if __name__ == '__main__':
  main()