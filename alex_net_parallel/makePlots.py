import numpy
from matplotlib import pyplot as plt
import glob
from scipy import optimize

# First, know the directory where the output is going:
outdir = "/home/coradam/deeplearning/alex_net/figures/"
sourcedir = "/home/coradam/deeplearning/alex_net/"

plot_prefix = "alexnet_argoneut_"

n_training_events = 500000 / 60.
n_testing_events = 200000 / 100.

#this is the prefix to the files:
fileprefix = "alex_argoneut_savestate_"

def readFiles():

  # This function finds all of the available files, reads them in, and combines their data
  # It returns the loss, accuracy, and testing Accuracy, and iterations where testing happened
  
  files = glob.glob(sourcedir+fileprefix+"*.npz")

  iterations = []  

  for f in files:
    # Strip off the front of the file, and the extension, to get the iteration:
    iterations.append(int(f.strip(sourcedir+fileprefix).rstrip('.npz')))

  # This sorts the two lists together:
  temp = zip(iterations,files)
  temp.sort()

  loss_list = []
  acc_list = []
  testacc_list = []


  n_points = 0

  for iteration, filename in temp:
    dat = numpy.load(filename)
    # If the iteration is less than 10000, switch the loss and accuracy:
    if iteration < 10000:
      continue
      loss_list.append(dat['accuracy'])
      acc_list.append(dat['loss'])
    else:
      loss_list.append(dat['loss'])
      acc_list.append(dat['accuracy'])
    testacc_list.append(dat['testAccuracy'])
    n_points += len(dat['loss'])
  
  # Package the various things into one 
  loss = numpy.asarray(loss_list).flatten()
  acc = numpy.asarray(acc_list).flatten()
    
  # The test accuracy has to be combined into 

  testacc = numpy.zeros(len(testacc_list))
  testacc_it = numpy.zeros(len(testacc_list))
  for i in xrange(len(testacc_list)):
    testacc[i] = numpy.mean(testacc_list[i])
    testacc_it[i] = temp[i][0]

  return acc, loss, testacc, testacc_it

def plotLoss(loss):

  # Calculate the number of epochs this has been (using the length )

  n_iterations = len(loss)

  epochs = numpy.arange(0,n_iterations) / (1.0*n_training_events)
  
  fig, ax = plt.subplots(figsize=(12,8))

  plt.semilogy(epochs,loss,label="Loss")
  plt.xlabel("Epochs",fontsize=25)
  plt.ylabel("Loss",fontsize=25)

  # Make the ticks bigger
  for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(16)
  for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(16)


  plt.title("Loss of Alex Net",fontsize=30)
  plt.legend(fontsize=25)
  plt.grid(True)

  plt.savefig(outdir + plot_prefix + "loss.png")
  plt.savefig(outdir + plot_prefix + "loss.pdf")

  plt.show()

def plotAccuracy(trainingAccuracy,testAccuracy, testacc_it):

  n_iterations = len(trainingAccuracy)


  epochs = numpy.arange(0,n_iterations) / (1.0*n_training_events)


  test_epochs = testacc_it / n_training_events

  # Fit the last 25 points from the training with a line
  


  fig, ax = plt.subplots(figsize=(12,8))

  plt.plot(epochs, trainingAccuracy,label="Test Accuracy")
  plt.plot(test_epochs, testAccuracy,label="Train Accuracy",marker="o",color='r',linewidth=4)

  plt.plot(test_epochs, [1]*len(test_epochs))

  plt.xlabel("Epochs",fontsize=25)
  plt.ylabel("Accuracy",fontsize=25)


  # Make the ticks bigger
  for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(16)
  for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(16)

  plt.title("Accuracy of Alex Net")
  ax.set_ylim([0,1.4])
  plt.legend(fontsize=25)


  plt.grid(True)
  plt.show()

def main():

  accuracy, loss, testaccuracy, testacc_it = readFiles()

  # # This is temporary:
  # temp = loss
  # loss = accuracy
  # accuracy = temp


  plotLoss(loss)
  plotAccuracy(accuracy, testaccuracy, testacc_it)


if __name__ == '__main__':
  main()