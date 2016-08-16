import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_gpu()
caffe.set_device(1)

import numpy

import alex_net.solver
import tempfile
import os

class solveControlParams(object):
    def __init__(self):
        self.max_iterations = 50000
        self.n_iteration_per_block = 1000
        self.n_tests = 200
        # Index starts from 0
        self.start_iteration = 20000

        self.training_prototxt = 'nova_net/nova_train_val.prototxt'
        self.testing_prototxt = 'nova_net/nova_test_val.prototxt'


        # This outlines the parameters of the solver:
        # The number of iterations over which to average the gradient.
        # Effectively boosts the training batch size by the given factor, without
        # affecting memory utilization.
        self.iter_size =1
        self.max_iter = 1000000     # # of times to update the net (training iterations)
        
        # Solve using the stochastic gradient descent (SGD) algorithm.
        # Other choices include 'Adam' and 'RMSProp'.
        self.type = 'SGD'

        # Set the initial learning rate for SGD.
        self.base_lr = 0.0005

        # Set `lr_policy` to define how the learning rate changes during training.
        # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
        # every `stepsize` iterations.
        self.lr_policy = 'step'
        self.gamma = 0.1
        self.stepsize = 20000

        # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
        # weighted average of the current gradient and previous gradients to make
        # learning more stable. L2 weight decay regularizes learning, to help prevent
        # the model from overfitting.
        self.momentum = 0.9
        self.weight_decay = 5e-4

        # Display the current training loss and accuracy every 1000 iterations.
        self.display = 1

        # Snapshots are files used to store networks we've trained.  Here, we'll
        # snapshot every 10K iterations -- ten times during training.
        self.snapshot = 2000
        self.snapshot_prefix = '/home/coradam/deeplearning/nova_net/nova_argoneut'
        # self.snapshot_format = caffe_pb2.SolverParameter.HDF5
        
        # Train on the GPU.  Using the CPU to train large networks is very slow.
        self.solver_mode = caffe_pb2.SolverParameter.GPU
    
    def getStartSnapshot(self):
        snapshot = self.snapshot_prefix
        it = self.start_iteration
        snapshot += "_iter_" + str(it) + ".solverstate"
        if os.path.isfile(snapshot):
            return snapshot
        else:
            return None


    def solver(self, train_net_path, test_net_path=None):
        s = caffe_pb2.SolverParameter()


        # Specify locations of the train and (maybe) test networks.
        s.train_net = train_net_path
        if test_net_path is not None:
            s.test_net.append(test_net_path)
            s.test_interval = 1000  # Test after every 1000 training iterations.
            s.test_iter.append(100) # Test on 100 batches each time we test.


        s.iter_size = self.iter_size
        s.max_iter = self.max_iter
        s.type = self.type
        s.base_lr = self.base_lr
        s.lr_policy = self.lr_policy
        s.gamma = self.gamma
        s.stepsize = self.stepsize
        s.momentum = self.momentum
        s.weight_decay = self.weight_decay
        s.display = self.display
        s.snapshot = self.snapshot
        s.snapshot_prefix = self.snapshot_prefix
        s.solver_mode = self.solver_mode
        # s.snapshot_format = self.snapshot_format

        # Write the solver to a temporary file and return its filename.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(str(s))
            return f.name

    def getSaveFile(self,iteration):
        savename = self.snapshot_prefix
        savename += "_savestate_" + str(iteration)
        return savename

    def saveTrainingData(self,iteration, trainingAccuracy,trainingLoss, testingAccuracy=None):
        fname = self.getSaveFile(iteration)
        if testingAccuracy is not None:
            numpy.savez(fname,accuracy=trainingAccuracy,loss=trainingLoss,testAccuracy=testingAccuracy)
        else:
            numpy.savez(fname,accuracy=trainingAccuracy,loss=trainingLoss)
        pass


def run_solver(niter, solver, name):
    """Run solver for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solver` is a list of (name, solver) tuples."""
    blobs = ('loss', 'accuracy')
    loss = numpy.zeros(niter)
    acc = numpy.zeros(niter)

    for it in range(niter):
        solver.step(1)  # run a single SGD step in Caffe
        loss[it], acc[it] = (solver.net.blobs[b].data.copy() for b in blobs)

    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    filename = 'weights.%s.caffemodel' % name
    weights[name] = os.path.join(weight_dir, filename)
    solver.net.save(weights[name])
    return loss, acc, weights



params = solveControlParams()

solverparams = params.solver(params.training_prototxt)
print solverparams

print params.getStartSnapshot()

solver = caffe.SGDSolver(solverparams)


if params.getStartSnapshot() is not None:
    print "Starting training from iteration {}".format(params.start_iteration)
    solver.restore(params.getStartSnapshot())
else:
    print "Starting training from iteration 0."
#   solver = caffe.
#   pass


# result = run_solver(5,solver, 'alex', 1)

# Loop for some number of iterations, and break it into blocks.
# At the end of each block, save the weights to a persistent space, save the 
# loss and accuracy to a persistent space for that training segment,
# and compute the accuracy for a set of training data.

print "Begin training"

n_blocks = params.max_iterations / params.n_iteration_per_block


for block in xrange(n_blocks):


    loss, acc, weights = run_solver(params.n_iteration_per_block, solver,'nova')
    print "Finished block {}, last loss: {}; last acc: {}".format(block, loss[-1],acc[-1])



    # # # At the end of the block, run a testing network:
    # testNet = caffe.Net(params.testing_prototxt,weights['nova'], caffe.TEST)
    # test_accuracy = numpy.zeros(params.n_tests)
    # for i in xrange(params.n_tests):
    #     test_accuracy[i] = testNet.forward()['accuracy']

    iteration = (block+1)*params.n_iteration_per_block + params.start_iteration


    # print "Accuracy after {} iterations: {} +\- {} ".format(iteration, 
    #                                                       numpy.mean(test_accuracy),
    #                                                       numpy.std(test_accuracy))



    # At this point, we have the accuracy and loss from training, and the accuracy from testing
    # Save it to a state file (which the params class can do)
    params.saveTrainingData(iteration, acc, loss)
