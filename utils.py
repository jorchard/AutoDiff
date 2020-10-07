# utils

import numpy as np
import matplotlib.pyplot as plt

# Abstract class
class MyDataset():
    def __init__(self):
        return

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError


# MyDataLoader
class MyDataLoader():
    def __init__(self, ds, batchsize=1, shuffle=False):
        '''
         dl = MyDataLoader(ds, batchsize=1, shuffle=False)

         Creates an iterable dataloader object that can be used to feed batches into a neural network.

         Inputs:
           ds         a MyDataset object
           batchsize  size of the batches
           shuffle    randomize the ordering

         Then,
           next(dl) returns the next batch
                    Each batch is a tuple containing (inputs, targets), where:
                     - inputs is a 2D numpy array containing one input per row, and
                     - targets is a 2D numpy array with a target on each row
        '''
        self.index = 0
        self.ds = ds
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.n_batches = (len(ds)-1)//self.batchsize + 1   # might include a non-full last batch

        self.MakeBatches()

    def Reset(self):
        self.index = 0
        if self.shuffle:
            self.MakeBatches()

    def MakeBatches(self):
        if self.shuffle:
            self.order = np.randperm(N)
        else:
            self.order = list(range(len(self.ds)))

        self.batches = []
        for batchnum in range(self.n_batches):
            low = batchnum*self.batchsize
            high = min((batchnum+1)*self.batchsize, len(self.ds))

            idxs = self.order[low:high]
            inputs = []
            targets = []
            for k in idxs:
                samp = self.ds.__getitem__(k)
                inputs.append(samp[0])
                targets.append(samp[1])
            self.batches.append([np.vstack(inputs), np.vstack(targets)])


    def __next__(self):
        '''
         Outputs:
           dl         a list of batches
        '''
        if self.index < self.n_batches:
            result = self.batches[self.index]
            self.index += 1
            return result
        raise StopIteration

    def __iter__(self):
        return self




# SimpleDataset: creates a simple classification dataset,
# mapping the row-vectors in A to the row-vectors in B.

class SimpleDataset(MyDataset):
    '''
     SimpleDataset
    '''
    def __init__(self, A, B, n=300, noise=0.1):
        self.samples = []
        self.n_classes = len(A)
        self.input_dim = len(A[0])
        for i in range(n):
            r = np.random.randint(self.n_classes)
            sample = [A[r]+noise*np.random.randn(*(A[r].shape)), B[r]]
            self.samples.append(sample)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

    def Inputs(self):
        x = []
        for s in self.samples:
            x.append(s[0])
        return np.stack(x)

    def Targets(self):
        t = []
        for s in self.samples:
            t.append(s[1])
        return np.stack(t)

    def InputsOfClass(self, c):
        x = []
        for s in self.samples:
            if torch.argmax(s[1])==c:
                x.append(s[0])
        return np.stack(x)

    def ClassMean(self):
        xmean = []
        for c_idx in range(self.n_classes):
            classmean = np.mean(self.InputsOfClass(c_idx), axis=0)
            xmean.append(classmean)
        return np.stack(xmean)

    def Plot(self, labels=[], idx=(0,1), equal=True):
        X = self.Inputs()
        if len(labels)==0:
            labels = self.Targets()
        colour_options = ['y', 'r', 'g', 'b', 'k']
        cidx = np.argmax(labels, axis=1)
        colours = [colour_options[k] for k in cidx]
        plt.scatter(X[:,idx[0]], X[:,idx[1]], color=colours, marker='.')

        if equal:
            plt.axis('equal');
