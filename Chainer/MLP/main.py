import numpy as np
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators, optimizers
from chainer import training
from chainer.training import extensions




class MLP(chainer.Chain):

    def __init__(self, n_nodes=4096, n_out=10):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(None, n_nodes)
            self.fc2 = L.Linear(None, n_nodes)
            self.fc3 = L.Linear(None, n_nodes)
            self.fc4 = L.Linear(None, n_nodes)
            self.fc5 = L.Linear(None, n_out)


    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.dropout(F.relu(self.fc2(h)), ratio=0.4)
        h = F.dropout(F.relu(self.fc3(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.6)
        return self.fc5(h)




def main():
    parser = argparse.ArgumentParser(description='MLP for MNIST')
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--nodes', '-n', type=int, default=1000, help='')
    parser.add_argument('--out', '-o', default='result')
    args = parser.parse_args()

    model = L.Classifier(MLP(args.nodes))
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_cifar10(ndim=1)

    train_itr = iterators.SerialIterator(train, args.batch_size)
    test_itr = iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(train_itr, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_itr, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()




if __name__ == '__main__':
    main()

