import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import iterators, optimizers
from chainer import training
from chainer.training import extensions

from model import VGG



def main():
    # 7 iters/sec
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100, help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result')
    args = parser.parse_args()

    train_ds, test_ds = chainer.datasets.get_cifar10()
    train_iter = iterators.SerialIterator(train_ds, args.batch_size)
    test_iter = iterators.SerialIterator(test_ds, args.batch_size, repeat=False, shuffle=False)

    model = L.Classifier(VGG(class_labels=10))
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger=(args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.ExponentialShift('alpha', 0.5), trigger=(30, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    


if __name__ == '__main__':
    main()
