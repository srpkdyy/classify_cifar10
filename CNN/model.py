import chainer
import chainer.links as L
import chainer.functions as F




class CNN(chainer.Chain):
    def __init__(self, n_nodes=4096, n_out=10):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, ksize=5, pad=2)
            self.conv2 = L.Convolution2D(None, 192, ksize=4, pad=2)
            self.conv3 = L.Convolution2D(None, 288, ksize=3, pad=1)
            self.conv4 = L.Convolution2D(None, 384, ksize=3, pad=1)
            self.conv5 = L.Convolution2D(None, 480, ksize=3, pad=1)
            self.bn1 = L.BatchNormalization(96)
            self.bn2 = L.BatchNormalization(192)
            self.bn3 = L.BatchNormalization(288)
            self.bn4 = L.BatchNormalization(384)
            self.bn5 = L.BatchNormalization(480)
            self.fc1 = L.Linear(None, n_nodes)
            self.fc2 = L.Linear(None, n_nodes)
            self.fc3 = L.Linear(None, n_out)

    
    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), ksize=5, stride=4)
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), ksize=4, stride=3)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.max_pooling_2d(F.relu(self.bn5(self.conv5(h))), ksize=3, stride=2)
        h = F.dropout(F.relu(self.fc1(h)))
        h = F.dropout(F.relu(self.fc2(h)))
        return self.fc3(h)
