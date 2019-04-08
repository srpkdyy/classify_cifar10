import chainer
import chainer.links as L
import chainer.functions as F



class Block(chainer.Chain):
    # A convolution, batch norm, ReLU block
    def __init__(self, out_channels, ksize, pad=1):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize=ksize, pad=pad, nobias=True)
            self.bn = L.BatchNormalization(out_channels)


    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))



class VGG(chainer.Chain):
    def __init__(self, class_labels=10):
        super().__init__()
        with self.init_scope():
            self.block1_1 = Block(64, ksize=3)
            self.block1_2 = Block(64, ksize=3)
            self.block2_1 = Block(128, ksize=3)
            self.block2_2 = Block(128, ksize=3)
            self.block3_1 = Block(256, ksize=3)
            self.block3_2 = Block(256, ksize=3)
            self.block3_3 = Block(256, ksize=3)
            self.block4_1 = Block(512, ksize=3)
            self.block4_2 = Block(512, ksize=3)
            self.block4_3 = Block(512, ksize=3)
            self.block5_1 = Block(512, ksize=3)
            self.block5_2 = Block(512, ksize=3)
            self.block5_3 = Block(512, ksize=3)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)


    def __call__(self, x):
        h = F.dropout(self.block1_1(x), ratio=0.3)
        h = F.max_pooling_2d(self.block1_2(h), ksize=2)
        
        h = F.dropout(self.block2_1(h), ratio=0.4)
        h = F.max_pooling_2d(self.block2_2(h), ksize=2)

        h = F.dropout(self.block3_1(h), ratio=0.4)
        h = F.dropout(self.block3_2(h), ratio=0.4)
        h = F.max_pooling_2d(self.block3_3(h), ksize=2)

        h = F.dropout(self.block4_1(h), ratio=0.4)
        h = F.dropout(self.block4_2(h), ratio=0.4)
        h = F.max_pooling_2d(self.block4_3(h), ksize=2)

        h = F.dropout(self.block5_1(h), ratio=0.4)
        h = F.dropout(self.block5_2(h), ratio=0.4)
        h = F.max_pooling_2d(self.block5_3(h), ksize=2)

        h = F.dropout(h, ratio=0.5)
        h = F.dropout(F.relu(self.bn_fc1(self.fc1(h))), ratio=0.5)
        return self.fc2(h)
