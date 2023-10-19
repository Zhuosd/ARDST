import numpy as np
class DataSet(object):

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._one_hot_labels = to_one_hot_encoding(labels)
        self._features = None
        self._num_samples = self._images.shape[0]
        self._num_epochs = 0
        self._index = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def one_hot_labels(self):
        return self._one_hot_labels

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_epochs(self):
        return self._num_epochs

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples, features and labels from this data set."""
        assert batch_size <= self._num_samples

        if self._index + batch_size >= self._num_samples:
            perm = np.arange(self._num_samples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._features = self._features[perm]
            self._index = 0
            self._num_epochs += 1

        images = self._images[self._index: self._index + batch_size]
        labels = self._labels[self._index: self._index + batch_size]
        features = self._features[self._index: self._index + batch_size]
        self._index += batch_size
        return images, features, labels, to_one_hot_encoding(labels)

class DataSets(object):

    def __init__(self, train, valid, test):
        self._train = train
        self._test = test
        self._valid = valid

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def test(self):
        return self._test

class DataSets_train_valid_test(object):

    def __init__(self, train_norm,valid, test_norm):
        self._train_norm = train_norm
        self._valid = valid
        self._test_norm = test_norm

    @property
    def valid(self):
        return self._valid

    @property
    def train_norm(self):
        return self._train_norm

    @property
    def test_norm(self):
        return self._test_norm

# 定义返回RGB三个不同的通道
class DataSets_RGB(object):

    def __init__(self, train_r, valid_r, test_r, train_g, valib_g, test_g, train_b, valid_b, test_b):
        self._train_r = train_r
        self._test_r  = test_r
        self._valid_r = valid_r
        self._train_g = train_g
        self._valid_g = valib_g
        self._test_g  = test_g
        self._train_b = train_b
        self._valid_b = valid_b
        self._test_b  = test_b

    @property
    def train_r(self):
        return self._train_r

    @property
    def valid_r(self):
        return self._valid_r

    @property
    def test_r(self):
        return self._test_r

    @property
    def train_g(self):
        return self._train_g

    @property
    def valib_g(self):
        return self._valid_g

    @property
    def test_g(self):
        return self._test_g

    @property
    def train_b(self):
        return self._train_b

    @property
    def valib_g(self):
        return self._valid_b

    @property
    def test_b(self):
        return self._test_b

class DataSets_novalid(object):

    def __init__(self, train_adv, test_adv, train_norm, test_norm):
        self._train_adv = train_adv
        self._test_adv = test_adv
        self._train_norm = train_norm
        self._test_norm  = test_norm

    @property
    def train_adv(self):
        return self._train_adv

    @property
    def test_adv(self):
        return self._test_adv

    @property
    def train_norm(self):
        return self._train_norm

    @property
    def test_norm(self):
        return self._test_norm

class DataSets_novalid_norm(object):

    def __init__(self, train, test):
        self._train = train
        self._test  = test
    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test

def to_one_hot_encoding(labels):
    num_classes = np.max(labels) + 1
    one_hot_labels = np.zeros(shape=(len(labels), num_classes), dtype=np.float32)
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1.0
    return one_hot_labels
