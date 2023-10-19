import gzip
import os
import numpy as np
from util.DataSet import DataSet, DataSets_novalid_norm

from attacker.FGSM_1 import FGSM_Mnist_train
from attacker.FGSM_1 import FGSM_Mnist_test

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images_and_labels(image_file, label_file, percentage=1.0):
    print('Extracting', image_file, label_file)
    with gzip.open(image_file) as image_bytestream, gzip.open(label_file) as label_bytestream:
        magic = _read32(image_bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in image file: %s' %
                (magic, image_file))
        magic = _read32(label_bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in label file: %s' %
                (magic, label_file))
        num_images = _read32(image_bytestream)
        rows = _read32(image_bytestream)
        cols = _read32(image_bytestream)
        num_labels = _read32(label_bytestream)
        if num_images != num_labels:
            raise ValueError(
                'Num images does not match num labels. Image file : %s; label file: %s' %
                (image_file, label_file))
        images = []
        labels = []
        labels_one = []
        # 图片的比例
        num_images = int(num_images * percentage)
        for _ in range(num_images):
            image_buf = image_bytestream.read(rows * cols)
            image = np.frombuffer(image_buf, dtype=np.uint8)
            image = np.multiply(image.astype(np.float32), 1.0 / 255.0)
            image[np.where(image == 0.0)[0]] = 1e-5
            image[np.where(image == 1.0)[0]] -= 1e-5
            label = np.frombuffer(label_bytestream.read(1), dtype=np.uint8)
            label_one = 1
            images.append(image)
            labels.append(label)
            labels_one.append(label_one)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32).squeeze()
        labels_one = np.array(labels_one, dtype=np.int32).squeeze()

        return images, labels, labels_one

def extract_images_and_labels_ten(image_file, label_file, percentage=1.0):
    print('Extracting', image_file, label_file)
    with gzip.open(image_file) as image_bytestream, gzip.open(label_file) as label_bytestream:
        magic = _read32(image_bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in image file: %s' %
                (magic, image_file))
        magic = _read32(label_bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in label file: %s' %
                (magic, label_file))
        num_images = _read32(image_bytestream)
        rows = _read32(image_bytestream)
        cols = _read32(image_bytestream)
        num_labels = _read32(label_bytestream)
        if num_images != num_labels:
            raise ValueError(
                'Num images does not match num labels. Image file : %s; label file: %s' %
                (image_file, label_file))
        images = []
        labels = []

        num_images = int(num_images * percentage)
        for _ in range(num_images):
            image_buf = image_bytestream.read(rows * cols)
            image = np.frombuffer(image_buf, dtype=np.uint8)
            image = np.multiply(image.astype(np.float32), 1.0 / 255.0)
            image[np.where(image == 0.0)[0]] = 1e-5
            image[np.where(image == 1.0)[0]] -= 1e-5
            label = np.frombuffer(label_bytestream.read(1), dtype=np.uint8)
            images.append(image)
            labels.append(label)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32).squeeze()

        return images, labels

def crop_augment(images, target_side_length=26):
    images = np.reshape(images, (-1, 28, 28))
    augmented_images_shape = list(images.shape)
    augmented_images_shape[0] *= 2
    augmented_images = np.zeros(shape=augmented_images_shape, dtype=np.float32) + 1e-5

    diff = (28 - target_side_length) // 2
    for i in range(len(images)):
        images_center = images[i][diff:-diff, diff:-diff]
        augmented_images[2*i] = images[i]
        choice = np.random.random()
        if choice < 0.25:
            augmented_images[2*i+1][:target_side_length, :target_side_length] = images_center
        elif choice < 0.5:
            augmented_images[2*i+1][:target_side_length, -target_side_length:] = images_center
        elif choice < 0.75:
            augmented_images[2*i+1][-target_side_length:, :target_side_length] = images_center
        else:
            augmented_images[2*i+1][-target_side_length:, -target_side_length:] = images_center

    augmented_images = np.reshape(augmented_images, (-1, 784))
    return augmented_images

def read_data_sets(dir, percentage = 1.0, training_type = "",testing_type = "", eps= 0):
    
    eps_values = eps

    train_image_file = "train-images-idx3-ubyte.gz"
    train_label_file = "train-labels-idx1-ubyte.gz"
    train_image_file = os.path.join(dir, train_image_file)
    train_label_file = os.path.join(dir, train_label_file)
    train_images_norm, train_labels_norm,train_labels_norm_one = extract_images_and_labels(train_image_file, train_label_file, percentage)

    test_image_file = "t10k-images-idx3-ubyte.gz"
    test_label_file = "t10k-labels-idx1-ubyte.gz"
    test_image_file = os.path.join(dir, test_image_file)
    test_label_file = os.path.join(dir, test_label_file)
    test_images_norm, test_labels_norm,test_labels_norm_one = extract_images_and_labels(test_image_file, test_label_file, percentage)

    '''Training'''
    if training_type == "FGSM":
        print("Training: This is the FGSM attack")
        Fgsm_train_num = len(train_images_norm)
        Fgsm_train_images,Fgsm_train_labels_norm,Fgsm_train_labels_zero = FGSM_Mnist_train(Fgsm_train_num, eps_values)

        train_images = np.concatenate((train_images_norm, Fgsm_train_images), axis=0)
        train_labels = np.concatenate((train_labels_norm, Fgsm_train_labels_norm), axis=0)
        train_labels_one = np.concatenate((train_labels_norm_one, Fgsm_train_labels_zero), axis=0)

    '''Testing'''
    if testing_type == "FGSM":
        print("Testing: This is the FGSM attack")
        Fgsm_test_num = len(test_images_norm)
        Fgsm_test_images, Fgsm_test_labels_norm, Fgsm_test_labels_zero = FGSM_Mnist_test(Fgsm_test_num, eps_values)

        test_images = np.concatenate((test_images_norm, Fgsm_test_images), axis=0)
        test_labels = np.concatenate((test_labels_norm, Fgsm_test_labels_norm), axis=0)
        test_labels_one = np.concatenate((test_labels_norm_one, Fgsm_test_labels_zero), axis=0)

    if training_type == "":
        train_norm = DataSet(train_images_norm,train_labels_norm)
        test_norm  = DataSet(test_images_norm,test_labels_norm)
    else:
        train_norm = DataSet(train_images, train_labels)
        test_norm  = DataSet(test_images, test_labels)

    return DataSets_novalid_norm(train_norm, test_norm)
