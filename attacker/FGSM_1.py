import torch
import numpy as np
import torchattacks
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import attacker.Model.MNIST_Model as MNIST_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''This is the datasets'''
Mnist_train_data = datasets.MNIST(root = 'dataset/Mnist/',
                                  train = True,
                                  download = True,
                                  transform = transforms.ToTensor())

Mnist_test_data = datasets.MNIST(root = 'dataset/Mnist/',
                                 train = False,
                                 download = True,
                                 transform = transforms.ToTensor())

Mnist_train_loader = torch.utils.data.DataLoader(Mnist_train_data,batch_size=1)
Mnist_test_loader = torch.utils.data.DataLoader(Mnist_test_data,batch_size=1)

'''Load the Model'''
Mnist_target_model = MNIST_Model.MNIST_Model()#.cuda()
Mnist_target_model.load_state_dict(torch.load("attacker/pt_file/mnist.pt",map_location='cpu'))
Mnist_target_model.eval()

def FGSM_Mnist_train(numbers, eps_values):
    image = []
    label = []
    label_one = []
    number = numbers
    atk = torchattacks.FGSM(Mnist_target_model, eps = eps_values)
    for i,(images,labels) in enumerate(Mnist_train_loader):
        if i == number:
            break
        adv_images = atk(images, labels)
        img = adv_images.cpu().numpy()
        img = img.reshape(784)

        image.append(img)
        label.append(labels)
        label_one.append(0)

    image = np.array(image)
    label = label
    label_one = label_one
    return image, label, label_one

def FGSM_Mnist_test(numbers, eps_values):
    image = []
    label = []
    label_one = []
    number = numbers
    atk = torchattacks.FGSM(Mnist_target_model, eps = eps_values)
    for i,(images,labels) in enumerate(Mnist_test_loader):
        if i == number:
            break
        adv_images = atk(images, labels)
        img = adv_images.cpu().numpy()
        img = img.reshape(784)

        image.append(img)
        label.append(labels)
        label_one.append(0)

    image = np.array(image)
    label = label
    label_one = label_one
    return image, label, label_one

