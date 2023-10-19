import os
import time
import argparse
import warnings
import torch
import torch.nn as nn
from torchvision import models
warnings.filterwarnings("ignore")
from util.structure.Vtree import Vtree as Dst
from defense.Mnist_mult import read_data_sets

from util.algo.LogisticCircuit import LogisticCircuit as DST

FLAGS = None
model = models.alexnet(pretrained=True)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
        self.l1 = nn.Sequential(*list(model.children())[:-1])#.to('cuda:0')
    def forward(self,x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.reshape(-1,shape)

def main():
    if FLAGS.dataset == "Mnist":
        print("This dataset is Mnist")
        data = read_data_sets(FLAGS.data_path, FLAGS.percentage, FLAGS.training_type, FLAGS.testing_type, FLAGS.attack_eps)

    dst = Dst.read(FLAGS.dst)

    if FLAGS.circuit != "":
        with open(FLAGS.circuit, "r") as circuit_file:
            circuit = DST(dst, FLAGS.num_classes, circuit_file=circuit_file)
            print("The saved circuit is successfully loaded.")
            data.train.features = circuit.calculate_features(data.train.images)
    else:
        circuit = DST(dst, FLAGS.num_classes)
        data.train.features = circuit.calculate_features(data.train.images)
        circuit.learn_parameters(data.train, 50)

    print(f"The starting circuit has {circuit.num_parameters} parameters.")
    data.test.features = circuit.calculate_features(data.test.images)
    print(
        f"Its performance is as follows. "
        f"Training accuracy: {circuit.calculate_accuracy(data.train):.5f}\t"
        f"Test accuracy: {circuit.calculate_accuracy(data.test):.5f}"
    )

    print("Start structure learning.")

    train_accuracy = circuit.calculate_accuracy(data.train)

    best_accuracy = train_accuracy
    for i in range(FLAGS.num_structure_learning_iterations):
        cur_time = time.time()

        circuit.change_structure(data.train, FLAGS.depth, FLAGS.num_splits)

        data.train.features = circuit.calculate_features(data.train.images)
        data.test.features = circuit.calculate_features(data.test.images)

        circuit.learn_parameters(data.train, FLAGS.num_parameter_learning_iterations)

        train_accuracy = circuit.calculate_accuracy(data.train)
        print(
            f"Training accuracy: {circuit.calculate_accuracy(data.train):.5f}\t"
            f"Test accuracy: {circuit.calculate_accuracy(data.test):.5f}"
        )
        print(f"Num parameters: {circuit.num_parameters}\tTime spent: {(time.time() - cur_time):.2f}")

        if FLAGS.save_path != "" and (train_accuracy > best_accuracy):
            best_accuracy = train_accuracy
            print("Obtained a logistic circuit with higher classification accuracy. Start saving.")
            with open(FLAGS.save_path, "w") as circuit_file:
                circuit.save(circuit_file)
            print("Logistic circuit saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type = str, 
                        default = "Mnist", 
                        choices = ["Mnist", "FashionMnist", "TrafficSigns"], 
                        help = "There are four types of data sets in total" )

    parser.add_argument("--training_type", type=str, 
                        default = "FGSM", 
                        choices = ["FGSM", "DeepFool", "BIM", "PGD"], 
                        help="There are baseline to learn adversarial examples")

    parser.add_argument("--testing_type", type = str, 
                        default = "FGSM", 
                        choices = ["FGSM", "DeepFool", "BIM", "PGD"], 
                        help = "There are four types of attacks in total")

    parser.add_argument("--attack_eps", type = float, 
                        default = 0.01, 
                        choices = [0.01, 0.1, 0.3, 0.5], 
                        help = "There are about attack model eps values")
    
    parser.add_argument("--data_path", type = str,  
                        default = "dataset/Mnist/MNIST/raw/", 
                        choices = ["./dataset/Mnist/MNIST/raw/", "./dataset/FashionMnist/", "./dataset/TrafficSigns"], 
                        help = "Directory for the stored input data.")

    parser.add_argument("--num_classes", type = int, 
                        default = 10, 
                        choices = [10, 10, 43], 
                        help = "Number of classes in the classification task.")

    parser.add_argument("--dst", type = str,
                        default = "Struct/DLTreeStr_784.vtree", 
                        help = "Path for dst.")

    parser.add_argument("--circuit", type = str,
                        default = "",
                        help = "[Optional] File path for the saved logistic circuit to load. "
                        "Note this circuit has to be based on the same dst as provided in --dst.", )

    parser.add_argument("--num_structure_learning_iterations", type = int,
                        default = 200,
                        help = "[Optional] Num of iterations for structure learning. Its default value is 5000.",)

    parser.add_argument("--num_parameter_learning_iterations", type = int,
                        default = 15,
                        help = "[Optional] Number of iterations for parameter learning after the structure is changed."
                        "Its default value is 15.",)

    parser.add_argument("--depth", type = int, 
                        default = 2, 
                        help = "[Optional] The depth of every split. Its default value is 2.")

    parser.add_argument("--num_splits", type = int,
                        default = 5,
                        help = "[Optional] The number of splits in one iteration of structure learning." "It default value is 3.", )

    parser.add_argument("--percentage", type = float,
                        default = 1,
                        help="[Optional] The percentage of the training dataset that will be used. " "Its default value is 100%%. 1.0",)

    parser.add_argument("--save_path", type = str, 
                        default = "", 
                        help = "[Optional] File path to save the best-performing circuit.")

    FLAGS = parser.parse_args()
    if FLAGS.num_classes == 2:
        FLAGS.num_classes = 1
        message = (
            "It is essentially a binary classification task when num_classes is set to 2, "
            + "and hence we automatically modify it to be 1 to be better compatible with sklearn."
        )
        warnings.warn(message, stacklevel=2)
    main()
