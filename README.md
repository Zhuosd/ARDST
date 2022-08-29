# ARDST: An Adversarial-Resilient Deep Symbolic Tree for Adversarial Learning

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


## Abstract
Deep neural networks (DNNs) have driven much of the progress in numer-ous intelligent systems such as natural language processing and autonomous driving. However, trivial and human-unnoticeable adversarial attacks can fool a well-trained DNN to make arbitrarily wrong decisions. The crux of this issue lies in DNNs’ hierarchical layer-by-layer learning structure, where the tiny distortions tend to be escalated infinitely. Although many related de-fense methods are proposed, they all are the 'vaccine' type, i.e., they require prior knowledge of adversarial attacks to design a specific defense strategy. While in real attack scenarios, it is impossible to learn such prior knowledge in advance. To address this issue, this paper proposes a novel 'immune' type learning model from a neuro-symbolic perspective, termed Adversarial-Resilient Deep Symbolic Tree (ARDST). ARDST possesses two unique prop-erties: 1) it is a semi-parametric tree model, where the nodes are logic opera-tors and the weights of edges are the learned parameters, and 2) it can pro-vide a clear reasoning path of how a decision is made in very fine granulari-ty. As such, ARDST can not only defend the various adversarial attack types but also has a much smaller size of parameter space than DNNs. Extensive experiments on three benchmark datasets are carried out. The results sub-stantiate that our ARDST can achieve comparable representation learning ability to that of DNNs on perceptive tasks and, more importantly, is resili-ent to state-of-the-art adversarial attacks, including FGSM, DeepFool, PGD, and BIM.

## File

The overall framework of this project is designed as follows
1. The **attacker** file is to store the relevant attack model and files

2. The **defense** file is to store the defense model parameters corresponding to the model

3. The **dataset** file is used to hold the dataset

4. The **Struct** file is to store the files required for DST Structure

5. The **util** is the storage of the relevant model adjustment process algorithm

### Getting Started
1. Clone this repository

```
git clone https://github.com/ARDST/ARDST_Algo.git
```

2. Make sure you meet package requirements by running:

```
pip install -r requirements.txt
```

3. Running ARDST model

```
python ARDST_main.py
```

### Example

Here we will show how to train a provably Deep Symbolic Tree defense model. We will use a DST defense FGSM as an example

### Operating

This is used to train the classification of normal examples
```
python ARDST_main.py
```

