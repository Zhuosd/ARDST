# ARDST: An Adversarial-Resilient Deep Symbolic Tree for Adversarial Learning

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


## Abstract
The advancement of intelligent systems, particularly in domains like natural language processing and autonomous driving, has been primarily driven by deep neural networks (DNNs).
  However, these systems exhibit vulnerability to adversarial attacks that can be both subtle and imperceptible to humans, resulting in arbitrary and erroneous decisions.
  This susceptibility arises from the hierarchical layer-by-layer learning structure of DNNs, where small distortions can be exponentially amplified.
  While several defense methods have been proposed, they often necessitate prior knowledge of adversarial attacks to design specific defense strategies.
  This requirement is often unfeasible in real-world attack scenarios. In this paper, we introduce a novel learning model, termed "immune" learning, known as \emph{Adversarial-Resilient Deep Symbolic Tree} (ARDST), from a neuro-symbolic perspective.
  The ARDST model is semi-parametric and takes the form of a tree, with logic operators serving as nodes and learned parameters as weights of edges.
  This model provides a transparent reasoning path for decision-making, offering fine granularity, and has the capacity to withstand various types of adversarial attacks, all while maintaining a significantly smaller parameter space compared to DNNs.
  Our extensive experiments, conducted on three benchmark datasets, reveal that ARDST exhibits a representation learning capability similar to DNNs in perceptual tasks and demonstrates resilience against state-of-the-art adversarial attacks.
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

