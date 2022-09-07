11-767 Lab 1: Feed-forward neural networks for language and vision in PyTorch.
===
In this lab you will build a basic feed-forward neural network for classification in PyTorch, and train and evaluate that model in terms of efficiency and classification accuracy on simple
language and vision tasks. The goals of this exercise are: 
 1. Proficiency training and evaluating basic feed-forward neural network architectures for language and computer vision classifcation tasks in PyTorch; 
 2. Implement basic efficiency benchmarks: Latency, parameter count, and FLOPs.
 3. Experiment with varying model size, depth, and input resolution and analyze how that impacts efficiency vs. accuracy.

Data
----

### Language 
For a language task we will use the [Stanford Sentiment Treebank (SST-2)](https://huggingface.co/datasets/sst2) dataset for sentiment classification. 
Examples in this dataset consist of tokenized English text labeled with binary categories indicating the binary sentiment (positive/negative) of the sentence. 
The data files we will use for this class are available [here](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip).
The files are formatted as follows, with one example per line:
```
tok1 tok2 tok3 ...  label
```
Note that the label is separated from the tokens by a tab character.
The provided split contains 67,350 training examples, 873 development examples and 1822 test examples.
You can read more about the dataset in the original paper available [here](https://www.aclweb.org/anthology/D13-1170). 

### Vision
For a vision task we will use the [MNIST dataset of handwritten digits](https://en.wikipedia.org/wiki/MNIST_database). 
Examples in this dataset consist of 28x28 greyscale images with pixel values between 0 and 255, labeled with digits 0-9.
The data files we will use for this class are available [here](https://pjreddie.com/projects/mnist-in-csv/). 
The files are formatted as follows, with one example per line:
```
label, pix-11, pix-12, pix-13, ...
```
You can read more about the dataset in the original paper [here](https://papers.nips.cc/paper/1989/hash/53c3bce66e43be4f209556518c2fcb54-Abstract.html).
The provided split contains 60,000 training examples and 10,000 test examples.

Features
----

### Language 
For the language task, the tokens of text need to be converted into something consumable by a neural network. 
Today, we will use a bag-of-words representation.
A bag-of-words representation is a fixed-length vector of length $|V|$ where $|V|$ is the size of the vocabulary $V$.
To start, you can define the vocabulary as all unique words occurring in the training data.
For a given example, its bag-of-words vector will consist of a count $c_i$ of the number of times word $i$ occurred in the example text.
For example, consider $V$ = {dog, cat, lizard, green, beans, bean, sausage, the, my, likes, but} and the example sentence: 
```
my dog likes green beans but my cat likes sausage
```
Then the corresponding BoW vector would look like:

| dog | cat | lizard | green | beans | bean | sausage | the | my | likes | but |
| --- | --- | ------ | ----- | ----- | ---- | ------- | --- | -- | ----- | --- |
| 1   | 1   | 0      | 1     | 1     | 0    | 1       | 0   | 2  | 2     | 1   |

### Vision
The raw pixel values will work as input for now.
