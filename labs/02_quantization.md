Lab 2: Quantization
===
(overview)

Hardware
----
You are welcome to use any hardware to which you have acccess for this assignment as long as you clearly include that information in your report, but *please note that PyTorch Quantization 
is not currently supported for the Apple M series chips.* So if you have a newer Macbook (November 2020 or later), you will need to use Colab or another compute resource. 
One way to check whether PyTorch quantization is supported for your hardware is to run the following:
```
import torch.quantization
torch.backends.quantized.engine
```
If the result is not `fbgemm` then your hardware may not be supported.

1. Exactly what hardware are you using for this assignment? Report the CPU, RAM and any graphics accelerator, in as much detail as is available to you.

Data, Models and Evaluation
----
Data, models and evaluation for this assignment will build off the code you wrote for Lab 1. You are encouraged (and expected) to re-use model and evaluation
code from Lab 1 to complete this assignment.

For the models in this assignment you will experiment with variants of the feed-forward networks with ReLU activations that you 
implemented in Lab 1. You will again experiment with a computer vision model trained on MNIST, and a text sentiment analysis model trained on SST-2. 
Here are the model and training hyperparameters you should use for the MNIST model:

| hyperparameter  | value |
| --------------- | ----- |
| learning rate   | 0.001 |
| batch size      | 64    |
| hidden size     | 1024  | 
| # hidden layers | 2     |
| training epochs | 2     |

You should also crop the input to 20x20 pixels at the center of the image.

2. Train this model and report its accuracy on the MNIST test set. You should be able to get 95.8% accuracy.

and here are the hyperparameters you shuold use for the SST model: 

| hyperparameter  | value |
| --------------- | ----- |
| learning rate   | 0.001 |
| batch size      | 64    |
| hidden size     | 512   | 
| # hidden layers | 2     |
| training epochs | 2     |

For SST, you should threshold your vocabulary to the 5000 most frequent words in the vocabulary. 

3. Train this model and report its accuracy on the SST development set. You should be able to get about 80% accuracy.

In this lab you will be evaluating models in terms of model size and inference latency. For inference latency report the average
inference time and standard deviation measured over 5 runs. For model size, 

4. Evaluate and report the inference latency of both models for batch sizes in [1, 64]. 

5. Report the size of both models.


Implementing a linear mapping function
----


Dynamic quantization in PyTorch
----
Now that you have an idea of how quantization works under the hood, follow the [PyTorch quantization tutorial](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) 
to implement PyTorch dynamic quantization for your models.


Static quantization in PyTorch
----

Extra Credit: Benchmarking dynamic quantization
----


Grading and submission (100 points / 10% of your final grade)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 100 points 
(or 10% / 10 points of your final grade for the class), distributed as follows: 
- **Submission [20 points]:** Assignment is submitted on time.
- **Basic requirements [50 points]:** Answers to all the questions (including all requested plots) are included in the write-up. See points breakdown as indicated above.
- **Report [20 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Plots are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [10 points]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
- **Extra Credit [5 points]:** See above for description of possible extra credit.
