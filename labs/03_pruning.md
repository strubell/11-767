Lab 3: Pruning
===
This lab is an opportunity to explore different pruning strategies and settings, with the goal of becoming familiar with unstructured pruning, iterative and non-iterative pruning, and criteria for selecting parameters for pruning.

Throughout this lab, we will use the term **sparsity level** to refer to the percent of the original model's *prunable* weights (**weight parameters, excluding bias parameters**) that have been pruned.

Preliminaries & Setup
---
0. Share your hardware specs
1. Copy over relevant code for training MNIST from Lab 2 (just the "Lab 2" model), including code for evaluation (in particular, accuracy, latency, and size on disk), *but don't train yet*!

| hyperparameter  | value |
| --------------- | ----- |
| learning rate   | 0.001 |
| batch size      | 64    |
| hidden size     | 1024  | 
| # hidden layers | 2     |
| training epochs | 2     |

Recall from Lab 2 that you can measure model size on disk in this way:

```
import os

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size
```
You can then compare model sizes:

```
f=print_size_of_model(model1, "model1")
q=print_size_of_model(model2, "model2")
print("{0:.2f} times smaller".format(f/q))
```

2. Initialize your model. Before training, **SAVE** your model's initial (random) weights. You will use them later for iterative pruning
3. Now train the base model and report:
   - dev accuracy, 
   - inference latency,
   - number of parameters,
   - space needed for storage (in MB) of `state_dict()` on disk

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |    ?    |    ?     |    ?    |

Also take a look at your model's `named_parameters()`. You'll need these later (no need to put in the table).


Basic pruning + more preliminaries
---
First, you will perform global, unstructured magnitude (L1) pruning on the model to a sparsity level of **30%**. Prune just the weight parameters (not biases). 
You should be able to use the `global_unstructured` pruning method in the PyTorch prune moduse.
For usage examples, see the [pytorch pruning tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html). 

Example input:
```
[(model.layers[0], 'weight'),
(model.layers[1], 'weight'),
(model.out, 'weight')]
```
or
```
[(m[1], "weight") for m in model.named_modules() if len(list(m[1].children()))==0]
```
Take a look at your `model.named_parameters()` again, and your `model.named_buffers()`. (Just observe, no need to answer a question.)

4. Write functions to calculate the sparsity level (using the percent of buffers that are 0):
    -  for each layer,
    -  for all pruned layers, and
    -  for the model overall
   **And report each of these values:** the sparsity level at each layer, across all pruned layers, and for the model overall.
5. Write a function to calculate the amount of space that a pruned model takes up when reparameterization is removed and tensors are converted to *sparse* representations.
[**TODO**: somehow share or paste in the notebook's "aside" which should give them a hint]

Using your new disk size function, fill in the next row of the same table:

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |        |         |        |
|     1   |   30.0%   |    ?    |    ?    |    ?    |


Repeated unstructured magnitude pruning
---
Now, keep performing the same unstructured magnitude pruning of 30% of the remaining weights on the same model without re-training or resetting the model. 
You will apply the same function as above with the same 0.3 proportion parameter.

6. Collect values for the rest of this table, and plot them. Your plot should have **accuracy** on the x axis and **sparsity** on the y axis. 
Sparsity reported should be the percentage of *prunable* parameters pruned.

| Iteration | Sparsity (%) | Accuracy | Latency (s) | Disk Size (MB) |
| ------- | ------- | ------ | ------- | ------ |
|     0   |   0.0%  |        |         |        |
|     1   |   30.0%   |        |         |        |
|     2   |         |        |         |        |
|     3   |         |        |         |        |
|     4   |         |        |         |        |
|     5   |         |        |         |        |
|     6   |         |        |         |        |
|     7   |         |        |         |        |
|     8   |         |        |         |        |
|     9   |         |        |         |        |

Here is some example code you might use to plot these values using Matplotlib:
```
TODO
```

[**Discussion question**] Comment on disk size and latency and why these models might be bigger (on disk) or slower (inference latency). 

Iterative magnitude pruning
---
TODO: with and without rewinding. Plot all three on the same plot.
(basically repeat above section with extra steps in between pruning steps. loading and saving shouldn't be too big a deal, or if it is, we can just give them crucial code. )

(if asking for more iterations, only ask them to report vaguely like "when" they start to see performance drop off, and/or plots!)



Extra Credit
---
#### 1. Implement a custom pruning method [1 point]
Implement an additional pruning method, and report your results. For example, you might implement structured pruning, second-order pruning, or anything else. Be creative! In order to get full credit, you must clearly describe your approach, and why you think it should work. You should apply your approach in the iterative magnitude pruning paradigm, and perform multiple iterations of pruning.

#### 2. Combining quantization and pruning [1 point]
Combine quantization from Lab 2 with iterative magnitude pruning from this lab, and report your results in terms of accuracy and size on disk. You can combine the approaches however you wish, but to get full credit you must clearly describe what type of quantization you used and how exactly you combined the approaches, and why you think that should work. You only have to report results for one sparsity level. 


Grading and submission (10 points + 2 extra credit)
----
Submit your answers to all the above questions to Canvas as a write-up in pdf format. This assignment is worth 10 points 
(or 10% of your final grade for the class), distributed as follows: 
- **Submission [2 points]:** Assignment is submitted on time.
- **Basic requirements [5 points]:** Answers to all the questions (including all requested tables) are included in the write-up. 
- **Report [2 points]:** Report is well-written and organized with minimal typos and formatting mistakes, and answers to all requested questions are easy to find. Tables are readable and well-labeled. Your results should be easily reproducible from the details included in your report.
- **Thoughtful discussion [1 points]:** The discussion of results in your write-up is thoughtful: it indicates meaningful engagement with the exercise and it is clear that you put effort into thinking through and explaining your results.
- **Extra Credit [2.5 points]:** See above for description of possible extra credit.
