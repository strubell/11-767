Lab 3: Quantization
===
The goal of this lab is for you to benchmark and compare model size and inference efficiency **between quantized and original models** on your devices. You should benchmark the same models as you benchmarked last lab, so ideally **2*N* models or model variants, where *N* is the size of your group (so, two models per person.)** For now, if you don't have appropriate evaluation data in place that's fine; you can provide pretend data to the model for now and just evaluate efficiency.

Ideally, the models you benchmark will be the same as last class, but if you find that you're unable to run out-of-the-box quantization on your models, feel free to try quantizing other models, or a subset. Just be sure to explain what you tried, and why.

Include any code you write to perform this benchmarking in your Canvas submission (either as a link to a folder on github, in a shared drive, zip, etc).

Group name:
---
Group members present in lab today:

1: Models
----
1. Which models and/or model variants will your group be studying in this lab? What is the original bit width of the models, and what precision will you be quantizing to? What parts of the model will be quantized (e.g. parameters, activations, ...)? Please be specific.
2. Why did you choose these models?
3. For each model, you will measure model size (in (mega,giga,...)bytes), and inference latency. You will also be varying a parameter such as input size or batch size. What are your hypotheses for how the quantized models will compare to non-quantized models according to these metrics? Do you think latency will track with model size? Explain.

2: Quantization in PyTorch
----
1. [Here is the official documentation for Torch quantization](https://pytorch.org/docs/stable/quantization.html). 

3: Model size
----
1. Compute the size of each model. Given the path to your model on disk, you can compute its size at the command line, e.g.:
   ```
   
   ```
2. Any difficulties you encountered here? Why or why not?

4: Latency
----
1. Compute the inference latency of each model. You can use the same procedure here as you used in the last lab. Here's a reminder of what we did last time: 
   You should measure latency by timing the forward pass only. For example, using `timeit`:
    ```
    from timeit import default_timer as timer

    start = timer()
    # ...
    end = timer()
    print(end - start) # Time in seconds, e.g. 5.38091952400282
    ```
    Best practice is to not include the first pass in timing, since it may include data loading, caching, etc.* and to report the mean and standard deviation of *k* repetitions. For the purposes of this lab, *k*=10 is reasonable. (If standard deviation is high, you may want to run more repetitions. If it is low, you might be able to get away with fewer repetitions.)
    
    For more information on `timeit` and measuring elapsed time in Python, you may want to refer to [this Stack Overflow post](https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python).
2. Repeat this, varying one of: batch size, input size, other. Plot the results (sorry this isn't a notebook):
   ```
   import matplotlib.pyplot as plt
   
   plot_fname = "plot.png"
   x = ... # e.g. batch sizes
   y = ... # mean timings
   
   plt.plot(x, y, 'o')
   plt.xlabel('e.g. batch size')
   plt.ylabel('efficiency metric')
   plt.savefig(plot_fname)
   # or plot.show() if you e.g. copy results to laptop
   ```
4. Any difficulties you encountered here? Why or why not?

5: Discussion
----
1. Analyze the results. Do they support your hypotheses? Why or why not? Did you notice any strange or unexpected behavior? What might be the underlying reasons for that behavior?

5: Extra
----
A few options:
1. Compute FLOPs and/or energy use (if your device has the necessary power monitoring hardware) for your models. 
2. Evaluate on different hardware (for example, you might run the same benchmarking on your laptop.) Compare the results to benchmarking on your device(s).
3. Use real evaluation data and compare accuracy vs. efficiency. Describe your experimental setup in detail (e.g. did you control for input size? Batch size? Are you reporting average efficiency across all examples in the dev set?) Which model(s) appear to have the best trade-off? Do these results differ from benchmarking with synthetic data? Why or why not?

----
\* There are exceptions to this rule, where it may be important to include data loading in benchmarking, depending on the specific application and expected use cases. For the purposes of this lab, we want to isolate any data loading from the inference time due to model computation.
