NoName:
---
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao Zhu

1: Models
----
* BERT
* AlBERT
* DistilBERT
* Longformer
* mobilenet_V2
* resnet18
* squeezenet
* vgg16

TODO: make hypothesis. examples: (a) longformer will not consume much more memory/latency per sequence size

2: Parameter count
----

* AlBERT: 11.7M
* BERT: 109.5M
* distilBERT: 66.4M
* longformer: 148.7M
* mnasnet: 4.4M
* mobilenet_v2: 3.5M
* resnet18: 11.7M
* squeezenet: 1.2M

No difficulties were found when computing total parameters

3: Latency
----

Latency while varying image size and batch size for computer vision models is plotted below.

![Latency (s) per Image Size (width)](vision_imgsize.png)
![Latency (s) per Batch Size (width)](vision_batchsize.png)


Latency while varying image size and batch size for natural langugage / transformer models is plotted below. 
Since Longformer was major outlier, we also plot latency vs batch size without it.

![Latency (s) per Sequence Size](nlp_sequencesize.png)
![Latency (s) per Batch Size](nlp_batchsize.png)
![Latency (s) per Batch Size (w/o Longformer)](nlp_batchsize_nolf.png)

The main problem came from the memory requirments for transformer models.
 We suspect that, for example, the latency of longformer is due to swap usage and we can see that we cannot even run it for batch size > 2. 
Other transformers run out of space at batch size > 4.


4: Energy use
----
1. Compute the energy use of each model. You can use the `powertop` tool on RPi and Jetson (must be run as root):
    ```
    sudo apt install powertop
    ```
    and/or the `jtop` tool on Jetson (see installation instructions [here](https://github.com/rbonghi/jetson_stats/)). 
    
    Follow the same procedure as you used to compute latency, but this time compute energy: (avg) watts * time. You will likely need to sample power a number of times throughout each inference, and average.
    
    By default, `powertop` takes measurements every 20 seconds. You can change it with the `--time` parameter, which specifies number of seconds and allows for non-integer intervals (0.5 for half a second) e.g. to poll every second for 10 seconds and write to 10 csv files:
    ```
    sudo powertop --time=1 --csv=powertop.log --iteration=10
    ```
    Here is a link to the [`powertop` users guide](https://01.org/sites/default/files/page/powertop_users_guide_201412.pdf) [PDF].
2. Any difficulties you encountered here? Why or why not?

5: Discussion
----

TODO: discuss hypothesis

Hypotheses:

1. We assume that for both vision and language models, the latency for each batch is proportional to the batch size.
2. We assume that for vision models, the latency for each batch is proportional to the number of pixels for each image, i.e., proportional to the square of batch size.
3. We assume that for transformers models, the latency for each batch is proportional to sequence lengths, except the Longformer, whose latency will be constant with varying sequence lengths according to its paper.

Discussion:

* The results from section 3 are consistent with hypotheses 2 and 3.
* For hypothesis 1, the results for vision models align with our hypothesis. However, the latency for AlBERT, BERT, and DistillBERT seems to be constant with varying batch sizes. One possible explanation could be that there are some other bottlenecks during the computations which take longer time than matrix multiplications since our devices can only support the relatively small batch size. To verify our assumption, we will run the experiments on GPU servers with larger batch sizes in the next section.

5: Extra
----

### Server Data

TODO (add server specs)

Latency while varying image size and batch size for computer vision models is plotted below.

![Latency (s) per Image Size (width)](vision_imgsize_server.png)
![Latency (s) per Batch Size (width)](vision_batchsize_server.png)


Latency while varying image size and batch size for natural langugage / transformer models is plotted below. 

![Latency (s) per Sequence Size](nlp_sequencesize_server.png)
![Latency (s) per Batch Size](nlp_batchsize_server.png)

----
\* There are exceptions to this rule, where it may be important to include data loading in benchmarking, depending on the specific application and expected use cases. For the purposes of this lab, we want to isolate any data loading from the inference time due to model computation.
