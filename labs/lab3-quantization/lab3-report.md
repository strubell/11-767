Lab 3: Quantization
===
NoName:
---
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao ZhuGroup name:

1: Models & Model Size
----

model: original precision, orginal model size, target precision, parts quantitized

* BERT: float32, 418M, int8, [Self-Attention Layers, Fully Connected Layers] 
* AlBERT: float32, 45M, int8,  [Self-Attention Layers, Fully Connected Layers] 
* DistilBERT: float32, 254M, int8,  [Self-Attention Layers, Fully Connected Layers] 
* Longformer: float32, 568M, int8,  [Self-Attention Layers, Fully Connected Layers] 
* mobilenet_V2: float32, 14M, int8, [Conv2D, Fully Connected Layers]
* resnet18: float32, 45M, int8, [Fully Connected Layers]
* squeezenet: float32, 4.8M, int8, [Fully Connected Layers]
* mnasnet: float32, 17M, int8, [Fully Connected Layers]

We expect that the quantize models are (1) smaller: for transformers this should be significant since all the layers quantitized, while the reductions in vision models will depend on how much linear layers make up total parameters (with the exception of mobilenet_v2).

2: Quantization in PyTorch
----

We had some problems with the the QNNPack not being included in the wheel we were using. Reinstalling PyTorch with the wheel provided solved it. For the transformer language models, dynamic quantization was applied without any additional problems. Quantization was applied to all self-attention and fully connected layers -- because the self-attention heads are implemented as linear layers within the HuggingFace library.  

Furthermore, we were unable to use the torchvision quantized models for all image models as only MobileNetv2 was quantized using QNNPack, other models were quantized using FBGEMM which our torch installation was not built with. As a result, the quantization for the remaining vision models as only applied to the final fully connected layer.

3: Model Size
----
* BERT: 174M
* AlBERT: 23M
* DistilBERT: 132M 
* Longformer: 262M
* mobilenet_V2: 10M
* resnet18: 44M
* squeezenet: 4.8M
* mnasnet: 14M

4: Latency
----
Latency while varying image size and batch size for computer vision models is plotted below.

![Latency (s) per Image Size (width)](jetson_vision_imgsize.png)
![Latency (s) per Batch Size (width)](jetson_vision_batchsize.png)


Latency while varying image size and batch size for natural langugage / transformer models is plotted below. 

![Latency (s) per Sequence Size](jetson_language_seql.png)
![Latency (s) per Batch Size](jetson_language_batchsize.png)


5: Discussion
----

Regarding size, we see the expected compression of between 50-75% for the transformer models and for mobilenet. Suprisingly for the vision models, there was barely any compression, which might mean that most of the parameters is in the form of convolutions

With repsect to latency, we see the same relative performance differences as in the uncompressed models. However, initially we were suprised by the fact that quantized models were *slower* than the original model. However after realizing that quantized models run on the CPU, this made sense.
