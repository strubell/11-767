# Lab 1: Project Proposal Outline
---

*NoName*
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao Zhu
---


## Ideas

During the brainstorming, we came up with the following ideas:

* **Real-world Visual Captioning with Camera+TTS**: 
Visual Scene Captioning is the task of taking an image and providing a description in natural language. The core idea here would be to take existing models and distill/prune/quantize them such that they would run in real-time in a Jetson. We could then couple them with the Camera and Speaker peripherals (plus a text-to-speech system) to provide a rough prototype of an on-device visual captioning system that we could experiment with.
* **Real-world Visual Question Answering with Camera+Microphone**:
Similar to the previous task, however now rather than describing an image, the task is to answer a (categorically) question posed in natural language about an image. The goal here would be to get such a model to run in the Jetson in useful time and couple it with the necessary peripheral to build an experimental prototype
* **Multilingual OCR-TTS**: Systems such as Google Lens allow users to take a picture of text and provide and modification of the image with text translated. The core of this idea would be to implement such a system. Could we create a system that would take a picture of text and provide the user with translations? What trade-off in terms of quality needs to be done for it to fit into the Jetson? 
* **On-device Speech-to-Speech(S2S) translation**: Speech2Speech is a useful real-world task, with many companies proposing systems (see *PocketTalk*). However, most systems are edge-devices, requiring connections to a server to work. Could we get such a system to work on-device? What is better for a constrained scenario, a cascaded or end-to-end approach? 
* **Incremental Model Deserialization**: On-device accelerators are limited in terms of both memory and inference speed. However, while inference speed can be problematic, memory is a more serious problem since if the model is too large it is not even possible to run it. Given a scenario where latency is not a huge requirement, could we devise an algorithm for "incremental" model deserialization, where only part of the model is loaded from persistent storage into RAM, and computation is performed partially each time? What trade-off do we need to do in terms of latency?
* **Adaptive Computation for On-Device Machines**: Full execution of neural networks on edge devices is constrained by the real-world runtime of such computations. To reduce the overhead for an individual model prediction, we propose developing strategies for dynamically selecting a subset of input features and model layers to operate on during inference. Selection can be determined by heuristics or learned parameters. By adjusting the amount of computation to the difficulty of examples, we can limit the amount of computation used by easier examples.


## Narrowing

After the brainstorming session, the group was torn between the more "practical" projects based around designing a prototype system based on existing, well-studied architectures and making them work on an on-device setting (the first four), and the more "research" projects around exploring new ideas to make models be more compute and memory-efficient (the latter two).
While the first group would certainly involve research, the group was generally more interested in the potential implications of the latter two projects. 

Also, we realized that these two ideas also fitted nicely with each other since we could deserialize layers as they were needed by the adaptive computation model. Given this and after a discussion with the professors, we ended up choosing as our proposed idea a chimera of the latter two ideas


## Project Proposal: *Adaptive Computing with On-Demand Model Instantiation*

### Motivation

Modern state-of-the-art systems often have billions of parameters, which makes it difficult to run on small devices with limited memory (e.g. Jetson). However current methods for circumventing this have trade-offs in terms of accuracy that are not always desirable.

### Hypotheses

With the help of adaptive compute and smart de-serialization, we can optimize the latency under the constraints of on-device memory for large models without losing accuracy. This is in contrast to the traditional pruning, distillation, and quantization methods, where one has to tradeoff between accuracy and model size/latency. 

### Setup

#### Datasets
- [GLUE benchmark](https://gluebenchmark.com/)
- [MSCOCO](https://cocodataset.org/#home)
- [VQA v2.0](https://visualqa.org/)

#### Hardware & IO
- Jetson Nano 2 GB
- Jetson Nano 4 GB (if possible)
- Camera, Microphone for practical experiments

#### Experiments

Evaluations will be performed using standard benchmark tasks in vision and language, including image classification, GLUE tasks, and visual question answering.

Currently, state-of-the-art models are too large to load into memory of standard edge accelerators. To evaluate the success of model serialization, we will determine the largest models that can be loaded into memory without any modification and observe whether model serialization can enable larger models to be executed at all. We will experiment with loading different numbers of parameters into memory and serialize the model at different frequencies.

We will evaluate the runtime speedup of our proposed adaptive inference methods by measuring the average latency for forward passes through the model. As these methods will reduce the total model parameters used, we expect to observe performance degradation. We will experiment with early exiting and feature selection methods to determine the relationship between inference speedups and resulting model performance.

To evaluate the impact of our proposed method on real-world scenarios, we plan to experiment with image and natural language classification *on-the-wild* to see how adaptive computation and model de-serialization affect latency and memory usage.