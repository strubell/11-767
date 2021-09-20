# Lab 1: Project Proposal Outline
---

*NoName*
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao Zhu
---


## Ideas

During the brainstorming, we came up with the following ideas:

* **Real-world Visual Captioning with Camera+TTS**: 
Visual Scene Captioning is the task of taking an image and providing a description in natural language. The core idea here would be to take existing models and distil/prune/quantize them such that they would run in real time in a Jetson. We could then couple them with the Camera and Speaker pheriphericals (plus a text-to-speach system) to provide a rought prototype of a on-device visual captioning system that we could experiment with.
* **Real-world Visual Question Answering with Camera+Microphone**:
Similar to the previous task, however now rather than describing an image, the task is to answer (categorically) question posed in natural language about an image. The goal here would be to get such a model to run in the Jetson in useful time and couple it with the necessary pheripherical to build a experimentable prototype
* **Multilingual OCR-TTS**: Systems such as Google Lens allow users to take a picture of text and provide and modification of the image with text translated. The core of this idea would be to implement such a system. Could we create a system that would take a picture of text and provide the user with translatations? What trade-off in terms of quality need to be done for it to fit into the Jetson?  
* **On-device Speech-to-Speech(S2S) translation**: Speech2Speech is a useful real-world task, with many companies proposing systems (see *PocketTalk*). However most systems are edge-devices, requiring connections to a server to work. Could we get such a system to work on-device? What is better for constrained scenario, a cascaded or end-to-end approach?  
* **Incremental Model Deserialization**: On-device accelarators are limited in terms of both memory and inference speed. However, while inference speed can be problematic, memory is a more serious problem since if the model is too large it is not even possible to run it. Given a scenario where latency is not a huge requriment, could we devise an algorithm for "incrementental" model deserialization, where only part of the model is loaded from persistant storage into RAM, and computation is performed partially each time? What trade-off do we need to do in terms of latency?
* **Adaptive Computation for On-Device Machines**:  Full execution of neural networks on edge devices is constrained by the realworld runtime of such computations. To reduce the overhead for an individual model prediction, we propose developing strategies for dynamically selecting a subset of input features and model layers to operate on during inference. Selection can be determined by heuristics or learned parameters. By adjusting the amount of computation to the difficulty of examples, we can limit the amount of computation used by easier examples.


## Narrowing

TODO: describe the gap between the first four ideas and the last two (pratical vs researchy)
TODO: describe the merging of the last two ideas (*adaptive computing with on-demand model instantiation*)
TODO: describe choice of the latter (?)

## Project Proposal

### Motivation

### Hypotheses

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
