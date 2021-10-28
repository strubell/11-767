Lab 4: Baselines
===
NoName:
---
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao Zhu

1: Related Work
----
## Adaptive Computation

- [BERT](https://arxiv.org/pdf/1810.04805.pdf)

BERT is a pretrained Transformers language model that learns bidirectional representations from unlabeled text by jointly conidtioning on both left and right context in all layers. It is pretrained on  13 GB text data with a combination of masked LM loss and next sentense prediction loss. The pretrained BERT can be finetuned with one additional output layer for various of downstream tasks. BERT is evaluated on benchmark datasets, including GLUE and SQuAD, and achieves state-of-the-art performances at the time. 

As a lot of follow-up works, our project aims to build upon BERT and improve the performance of BERT on resource limited devices, such as Jetson Nano. From our preliminary experiments, running BERT-base on 2GB Jetson Nano leads to out of memory. Our hypothesis is that we can run BERT-base's inference computation graph on a 2GB device by iteratibely instantiaing the model's parameters. Also, our experiments for adaptive computing with hardware features will uild upon BERT.

- [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)

RoBERTa presents a replication study of BERT. The authors pretrained the model with several modifications durign pretraining, which include:  (1) training the model longer, with bigger batches, over more data; (2) removing the next sentence prediction objective; (3) training on longer sequences; and (4) dynamically changing the masking pattern applied to the training data. The work shows that the original BERT is undertrained and it achieves better performance than BERT in nearly all the downstream tasks. For our project, we will use RoBERTa as another pretrained model target for our adaptive computing experiments. 

- [ALBERT](https://arxiv.org/pdf/1909.11942.pdf)

ALBERT proposes two parameter reduction techniques to address the memory limitation problem in scaling pretrained language models, BERT specifically. The first one is decomposing the vocabulary embedding matrix into two small matrix, which allows growing the hidden size without siginicantly increasing the parameter size of vocabulary embeddings. The second technique is cross-layer parameter sharing, which prevents the parameter from growing with the depth of the network. With these two techniques, the authors ends up with models with less parameters, lower latencies, comparable or even better performance than BERT. Similar as RoBERTa, we is ALBERT as another another pretrained model target for our adaptive computing experiments in this project.

- [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf)

DistillBERT is a smaller transformer-based model trained through knowledge distillation from BERT. It is trained with a combination of three different losses: distillation loss, masked LM loss, and cosine embedding loss. The model ends up to be 40% smaller than a BERT model, while  retaining 97% of its language understanding capabilities and being 60% faster. DistillBERT provides a smaller, faster and lighter model suitable for on-device computations. In our project, we will compare our iterative ayer instatiation method to DistillBERT as an alternative on-device solution with zero loss in performance but higher latency. 


- [DeeBERT](https://aclanthology.org/2020.acl-main.204.pdf)

DeeBERT introduces the technique of dynamic early exiting to accelerate BERT Inference. Specifically, it inserts extra classification layers (referred as off-ramps) between each transformer layer of BERT during training. At inference time, the model will output the prediction of the first off-ramp where the entropy of the probavility distribution is less than the preset threshold. This method saves up to 40% inference time for both BERT and RoBERTa with minimal degradation in model quality. For the adaptive computing part of our project, we will build upon DeeBERT by including time constraints and hardware features during training. We will compare our experiment results to DeeBERT and hope to see some improvement in performance.

- [BERT Loses Patience](https://arxiv.org/pdf/2006.04152.pdf)

Similar as DeeBERT, PABEE adopted early exiting method to accelerate BERT Inference. Rather than comparing entropy to threshold at inference time, PABEE exits with the output of an off-ramp layer when the intermediate predictions of the internal classifiers remain unchanged for t times consecutively, where t is a pre-defined patience. For experiments on ALBERT, PABEE can even improve the perfermance on downstream tasks, while speeding up inference time simultaneously. For our project, we will use PABEE as the baseline for our adaptive computing part.

- 

## Tensor Computation with Heterogeneos Memory

## Adaptive Computation

2: Baselines
----

As a general test bed we will consider models in BERT (**TODO**: cite) family as baselines and we will evaluate the performance the GLUE benchmark, and in particular on the CoLA, MNLI, MNLI-MM, QNLI, QQP tasks. We will evaluate the model in terms of accuracy, memory usage and latency.
In particular given the two apparent *branches* of the project we will consider two slightly different variations as baselines

## Tensor Computation with Heterogeneos Memory

(**TODO**: elaborate more) BERT-base, BERT-base quantitized, BERT-small. Expected: BERT-base won't fit. BERT-base-qt will fit, BERT-small will fit, will be faster and use less memory than BERT-base-qt but will be less accurate. 

(**TODO**: Table with columns - GLUE Tasks, with subcolumns of Acc, Latency and Memory per sample - rows will be baseline names) 

## Adaptive Computation

1. We will use PABEE (code available at [link](https://github.com/huggingface/transformers/tree/master/examples/research_projects/bert-loses-patience)) on ALBERT model as our baseline for the adaptive computation branch. We will adopt a pretrained ALBERT checkpoint from transformers library. We will use (**TODO**: machine config) to fine-tune the model with PABEE implementation on **TODO** tasks from GLUE benchmark datasets, with the original train test splits. We fine-tune each model for 5 epochs and then evaluate models on our 4GB Jetson Nano device. 

2. Yes, the baselines are able to run on device.

3. After training the baselines on clusters, we will evaluate the models on 4GB Jetson Nano device, with the corresponding validation data for each downstream task. We will compare the latency and GLUE score with the original ALBERT models to evaluate the performance of PABEE. 

3: Extra
----
(**TODO**: preliminary results with iterative desirialization)
