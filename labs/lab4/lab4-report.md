Lab 4: Baselines
===
NoName:
---
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao Zhu

1: Related Work
----
## Adaptive Computation

- [BERT](https://arxiv.org/pdf/1810.04805.pdf)

BERT is a pretrained Transformers language model that learns bidirectional representations from unlabeled text by jointly conidtioning on both left and right context in all layers. It is pretrained on  13 GB text data with a combination of masked LM loss and next sentense prediction loss. The pretrained BERT can be finetuned with one additional output layer for various of downstream task. BERT is evaluated on benchmark datasets, including GLUE and SQuAD, and achieves state-of-the-art performances at the time. 

As a lot of works later than BERT, our project aims to build upon BERT and improve the performance of BERT on resource limited devices, such as Jetson Nano. From our preliminary experiments, running BERT-base on 2GB Jetson Nano will leads to out of memory. Our hypothesis is that we can run BERT-base's inference computation graph on a 2GB device by iteratibely instantiaing the model's parameters. Also, our experiments for adaptive computing with hardware features will uild upon BERT.

- [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)

RoBERTa presents a replication study of BERT. The authors pretrained the model with several modifications durign pretraining, which include:  (1) training the model longer, with bigger batches, over more data; (2) removing the next sentence prediction objective; (3) training on longer sequences; and (4) dynamically changing the masking pattern applied to the training data. The work shows that the original BERT is undertrained and it achieves better performance than BERT in every downstream task. For our project, we will use RoBERTa as another pretrained model target for our adaptive computing experiments. 

- [ALBERT](https://arxiv.org/pdf/1909.11942.pdf)

ALBERT proposes two parameter reduction techniques to address the memory limitation problem in scaling pretrained language models, BERT specifically. The first one is decomposing the vocabulary embedding matrix into two small matrix, which allows growing the hidden size without siginicantly increasing the parameter size of vocabulary embeddings. The second technique is cross-layer parameter sharing, which prevents the parameter from growing with the depth of the network. With these two techniques, the authors ends up with models with less parameters, lower latencies, comparable or even better performance than BERT. Similar as RoBERTa, we is ALBERT as another another pretrained model target for our adaptive computing experiments in this project.

- [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf)

DistillBERT is a BERT model with less layers trained through the technique of knowledge distillation. It is trained with a combination of three different losses, including distillation loss, masked LM loss and cosine embedding loss. The model ends up to be 40% smaller than a BERT model, while  retaining 97% of its language understanding capabilities and being 60% faster. DistillBERT provides a smaller, faster and lighter model suitable for on-device computations. In our project, we will compare our iterative ayer instatiation method to DistillBERT as an alternative on-device solution with zero loss in performance but higher latency. 


- DeeBERT
- BERT Loses Patience
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

3: Extra
----
(**TODO**: preliminary results with iterative desirialization)
