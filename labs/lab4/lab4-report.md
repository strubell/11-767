Lab 4: Baselines
===
NoName:
---
Patrick Fernandes, Jared Fernandez, Haoming Zhang, Hao Zhu

1: Related Work
----
## Adaptive Computation

TODO: 

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
