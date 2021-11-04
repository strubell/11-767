Lab 5: Group work on projects
Group name:
1: Plan
The main plan for this week is to build a unified framework for experimentations on both adaptive computation and iterative model deserialization. 
Previous week’s baselines were all based on the BERT family of models, but both branches of the project used different models (ALBERT vs BERT-base and BERT-small). It is therefore better to use a similar set of baselines for both branches
Besides this, both branches were using different codebases, so the plan for this week is also to unify the codebase for ease of comparison and integration of techniques from both branches in the final unified approach.
The following distribution of work is proposed:
All members will discuss the best baselines to use for both branches based on the model properties
Tony will implement PABEE for DistillBERT as a new baseline
Jared will train models for the new DistilBERT baseline
Jared and Tony will work on Jetson deployment of the finetuned DistilBERT-PABEE models
Hao will implement a prototype for checkpoint division
Patrick will benchmark memory consumption per layer of the divided checkpoint
Patrick will start implementation of the integration of the checkpoint deserialization into the PABEE framework
All members will contribute to this week’s report
2: Execution
Progress has made in various aspect of the project during this week:
We had discussions on what the best baselines to have for both branches of the project:
Keep BERT-base: this is the core model for the final approach since naively it doesn't fit into the memory of the Jetson 2Gb, therefor making it a good target for iterative deserialization
Replace BERT-small with DistillBERT: both these models are good smaller baselines to compare with BERT-base that fit into the device. However DistillBERT is a more realistic baseline since it achieves a closer performance to BERT-base
Removed ALBERT: the shared weights for all layers made it not well suited for the iterative deserialization part of our project since there would be less benefits memory-wise
We choose to integrate the iterative deserialization method into the adaptive computation codebase
The reason for this choice is that the adaptive computation models already explicitly separated the computation between different layers due the off-ramping mechanism, making it suitable to later just including an algorithm for destroying old layers and states while de-serializing news in between layer computations
To divide the checkpoints, we grouped the parameters with their names. Parameters in the same layer are stored in the same checkpoint. The above graph shows the trend of memory consumption during load the parameters layer by layer. 
We implemented PABEE for DistilBERT, and trained models for five GLUE downstream tasks (CoLA, MNLI, MNLI-MM, QNLI, QQP)
The codebase of PABEE only supports BERT and ALBERT. So we implemented the DistilBERT part on top of it.
Due to torch.nn version incompatibilities, new issues emerged in the deployment of inference on device. While these are being resolved, complete results for DistilBERT-PABEE are still in progress. 
 
We were also hoping to finish integrating a prototype of iterative deserialization into the PABEE framework. However we realised that this would be more work than expected for one week (and it would almost finish the iterative deserialization part) so implementation is not complete yet. 
3: Next steps
Adaptive Computing
Jared and Tony are making steady progress towards deployment of early-exit models on the edge devices. Furthermore, we are gaining familiarity with the architecture of the baseline model and offramp modules as well as the heuristics/thresholds used to determine exiting. We are using the knowledge we have learned to deploy additional models with adaptive exiting on device -- and most importantly, we are using the knowledge we have learned from these initial experiments to design a cost function conditioned on model latency module to determine exiting. Jared and Tony will brainstorm potential formulations of this module and begin implementing the control flow, modules, and training in the next couple of weeks.  
 
Iterative Deserialization 
Hao and Patrick will work on integrating the current checkpoint division method into PABEE models. The most challenging part is the design of a generic method for dividing both the code and checkpoints for both vision and transformer models. In theory, one can run topological sorting on the computation graph to account for all kinds of different architectures. However, this will involve huge engineering efforts. We plan to study the codebase of huggingface transformer and torchvision to figure out less generic but more feasible ways.  

