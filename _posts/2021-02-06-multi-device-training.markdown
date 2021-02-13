---
layout: post
title: "Distributed Multi-Device Training (Draft)"
date: 2020-08-03 15:39:00 +0900
image_url: ""
mathjax: true
comments: true
---

# Table of Contents
1. [Introduction](#Introduction)
2. [Data Parallelism](#Data-Parallelism)
3. [Model Parallelism](#Model-Parallelism)
4. [Pipeline Parallelism](#Pipeline-Parallelism)
5. [Collective Commnuication](#Collective-Commnuication)
    1. [Frameworks](#Frameworks)
    2. [In Data Parallelism](#In-Data-Parallelism)
    3. [In Model Parallelism](#In-Model-Parallelism)
    4. [In Pipeline Parallelism](#In-Pipeline-Parallelism)
6. [Frameworks for Parallelism](#Frameworks-for-Parallelism)
    1. [Tensorflow](#Tensorflow)
    2. [PyTorch](#PyTorch)
7. [Conclusion](#Conclusion)

# [Introduction](Introduction)
There are several strategies used to train a deep learning model with multi devices. In order to train a model across multiple devices, deep learning frameworks provide some features for distributed training such as:  
1. Data Parallelism
2. Model Parallelism
3. Pipeline Parallelism

Each parallelism scheme has pros and cons, and engineers should decide among these to efficiently exploit their devices.

# [Data Parallelism](Data-Parallelism)
Data Parallelism is well-known distributed method for training deep learning model. The notion of data parallelism is not only in deep learning domain but in plenty of other domains. [SIMD](https://en.wikipedia.org/wiki/SIMD) instructions process multiple data simultaneously within one instruction, which is one of the data parallelism. Also, [SPMD](https://en.wikipedia.org/wiki/SPMD) programming model supports engineers to effectively do parallel programming. Data parallelism with multiple devices is known as batch-splitting meaning that the task is splited into subtasks and each device conducts a subtask. For example, with (256, 32, 32, 3)-shaped input and 4 GPUs, it is easy to divide input into 4 (64, 32, 32, 3)-shaped inputs because there is no dependence among batch axes in common deep learning task. 

Of course, layers like Batch Normalization have to be synchronized across all subtasks so that means and variances are the same across multiple devices. We will going to talk about this later.

## Implementation
The implementation of data parallelism varies. Here I introduce common concept and algorithm of batch-splitting.  
0. Copy all parameters to each device.
1. For each iteration, split the training batch into sub-batches.
2. Distribute one sub-batch for one device.
3. Each device computes the forward and backward passes on its-batch.
4. Sum all the gradients on devices and distribute the sum.
5. Update the model parameters.

# [Model Parallelism](Model-Parallelism)
Although data parallelism is dominant strategy for training on multiple devices, it suffers from the inability to train very large models due to memory constraints of GPU. 

# [Pipeline Parallelism](Pipeline-Parallelism)

# [Collective Commnuication](Collective-Commnuication)
## Frameworks
## In Data Parallelism
## In Model Parallelism
## In Pipeline Parallelism

# [Frameworks for Parallelism](Frameworks-for-Parallelism)
## Tensorflow
### Mesh-Tensorflow
## PyTorch
### DeepSpeed

# [Conclusion](Conclusion)

# References
1. [PyTorch Distributed: Experiences on Accelerating
Data Parallel Training](https://arxiv.org/pdf/2006.15704.pdf)
2. [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html#data-parallel-training)
3. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054.pdf)
4. [Mesh-TensorFlow: Deep Learning for Supercomputers](https://arxiv.org/pdf/1811.02084.pdf)
5. [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/pdf/1811.06965.pdf)
6. [PipeDream: Fast and Efficient Pipeline Parallel DNN Training](https://arxiv.org/pdf/1806.03377.pdf)