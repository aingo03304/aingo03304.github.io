---
layout: post
title: "Multi-Device Training (Draft)"
date: 2020-08-03 15:39:00 +0900
image_url: ""
mathjax: true
comments: true
---

# Introduction
There are several strategies used to train a deep learning model with multi devices. In order to train a model across multiple devices, deep learning frameworks provide some features for distributed training such as:  
1. Data Parallelism
2. Model Parallelism
3. Pipeline Parallelism

Each parallelism scheme has pros and cons and engineers should decide among these to efficiently exploit their devices.

# Data Parallelism
Data Parallelism is well-known distributed method for training deep learning model. The notion of data parallelism is not only in deep learning domain but in plenty of domains. [SIMD](https://en.wikipedia.org/wiki/SIMD) instructions process multiple data simultaneously within one instruction, which is one of the data parallelism. Also, [SPMD](https://en.wikipedia.org/wiki/SPMD) programming model supports engineers to effectively do parallel programming. Data parallelism with multiple devices means that the task is splited into subtasks and each device conducts a subtask. For example, with (256, 32, 32, 3)-shaped input and 4 GPUs, it is easy to divide input into 4 (64, 32, 32, 3)-shaped inputs  because there is no dependence among batch axis in common deep learning task. 

Of course, layers like Batch Normalization have to be synchronized so that means and variances are the same across multiple devices. We will going to talk about this later.

# Model Parallelism

# Pipeline Parallelism

# Collective Commnuication
## Frameworks
## In Data Parallelism
## In Model Parallelism
## In Pipeline Parallelism

# Frameworks for Parallelism
## Tensorflow
## PyTorch
## DeepSpeed

# Conclusion