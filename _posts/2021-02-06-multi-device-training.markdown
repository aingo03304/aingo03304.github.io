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

