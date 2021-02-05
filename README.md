# Learning discriminative embeddings from multi-channel raw audio for semi-supervised machine condition monitoring

**ATTENTION: Unpublished work** This study is currently under-review

This repository containes the source code along with the detailed results and visualization of the research study "Learning discriminative embeddings from multi-channel raw audio for semi-supervised machine condition monitoring".

## Authors
Iordanis Thoidis, Marios Giouvanakis, George Papanikolaou. 

Laboratory of Electroacoustics and TV Systems,

School of Electrical and Computer Engineering, 

Faculty of Engineering, 

Aristotle University of Thessaloniki, 

Greece

## Abstract

In this study, we aim to learn highly descriptive representations for a wide set of machine sounds and use this knowledge to perform condition monitoring of individual machines. We propose a comprehensive feature learning approach from raw audio based on deep convolutional neural networks, that aims to leverage unrelated information from other sources to form distinct clusters for each machine.  A deep one-class neural network is then trained based on the Deep Support Vector Data Description objective to detect anomalies in different operational states of each machine. The effectiveness of both single- and multi-channel approaches are investigated. Additionally, we incorporate spatial invariance in the multi-channel convolutional neural network by exploring a front-end strategy for circular microphone arrays. Experimental results on the MIMII dataset demonstrate the effectiveness of the proposed method, reaching a mean AUC score of 91.0\%.  The anomaly detection performance is significantly improved by involving multi-channel audio data in the embedding extraction process, as well as training the convolutional neural network on the spatial invariance front-end. Finally, the proposed semi-supervised approach allows the concise modelling of normal machine conditions and accurately detects system anomalies, compared to existing anomaly detection methods.

## Dataset Distribution

![plot](./dataset_info.png)

## Discriminative embeddings on MIMII samples

![plot](./images/img.gif)

## Spatial filtering patterns of the of the 1st convolutional layer of RawdNet

![plot](./Figure_4.pdf)




