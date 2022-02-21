# Bayesian Mixture Modeling with Wasserstein Distance

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

## Overview

Mixture Models typically find application in density estimation and clustering problems.
Conventional Bayesian posterior inference on cluster-specific parameters struggles when clusters are placed too close to each other.

As an alternative, Repulsive Mixture Models generate the components from a repulsive process that naturally favours separation of clusters.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/154936048-3440ad25-d609-4dee-b460-a3bbd28d3262.png" width="400" alt="Scenario"/>
    <img src="https://user-images.githubusercontent.com/91596609/154936064-91982abe-a572-495d-8c5e-f772376e78e8.png" width="400" alt="Scenario"/>
</p>

Mixture Model is a widespread probabilistic model that finds application in numerous different areas from density estimation, pattern identification, feature selection to clustering and many other more. Many times sets of independent and identically distributed observations cannot be described by a single distribution, but a combination of a small number of distributions belonging to the same parametric family is needed. In our report we will focus on the clustering problem.

In this report we want to address this problem by introducing a penalization term for the joint prior of the component-specific parameters, i.e. mean μ and shape Σ of each cluster. This penalization is based on the sum of Wasserstein distances among all the possible couple (μh,Σh). 
This metric is a sensible choice because it accounts for the similarity between two distribution in terms of location but also in terms of shape. In this way the Wasserstein distance can be large even if the distributions are centered in the same point because they can have different shape. This new model is then called Repulsive Mixture Model due to its behaviour.

Alongside of our effort to program this Repulsive Mixture Model we also pursued a parallel objective to implement all the models in a computationally efficient way, relying on the Python High Performance Computing library JAX. Thanks to the Just-in-Time code compilation feature, we are able to compile the code and obtain much faster computations. For example, before the optimization a typical run of 4000 total iterations would take 60 minutes. 
After optimizing the code, this takes just 20 seconds, yielding a speed up of around 180 times.

## Test Datasets

To evaluate and analyse the performance of the models we develop, we decide to fix two datasets to use as benchmark. One dataset is the so called ’Old Faithful Geyser’ dataset. 
The other test dataset is generated fictitiously, through perturbation of Normal Distributions. The starting point is a 4-component Normal Distribution with equal weights. The distributions are equally weighted and all have covariance matrix equal to the identity matrix.
The means of the components are: [0, 0], [0, 4], [4, 0], [4, 4].
<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/154937345-8f96048c-c66a-4fbf-81d9-b290ec8db4b4.png" width="400" alt="Scenario"/>
    <img src="https://user-images.githubusercontent.com/91596609/154937363-d39a87c6-b9f6-40b0-92da-c17ce581ae18.png" width="400" alt="Scenario"/>
</p>

## Standard Gaussian Mixture Models

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/154938235-873dba76-ce98-4848-9aaf-1ba39dfd4a68.png" width="400" alt="Scenario"/>
</p>

## Credits

This project is under the supervision of PHD Researcher <a href="https://github.com/mberaha">Mario Beraha</a> from Politecnico di Milano. 


