# Bayesian Mixture Modeling with Wasserstein Distance

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

## For an exhaustive explanation + t-Student model we suggest to refer to the Report.

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

Alongside of our effort to program this Repulsive Mixture Model we also pursued a parallel objective to implement all the models in a computationally efficient way, relying on the Python High Performance Computing library JAX. 
Thanks to the Just-in-Time code compilation feature, we are able to compile the code and obtain much faster computations. For example, before the optimization a typical run of 4000 total iterations would take 60 minutes. 
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

### Model

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/154938235-873dba76-ce98-4848-9aaf-1ba39dfd4a68.png" width="300" alt="Scenario"/>
</p>

Moreover, we set H = 15 as an upper bound for the number of components. This is justified by the fact that the datasets considered have a low true number of components, and so we do not want the algorithm to generate too many clusters.
In order to sample from the posterior distributions we use Gibbs sampling, computing the full conditionals of the model.

### Results

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/154939543-6f4e6fd7-e77a-43af-87d9-193c4775afad.png" width="300" alt="Scenario"/>
    <img src="https://user-images.githubusercontent.com/91596609/154939550-1976c02b-d891-4f29-98f2-7cdecb493a89.png" width="300" alt="Scenario"/>
</p>
<p align="center">
    Frequencies of estimated clusters during the iterations and best clustering w.r.t. Binder’s loss on simulated dataset
</p>

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/154939560-733c4eb6-9f5d-4082-90f3-ee907a62524a.png" width="300" alt="Scenario"/>
    <img src="https://user-images.githubusercontent.com/91596609/154939571-d4601939-5109-4ca1-9519-218e690b6698.png" width="300" alt="Scenario"/>
</p>
<p align="center">
    Frequencies of estimated clusters during the iterations and best clustering w.r.t. Binder’s loss on Old Faithful dataset
</p>

## Gaussian Mixture Models with the Wasserstein distance

In the case of location-scale family of distribution we have a closed form for the Wasserstein distance, which is
<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/155035482-2371dbcb-b23e-4224-b31c-73cf7abfb685.png" width="500" alt="Scenario"/>
</p>
where μX and μY denote the means of the two Gaussians, while ΣX and ΣY denote the corresponding covariance matrices.
The idea at this point is to employ the Wasserstein distance inside our Mixture Model in the prior for (μh,Σh) in order to penalize all the configuration of (μh,Σh) that are placed too close to each other.
In this way, whenever a distribution is not Gaussian distributed, even if the Gaussian Mixture Model will try to fit more than one Gaussian distribution, the penalization term will favour the spreading of the distributions, avoiding the aforementioned problem.
The penalized prior becomes the following:
<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/155035786-16501426-24f8-44ba-a85b-a4b840aac816.png" width="650" alt="Scenario"/>
</p>

In order to sample from the posterior distributions we still rely on the Gibbs Sampler we used for the standard Mixture Model. 
The problem, in this case, is that we lost conjugacy for the full conditional (μh, Σh)|(μ, Σ)−h, c, y, and hence we cannot simulate directly from its posterior. 
Instead, we have to build a Metropolis-within-Gibbs step.
In particular, the full conditionals read:
<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/155035965-2b495d1c-566b-4664-a7b2-9ab4325aec70.png" width="750" alt="Scenario"/>
</p>

In order to perform this simulation step we have to choose a proper proposal distribution to sample from. Since we want to use a random-walk Metropolis Hastings algorithm, the idea is to center our proposal distribution on the previous values of mean and covariance.

Since using a Normal distribution to sample μ∗ provides some good traceplots, we would like to use aNormal also to sample Σ∗ . Notice that the precision parameter is distributed over positive semi-definite matrices, therefore we have to maintain this constrain throughout some suitable reparametrization. To achieve such a sampling strategy, we use a bijector B between the space of d×d symmetric, positive definite matrices and the space of d(d + 1)/2 vectors.
We can therefore perform a change of variables with respect to the covariance matrix Σh as follows:
<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/155036221-78dbd195-7666-4586-935f-f4b26acb23cb.png" width="600" alt="Scenario"/>
</p>
In this way, any covariance matrix can be sampled using a Multivariate Normal centered on the vectorized covariance matrix itself.
With this new approach the algorithm eventually worked and we implemented two more refined proposal methods: 

- Shoot Away proposal
- MALA proposal

### Results

In our case studies, MALA proposal brought improvements to the Metropolis step. The traceplots show much more mixing of the chain. The overall performance of the clustering does not change significantly, however the state exploration is much more exhaustive. This is extremely important and helps to avoid the problem of a falling acceptance rate we experienced with other proposals.

#### Dataset 1
<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/155036920-f73dbfc1-5caf-4a72-a369-f5af23c7b753.png" width="700" alt="Scenario"/>
</p>

#### Dataset 2
<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/155037007-45117c0b-9cff-4b3d-b928-5f18395cd37f.png" width="700" alt="Scenario"/>
</p>

## Team

- Manuel Bressan [[Github](https://github.com/manubre98)] 
- Leonardo Perelli [[Github](https://github.com/LeoPerelli)]
- Federico Ravanetti
- Sebastiano Rossi [[Github](https://github.com/Seb1198)]
- Edward Wiels
- Daniela Zanotti [[Github](https://github.com/DanielaZanotti)] 

## Credits

This project is under the supervision of PHD Researcher <a href="https://github.com/mberaha">Mario Beraha</a> from Politecnico di Milano. 


