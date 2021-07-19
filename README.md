# Bayesian Inferrer
<p align="center">
 <img alt="C++" src="https://img.shields.io/badge/cmake-v3.14.0-green"/>
 <img alt="developement" src="https://img.shields.io/badge/C++-17 | 20-blue.svg?style=flat&logo=c%2B%2B"/> 
</p>

Bayesian Inferrer is a simple inference engine library for Bayesian networks based on the [logic sampling algorithm](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9b660beb-e839-4ee7-a30e-5bd6a12e56de/henrion1988.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210719%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210719T102930Z&X-Amz-Expires=86400&X-Amz-Signature=e2feeb58bb144451357355bccbcd2798e3a32a74d25f39a3432b10ed0efab4bc&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22henrion1988.pdf%22) characterized by the following features:
* Copy-On-Write semantics for the graph data structure, including the conditional probability table (CPT) of each node 
* parallel implementation of the algorithm 
* template-based classes 
* allocation policy based on `std::pmr::polymorphic_allocator`
* input and output compatible with the [XDSL format](https://support.bayesfusion.com/docs/) prived by the SMILE library
* cmake-based deployment

## Abstract
A Bayesian Network (BN) is a probabilistic graphical model for representing knowledge about an uncertain domain where each node corresponds to a random variable and each edge represents the conditional probability for the corresponding random variables.

The graphical model measures the conditional dependence structure of a set of random variables based on the [Bayes theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).


The problem requires to model the casual dependency between the variables and to build a direct, acyclic graph. Besides, it requires to identify for each variable a subset of states and probabilities.  

Once the engine produces the graph, inference algorithms process the conditional probabilities traversing the graph.
 
## The Algorithm
Logic Sampling generates possible scenarios with a random number generator and calculates a realization of all the network. After computing enough scenarios, the queries are answered with a frequentist approach and simple probability formulas. We integrate evidence by adding a validation step where we simply throw out scenarios that are not consistent with the evidence

**NOTICE:** When dealing with very low probabilities, the algorithm might introduce serious errors as certain cases might be ignored.

Moreover, the more evidence we add, the worst the model performs.

## Deployment


