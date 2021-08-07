# baylib C++ library
<p align="center">
 <img alt="C++" src="https://img.shields.io/badge/cmake-v3.14.0-green"/>
 <img alt="developement" src="https://img.shields.io/badge/C++-17 | 20-blue.svg?style=flat&logo=c%2B%2B"/> 
</p>

Baylib is a simple inference engine library for Bayesian networks developed as final project for System Programming class at PoliTO.
The engine supports approximate inference algorithms.

Here's a list of the main requested features:
* Copy-On-Write semantics for the graph data structure, including the conditional probability table (CPT) of each node 
* parallel implementation of the algorithm 
* template-based classes for probability format
* input and output compatible with the [XDSL format](https://support.bayesfusion.com/docs/) prived by the SMILE library
* cmake-based deployment

## Abstract
A Bayesian Network (BN) is a probabilistic graphical model for representing knowledge about an uncertain domain where each node corresponds to a random random_variable and each edge represents the conditional probability for the corresponding random variables.

The graphical model measures the conditional dependence structure of a set of random variables based on the [Bayes theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem).


The problem requires to model the casual dependency between the variables and to build a direct, acyclic graph. Besides, it requires to identify for each random_variable a subset of states and probabilities.  

Once the engine produces the graph, inference algorithms process the conditional probabilities traversing the graph.

## The Algorithm
### Brief description
Logic Sampling generates possible scenarios with a random number generator and calculates a realization of all the network. After computing enough scenarios, the queries are answered with a frequentist approach and simple probability formulas. We integrate evidence by adding a validation step where we simply throw out scenarios that are not consistent with the evidence

**NOTICE:** When dealing with very low probabilities, the algorithm might introduce serious errors as certain cases might be ignored.

Moreover, the more evidence we add, the worst the model performs.
### algorithmic steps

1. define `s` as number of samples
2. acquire the graph
3. apply the function rank to the graph (rank the nodes using the longest path starting from a virtual source and assuming all edges with unitary weight)
4. initialize the thread pool
5. define a task as the simulation of a single node `n` given the results of the preceding nodes, i.e. `n` tasks with `n` corresponding to the number of nodes in the graph
6. send tasks to the thread pool in rank order
7. for each `n` node: 
   - wait for `parent(n)`
   - generate random vector `V` of floats with uniform distribution (0,1) of length `s`
   - for each element of index `i` :
     * find threshold `t` from probability table using the `i-th` element of parents results
     * transform into bool using `t`
   - end for
   - send boolean vector to `child(n)`
   - `tot = sum(V)`
   - `P(n) = tot/s`
8. end.

## Deployment

## External references

* [copy-on-write](https://www.codeproject.com/Tips/5261583/Cplusplus-Copy-On-Write-Base-Class-Template)
* [thread pool](https://github.com/bshoshany/thread-pool)
