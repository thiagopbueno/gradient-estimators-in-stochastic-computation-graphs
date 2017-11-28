# gradient-estimators-in-stochastic-computation-graphs

Gradient Estimation in Stochastic Computation Graphs using *TensorFlow*.

In this IPython notebook, we aim at exploring different ways to **estimate gradients** in stochastic computation graphs in order to optimize an objective function defined by an expectation over a set of random variables.

In terms of implementation, we'll use the TensorFlow library, mainly because of its automatic reverse-mode differentiation capabilities.

The notation and theory are based on the following NIPS paper:

> Schulman, J., Heess, N., Weber, T. and Abbeel, P., 2015.<br>
> **[Gradient estimation using stochastic computation graphs](http://papers.nips.cc/paper/5899-gradient-estimation-using-stochastic-computation-graphs.pdf).**
> In Advances in Neural Information Processing Systems (pp. 3528-3536).
