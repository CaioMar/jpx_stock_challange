# Hidden Markov Models

Hidden Markov Models (HMMs) are commonly employed when modelling sequence data where the observed data $x_t$ at a time step $t$ depends *only* on a *hidden* or *latent* variable $z_t$. Another point worth mentioning is that HMMs do follow the Markov Property, meaning that $z_t$ depends only on the previous hidden state $z_{t-1}$. A possible graphical representation of HMMs, that illustrate the Markov is given in in the following [link](https://www.researchgate.net/profile/Jan-Bulla-2/publication/24115579/figure/fig2/AS:669552555872262@1536645177600/Basic-structure-of-a-Hidden-Markov-Model.png) (note that in the link $z_t$ is $S_t$).

A possible use case of HMMs when modeling stock returns is *volatility clustering*. In this scenario, the random variable $z$ represents the volatility regime (either *high* or *low* volatility), and each regime is associated with a different distribution of stock returns, one with higher and another one with lower variance.

## Steps

* **Determining $p(x)$**: finding $p(x)$ can be achieved by finding the *joint* distribution between $x$ and $z$, $p(x,z)$ and then marginalizing;
* **Maximum Likelihood Estimation:** achieved either using Gradient Ascent or Expectation-Maximization, the idea is to find the distribution parameters (for instante, the *mean* and *variance* of each guassian distribution), that most likely generated the data;
* **Decoding** $\rightarrow$ obtaining sequence of hidden variables $z(t)$ that most likely generated the sequence of *observed* data $\vec{x}$. Possible through the use of the Viterbi algorithm for instance (just as a side node, Viterbi a the founder of Qualcomm). When using the library *hmmlearn*, we can perform decoding using the *predict* method.