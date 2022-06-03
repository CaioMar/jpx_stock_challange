# Portifolios 

## Why diversify

## Characterizing a portifolio

* **Weight Vector**: the weight $w$ is a vector where each element corresponds to the weight of the asset in the portifolio;
* **Return of the portifolio**: represented by the symbol $R_p$;
* **Mean return of a portifolio**: represented as the expectation of the return of the portifolio $E[R_p]=\mu_p=w^T\mu$, where $\mu_p$ is the **mean return of the portifolio** and $\mu$ is a vector with the mean return of each security that is being assessed;
* **Variance of a portifolio**: the variance of a portifolio, a measure of its risk (we measure the risk as being the standard deviation of a portifolio), is related to the covariance matrix between all assets of the portifolio $\Sigma$ and the set of weights $w$. It is given by the following formula:
$$\sigma_p^2=w^T\Sigma w$$
* **Risk-Return Plot**: It is a plot that can be used to compare a multitude of portifolios. Each point the plot represents a portifolio, where the y-axis correspond to $\sigma_p$ and the x-axis represents $\mu_p$. A variety of portifolios can be simulated by changing the weight vector, for instance. When plotted this multitude of portifolio form what is known as the [Markowitz Bullet](https://miro.medium.com/max/737/1*qQMmz8u_9xWS5uLiEvV44Q.png).



## Maximum return of a portifolio of assets

$$\mu_{p} = \mu^Tw$$

This can be characterized as a constrained optimization problem. Give the following:

$$\max_{w}{\mu^Tw} \newline subject\space to\space 1_{D}^Tw=1$$
This problem still has a maximum that goes towards infinity, since we could theoretically short on one of the assets that have negative mean return, while investing infinitely on the mean positive return asset. In practice, under realistic scenarios we have additional constraints that reduce the set of solutions for this problem, eliminating the infinite return portifolio solutions. For example, maybe we are not allowed to invest more than 30% of our wealth on any single asset, or maybe we are not allowed to short sell any single asset. Take the following more realistic constraint optimization problem:
$$\max_{w}{\mu^Tw} \newline subject\space to\space 1_{D}^Tw=1 \newline w_i \geq0$$
We could solve this with Linear Programming (LP). We can solve this kind of problem with Scipy or ORTools libraries in Python, for instance.

## Mean-Variance Optimization

There is an intrinsic relationship between return and risk - take the Markowitz bullet for instance, it showcases a multitude of possible portifolios that have different values of risk and return (there is an inherit trade-off between risk and return).

### Efficient frontier

#### Ways of finding the efficient frontier

* Minimizing Variance, subject to return >= min return (Target Return)
* Maximizing return, subject to variance <= max variance (Target Risk)
* Maximize return - risk-aversion constant * variance (Risk Aversion)

## Global Minimum Variance (GMV) Portifolio
In order to obtain the portifolio that represent the tip of the Markowitz bullet, we just have to solve the following QP problem:
$$\min_{w}{w^T\Sigma w} \newline subject\space to\space 1_{D}^Tw=1 \newline w_i \geq0$$

## Sharpe Ratio

Defined by William Shape (winner of the economy Nobel Prize), the sharpe ratio is a quantity that allows the investor to compare two different portifolios in the efficient frontier using a single quantity that encompasses both risk and return. The Sharpe ratio $SR$ is defined as:

$$SR=\frac{\mu_p-r_f}{\sigma_p}$$

where $r_f$ is a risk free asset.

### Maximum Sharpe Ratio of a Portifolio

### Tangency Portifolio

The tangency portifolio is a portifolio that includes a risk free asset.

## Capital Asset Pricing Model (CAPM)

