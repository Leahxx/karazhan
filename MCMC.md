# Markov Chain & Monte Carlo

### Markov Chain:

$$
p(q_t | q_{t-1}, q_{t-2} ... q_1) = p(q_t | q_{t-1})
$$



- 采样的动机：求和或求平均

- 什么是好的采样：

  样本趋向高概率区域

  样本之间独立

- 采样为什么困难：partation function is intractable



- Metropolis Hasting Sampling:

To find detailed balance, equals to find trainsition matrix $P \sim \{P_{ij}\}$

To find $P$, propose a matrix $Q \sim \{Q_{ij}\}$ such that: 
$$
p(x) Q(z^*|z) \alpha(z, z^*) = p(z^*) Q(z|z^*) \alpha(z^*, z)
$$
Where the $\alpha$ is the acceptance rate:
$$
\alpha (z, z^*) = min(1, \frac{p(z^*) Q(z|z^*)}{p(z)Q(z^*|z)})
$$

- Gibbs sampling

For high dimension vector $z$, sample each dimentsion iteratively. 

$\alpha = 1$ and transition matrix is $p(z)$ itself. 



# Hidden Markov Model

- Transition probability:  $P(q_t | q_{t-1})$  where $q_t$ denotes hidden status at t moment, $q$ must be **discrete**

- Emmision/Measure probabiliyu: $P(y_t | q_t)$ 

  

  

  





