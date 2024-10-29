# Deep Generative Models

## Chapter2



### Notions:  conditional independence

- **Two events** A, B are conditionally independent given event C if: 
  $$
  p(A \cap B|C) = p(A|C)p(B|C)
  $$

- **Random variables** $X, Y$ are  conditionally independent given Z if for all values $x \in Val(X), y \in Val(Y), z \in Val(Z)$:
  $$
  p(X, Y|Z) = p(X|Z)p(Y|Z)
  $$
  also write 
  $$
  p(X|Y,Z) = p(X|Z) \\
  X \perp Y | Z
  $$

- **Chain rule**  Let $S_1, ... S_n$ be events, $p(S_i) > 0$
  $$
  p(S_1 \cap S_2 \cap ... \cap S_n) = p(S_1)p(S_2|S_1)...p(S_n|S1 \cap ...\cap S_{n-1})
  $$
  
- **Bayes' rule** Let $S_1, S_2$ be events, $p(S_1) > 0$ and $p(S_2) > 0$. 
  $$
  p(S_1|S_2) = \frac{p(S_2|S_1)p(S_1)}{p(S_2)}
  $$
  
- **Probability density** $p(x)$ 
  $$
  p(x) \ge 0 \\
  \int_{\infty}^{-\infty} p(x) dx = 1
  $$

- Discrete variable $x$ , **expectation** is the average value of some function $f(x)$ under a probability distribution $p(x)$
  $$
  E[f] = \sum_x p(x)f(x)
  $$
  if x is continuous variable, expectation is
  $$
  E[f] = \int p(x)f(x) dx
  $$
  conditional expectation: 
  $$
  E_x[f|y] = \int_x p(x|y)f(x)
  $$
  
  

- Kullback-Leibler divergence or relative entropy between distribution $p(x)$ and $q(x)$:
  $$
  KL(p||q) = - \int p(x)log(\frac{q(x)}{p(x)}) dx
  $$
   



