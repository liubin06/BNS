# Bayesian Negative Sampling

## Required Packages
- numpy
- pytorch
>Implement [BNS](https://doi.org/10.48550/arXiv.2204.06520) for Matrix Factorization (MF) (run main_MF.py); <br>
>Implement [BNS](https://doi.org/10.48550/arXiv.2204.06520) for light Graph Convolution Network (LightGCN) (Xiangnan et al. 2020) (run main_lightGCN.py).

## About BNS Framework:
We summarize the proposed negative sampling algorithm as: for each negative instances in the samll candidate set $\mathcal{M}_u$:

### Computing prior probability
Since  $pop_l \sim B (N, P_{fn}(l))$,  so we computing prior probability by:
[\P_{fn}(l) = \frac{pop_l}{N}]\
This step needs $\mathcal{O}(1)$.


### Computing  empirical distribution function (likelyhood) 
$$ F_{n}(x_l)= \frac{1}{n}\sum_l I_{|X_\cdot \leq \hat{x}_l|}$$
This step needs $\mathcal{O}(|\mathcal{I}|)$.
  
The calculation of empirical distribution function is easy to implement, which  converges to common cumulative distribution function $F(x)$ almost surely by the strong law of large numbers. *Glivenko Theorem* (1933) strengthened this result by proving uniform convergence of $F_n(\cdot)$ to $F(\cdot)$. This property of $F_n(\cdot)$ makes it possible for us to compute $F(\cdot)$ for the abstract distribution. <br>

A new understanding of $F(\hat{x}_l)$: $F(\hat{x}_l)$ describes the joint probability of the observed instance $\hat{x}_l$ as a function of the parameters of the ranking model. For the specific parameter $l \in fn$, $F(\hat{x}_l)$ assigns a probabilistic prediction valued in $[0,1]$ of $l$ being false negative (positive).

### Computing  negative signal (posterior probability) 
$$\mathtt{unbias}(l) = \frac{  [1 - F(\hat{x}_l)][1-P_{fn}(l)] }{1 - F(\hat{x}_l) -P_{fn}(l) + 2F(\hat{x}_l)P_{fn}(l) }$$
Note $\mathtt{unbias}(j)$ is an unbiased estimator for $l$ being true negative (see Lemma 3.1). This step needs $\mathcal{O}(1)$.<br>

### Performing Bayesian negative sampling
$$ j   &=&   \mathop{\arg\min}\limits_{l \in\mathcal{M}_u}~ \mathtt{info}(l)\cdot [1-(1+\lambda)\mathtt{unbias}(l)]$$
If the sampler $h^*$ minimizes the conditional sampling risk $R(l|i)$, then the empirical sampling risk will be minimized (see Theorem 3.1). Thus the Bayesian optimal sampling rule is essentially sampling the instance with minimal conditional sampling risk.This step needs $\mathcal{O}(1)$. <br>

More details about BNS(Bin Liu & Bang Wang, 2022) see our paper or poster at https://doi.org/10.48550/arXiv.2204.06520 .

## Dataset describe: 
>MovieLens100K; More details about MovieLens datasets at https://grouplens.org/datasets/movielens .<br>

>Our Bayesian Negative Sampling algothrim does not depend on the datasets. Just split the dataset you are interested in into the training set and test set, replace the train.csv and test.csv files, you can run Bayesian negative sampling easily. You just need to make sure that you have chosen appropriate prior information for **modeling prior probability** $P_{TN}(\cdot)$ or $P_{FN}(\cdot)$.

