# Bayesian Negative Sampling

## Required Packages
- numpy  : Implement [BNS](https://doi.org/10.48550/arXiv.2204.06520) for Matrix Factorization (MF) (run main_MF.py); 
- pytorch: Implement [BNS](https://doi.org/10.48550/arXiv.2204.06520) for light Graph Convolution Network (LightGCN) (run main_lightGCN.py).

## About BNS Framework:
We summarize the proposed negative sampling algorithm as: Randomly select a small set of candidates $\mathcal{M}_u$ from the negative instances set $\mathcal{I}_u^-$, for each negative instances $l$ in the samll candidate set $\mathcal{M}_u$:

>### (i) Computing prior probability
>>By assuming $pop_l \sim B (N, P_{fn}(l))$, we compute prior probability as:
$$P_{fn}(l) = \frac{pop_l}{N}$$
>>This step needs $\mathcal{O}(1)$.


>### (ii) Computing cumulative distribution function (likelihood) 
$$ F_{n}(\hat{x} _ l) = \frac{1}{n} \sum  I_ {|x_ \cdot \leq \hat{x}_l |} $$
>>This step needs $\mathcal{O}(|\mathcal{I}|)$.
  
>>The empirical distribution function $F_n (\cdot)$  converges to common cumulative distribution function $F(\cdot)$ almost surely by the **strong law of large numbers**. **Glivenko Theorem** (1933) strengthened this result by proving **uniform convergence** of $F_n(\cdot)$ to $F(\cdot)$. This property makes it possible for us to compute the $F(\cdot)$, even if we do not know its explicit expression. Given the observation $\hat{x}_l$, $F(\hat{x}_l)$ describes the joint probability of the observed instance $\hat{x}_l$ as a function of the parameters of the ranking model. For the specific parameter $l \in fn$, $F(\hat{x}_l)$ assigns a probabilistic prediction valued in $[0,1]$ of $l$ being false negative (positive).<br>

>### (iii) Computing  negative signal (posterior probability) 
$$ \mathtt{unbias}(l) = \frac{[1 - F(\hat{x} _ l)] [1-P _ {fn} (l)] }{1 - F(\hat{x}_ l) -P_{fn}(l) + 2F(\hat{x}_ l)P_{fn}(l)} $$!

<div align=center>
<img src="https://github.com/liubin06/test/raw/master/fig3.png">
</div>

>>Note $\mathtt{unbias}(j)$ is an unbiased estimator for $l$ being true negative (see Lemma 3.1). This step needs $\mathcal{O}(1)$.

>### (iv) Performing Bayesian negative sampling
$$ j  =  \mathop{\arg\min}\limits_{l \in\mathcal{M}_u}~ \mathtt{info}(l)\cdot [1-(1+\lambda)\mathtt{unbias}(l)]$$
>>If the sampler $h^*$ minimizes the conditional sampling risk $R(l|i)$, then the empirical sampling risk will be minimized (see Theorem 3.1). Thus the Bayesian optimal sampling rule is essentially sampling the instance with minimal conditional sampling risk.This step needs $\mathcal{O}(1)$. <br>

More details about BNS(Bin Liu & Bang Wang, 2022) see our paper or poster at https://doi.org/10.48550/arXiv.2204.06520 .

## Distribution Analysis
>On the basis of order relation analysis of negatives' scores, we derive the class conditional density of true negatives and that of false negatives, and provide an affirmative answer from a Bayesian viewpoint to distinguish true negatives from false negatives. 

> Real distribution
<div align=center>
<img src="https://github.com/liubin06/test/raw/master/fig1.png">
</div>

> Theoretical distribution
> <div align=center>
<img src="https://github.com/liubin06/test/raw/master/fig2.png">
</div>

## Dataset Description: 
>MovieLens100K; More details about MovieLens datasets at https://grouplens.org/datasets/movielens .<br>

>Our Bayesian Negative Sampling algothrim does not depend on the datasets. Just split the dataset you are interested in into the training set and test set, replace the train.csv and test.csv files, you can run Bayesian negative sampling easily. You just need to make sure that you have chosen appropriate prior information for **modeling prior probability** $P_{tn}(\cdot)$ or $P_{fn}(\cdot)$.
