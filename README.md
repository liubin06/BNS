# Bayesian Negative Sampling

This work is accepted for publication in [***IEEE International Conference on Data Engineering (ICDE 2023)***](https://ieeexplore.ieee.org/document/10184562). Should you use this work in your research, please cite the following paper:


```bibtex
@inproceedings{Bin:ICDE:2023,
      title={Bayesian Negative Sampling for Recommendation}, 
      author={Liu Bin and Wang Bang},
      booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
      year={2023},
      pages={749-761},
      doi={10.1109/ICDE55515.2023.00063}, 
}
```

## Required Packages
- numpy  : Implement BNS for Matrix Factorization (MF) (run [main_MF.py](https://github.com/liubin06/BNS/blob/main/BNS_MF/main_MF.py) ); 
- pytorch: Implement BNS for light Graph Convolution Network (LightGCN) (run [main_lightGCN.py](https://github.com/liubin06/BNS/blob/main/BNS_lightGCN/main_lightGCN.py) ).

## About BNS Framework:
We summarize the Bayesian negative sampling algorithm as: Randomly select a small set of candidates $\mathcal{M}_u$ from the negative instances set, for each negative instances $l \in \mathcal{M}_u$ do:

>### (i) Computing prior probability
>>By assuming $pop_l \sim B (N, P_{fn}(l))$, we compute prior probability as:
$$P_{fn}(l) = \frac{pop_l}{N}$$
>>This step needs $\mathcal{O}(1)$.


>### (ii) Computing cumulative distribution function (likelihood) 
$$ F_{n}(\hat{x} _ l) = \frac{1}{n} \sum  I_ {|x_ \cdot \leq \hat{x}_l |} $$
>>This step needs $\mathcal{O}(|\mathcal{I}|)$.
  
>>The empirical distribution function $F_n (\cdot)$  converges to common cumulative distribution function $F(\cdot)$ almost surely by the **strong law of large numbers**. **Glivenko Theorem** (1933) strengthened this result by proving **uniform convergence** of $F_n(\cdot)$ to $F(\cdot)$. This property makes it possible for us to compute the $F(\cdot)$, even if we do not know its explicit expression. 
>> Given the observation $\hat{x}$, $F(\hat{x})$  
>> assigns a probabilistic prediction valued in $[0,1]$ 
>> of $l$ being false negative (positive).<br>

>### (iii) Computing  negative signal (posterior probability) 
$$ \mathtt{unbias}(l) = \frac{[1 - F(\hat{x} _ l)] [1-P _ {fn} (l)] }{1 - F(\hat{x}_ l) -P_{fn}(l) + 2F(\hat{x}_ l)P_{fn}(l)} $$
>>Note that $\mathtt{unbias}(l)$ is an unbiased estimator for $l$ being true negative (see Lemma 3.1). 
>>This step needs $\mathcal{O}(1)$.


<div align=center>
<img src="https://github.com/liubin06/test/raw/master/fig3.png">
</div>

>### (iv) Performing Bayesian negative sampling
$$ j  =  \mathop{\arg\min}\limits_{l \in\mathcal{M}_u}~ \mathtt{info}(l)\cdot [1-(1+\lambda)\mathtt{unbias}(l)]$$

>> If the sampler $h^*$ minimizes the conditional sampling risk $R(l|i)$, then the empirical sampling risk will be minimized (see Theorem 3.1). 
>> Thus the Bayesian optimal sampling rule is essentially sampling the instance with minimal conditional sampling risk.
>> This step needs $\mathcal{O}(1)$. <br>

If you have any questions, please feel free to contact contact **"wangbang@hust.edu.cn; liubin0606@hust.edu.cn"**.
More details about BNS(Bin Liu & Bang Wang, 2022) see our paper or poster at https://doi.org/10.48550/arXiv.2204.06520 .

## Distribution Analysis
> The key is the class conditional density. On the basis of order relation analysis of negatives' scores, we derive the class conditional density of true negatives and that of false negatives, and provide an affirmative answer from a Bayesian viewpoint to distinguish true negatives from false negatives. 

**Real distribution**
<div align=center>
<img src="https://github.com/liubin06/test/raw/master/fig11.png">
</div>

**Theoretical distribution**
> <div align=center>
<img src="https://github.com/liubin06/test/raw/master/fig22.png">
</div>

> We exhibits the distribution morphology of false negatives and true negatives derived from the ordinal relation with different types of $f(x)$: Gaussian distribution $x \sim \mathcal{N}(\mu,\sigma)$ (symmetrical), student distribution $x \sim t(n)$ (symmetrical), and Gamma distribution $x\sim Ga(\alpha,\lambda)$ (asymmetrical) . As we can see, during the training process, the real distribution of true/false negatives gradually exhibit the same structure as theoretical distribution. Although the probability density function $f(x)$ changes during the training process, this separated structure is sufficient for us to classify the true negative instances from false negative instances. <br>

## Dataset Description: 
>MovieLens100K; MovieLens1M; Yahoo!-R3.<br>
- More details about MovieLens datasets at https://grouplens.org/datasets/movielens .<br>
- More details about Yahoo datasets at http://webscope.sandbox.yahoo.com/catalog.php?datatype=r .<br>

>Our Bayesian Negative Sampling algothrim does not depend on the datasets. Just split the dataset you are interested in into the training set and test set, replace the train.csv and test.csv files, you can run Bayesian negative sampling easily. You just need to make sure that you have chosen appropriate prior information for **modeling prior probability** $P_{tn}(\cdot)$ or $P_{fn}(\cdot)$.
