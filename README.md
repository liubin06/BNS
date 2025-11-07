[![Academic Paper](https://img.shields.io/badge/IEEE_ICDE-2023-important)](https://doi.org/10.1109/ICDE55515.2023.00063)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICDE55515.2023.00063-royalblue)](https://doi.org/10.1109/ICDE55515.2023.00063)

# Bayesian Negative Sampling

Official implementation for the paper accepted at [**IEEE ICDE 2023**](https://ieeexplore.ieee.org/document/10184562). 

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{Bin:ICDE:2023,
      title={Bayesian Negative Sampling for Recommendation}, 
      author={Liu Bin and Wang Bang},
      booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
      year={2023},
      pages={749-761},
      doi={10.1109/ICDE55515.2023.00063}
}
```

## 🛠️ Installation

- **numpy**: For Matrix Factorization implementation ([`main_MF.py`](https://github.com/liubin06/BNS/blob/main/BNS_MF/main_MF.py))
- **pytorch**: For LightGCN implementation ([`main_lightGCN.py`](https://github.com/liubin06/BNS/blob/main/BNS_lightGCN/main_lightGCN.py))

## 🎯 Algorithm Overview

The Bayesian Negative Sampling framework operates as follows:

### 1. Candidate Selection
Randomly select candidate set $\mathcal{M}_u$ from negative instances.

### 2. Prior Probability Computation
Assume $pop_l \sim B (N, P_{fn}(l))$, compute:
$$P_{fn}(l) = \frac{pop_l}{N}$$
Time complexity: $\mathcal{O}(1)$

### 3. Likelihood Estimation
Compute empirical CDF:
$F_{n}(\hat{x} _ l) = \frac{1}{n} \sum  I_ {|x_ \cdot \leq \hat{x}_l |}$

Time complexity: $\mathcal{O}(|I|)$

> 📊 **Theoretical Foundation**: By the **Glivenko Theorem** (1933), $F_n(\cdot)$ uniformly converges to $F(\cdot)$, enabling probabilistic prediction of false negatives.

### 4. Posterior Calculation
Compute unbiased estimator:
$\mathtt{unbias}(l) = \frac{[1 - F(\hat{x} _ l)] [1-P _ {fn} (l)] }{1 - F(\hat{x}_ l) -P_{fn}(l) + 2F(\hat{x}_ l)P_{fn}(l)}$

Time complexity: $\mathcal{O}(1)$

### 5. Bayesian Sampling
Select instance minimizing:
$j  =  \mathop{\arg\min}\limits_{l \in\mathcal{M}_u}~ \mathtt{info}(l)\cdot [1-(1+\lambda)\mathtt{unbias}(l)]$

Time complexity: $\mathcal{O}(1)$


## 📊 Datasets

- **MovieLens100K** & **MovieLens1M**: [GroupLens](https://grouplens.org/datasets/movielens)
- **Yahoo!-R3**: [Yahoo Webscope](http://webscope.sandbox.yahoo.com/catalog.php?datatype=r)

> 💡 **Flexibility**: BNS is dataset-agnostic. Simply replace `train.csv` and `test.csv` files, ensuring appropriate prior probability modeling.

## 📬 Contact

For questions, please contact:
- **Bin Liu**: binliu@swjtu.edu.cn
