#Bayesian Negative Sampling

## Required Packages
- numpy
- pytorch
>Implement BNS(Bin Liu & Bang Wang, 2022) for matrix factorization (MF) (run main_MF.py); Implement BNS(Bin Liu & Bang Wang, 2021) forlight graph neural
network (LightGCN) (Xiangnan et al. 2020) (run main_lightGCN.py).

## Dataset describe: 
>MovieLens100K; More details about MovieLens datasets at https://grouplens.org/datasets/movielens .
>Our Bayesian Negative Sampling algothrim does not depend on the datasets. Just split the dataset you are interested in into the training set and test set, replace the train.csv and test.csv files, you can run Bayesian negative sampling easily. You just need to make sure that you have chosen appropriate prior information for **modeling prior probability** $P_{TN}(\cdot)$ or $P_{FN}(\cdot)$.

## About MPR Framework:
More details about MPR see our paper or poster at https://doi.org/10.48550/arXiv.2204.06520 .
