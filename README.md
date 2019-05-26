# Deep Dominance - How to properly compare deep neural models

This repository contains an implementation of a method for comparing between two deep neural models described in (Dror et al., 2019):

"Deep Dominance - How to Properly Compare Deep Neural Models." Rotem Dror, Segev Shlomov and Roi Reichart. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL2019).

Given two algorithms, each associated with a set of test-set scores, our goal is to determine which algorithm, 
if any, is superior. The score distributions are generated when running two different DNNs with various 
hyperparameter configurations and random seeds. This code implements a method for comparing between 
two score distributions based on a measure of "almost stochastic dominance".

Details about the implementation and theoretical justifications is described in our paper: Deep Dominance - How to properly compare deep neural models.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The implementation is implemented in Python 3.6.

### Running the Code

To run the code you need to run the following command line:

```
python ASD.py filename_AlgA filename_AlgA alpha 
```
This script will read the first file of scores from algorithm A from `filename_AlgA` and the scores from algorithm B from 
`filename_AlgB`. The last input to the script is the desired significance level of the statistical test which should be 
entered instead of `alpha`. 

An example run:

```
python ASD.py ./scores/scoresA ./scores/scoresB 0.05 
```

### Input Files

The input consists of two files with the results of applying each algorithm (A and B) on a dataset. The results for each algorithm should be in the following form (the result for each sample in X separated by lines) :

```
46.1726
68.5210
51.1151
45.8590
55.2119
36.5653
37.4119
39.8117
51.7002
```


### Output

There are 2 possible outputs for the script. The output form depends on whether algorithm A is better than B or otherwise.

If algorithm A is better than algorithm B according to the test then the output will be of the form:
```
The minimal epsilon for which Algorithm A is almost stochastically greater than algorithm B is _____
since epsilon <= 0.5 we will claim that A is better than B with significance level alpha= ______
```

If algorithm B is better than algorithm A according to the test then the output will be of the form:
```
The minimal epsilon for which Algorithm A is almost stochastically greater than algorithm B is _____
since epsilon > 0.5 we will claim that A is not better than B with significance level alpha= ______
```
For more details about the meaning of the output please read our paper: Deep Dominance - How to properly compare deep neural models.

### Citation
If you make use of this code for research purposes, we'll appreciate citing the following:
```
@InProceedings{P,
  author = 	"Dror, Rotem
		and Shlomov, Segev
		and Reichart, Roi",
  title = 	"Deep Dominance - How to Properly Compare Deep Neural Models",
  booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2019",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"",
  location = 	"Florence, Italy",
  url = 	""
}
```




