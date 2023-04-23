# ML-stat-sig-tests
This repository provides statistical significance test implementations for comparing various Machine Learning models using a single dataset.

The existing implementations were specifically designed to work with certain types of estimators (e.g., sklearn estimators). Therefore, this repository shares implementations of these tests that can be used with any ML model by simply integrating the experiment results.

## Tests
Before applying the methods below, divide your dataset into two parts: a training set and a test set. For Combined 5x2CV tests, repeat the splitting process (50% training and 50% test data) five times.
Assuming we are comparing two estimators A and B, in each of the five iterations, we fit A and B to the training split and evaluate their performance on the test split. We then swap the training and test sets (the train set becomes the test set, and the test set becomes the train set) and calculate the performance again, resulting in two performance measurements.

### Combined 5x2CV F-test
This repository features a generalized implementation of the [Mlxtend library's implementation](https://rasbt.github.io/mlxtend/) for Sklearn estimators.

#### References
[1] Alpaydin, E. (1999). Combined 5×2 cv F test for comparing supervised classification learning algorithms. Neural computation, 11(8), 1885-1892.
[2] Dietterich TG (1998) Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Comput 10:1895–1923.

### Paired 5x2CV Paired T-test
This repository includes a generalized implementation of the [Mlxtend library's implementation](https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/) for Sklearn estimators.

#### References
[1] Dietterich TG (1998) Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Comput 10:1895–1923.