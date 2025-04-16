## Imbalanced Learning Regression

[![PyPI version](https://badge.fury.io/py/ImbalancedLearningRegression.svg)](https://badge.fury.io/py/ImbalancedLearningRegression)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![GitHub last commit](https://img.shields.io/github/last-commit/paobranco/ImbalancedLearningRegression)

## Description

A Python implementation of sampling techniques for Regression. Conducts different sampling techniques for Regression. Useful for prediction problems where regression is applicable, but the values in the interest of predicting are rare or uncommon. This can also serve as a useful alternative to log transforming a skewed response variable, especially if generating synthetic data is also of interest.
<br>

> [!NOTE]  
> You can re-sample your data online at [imbalanced-learning-regression.com](https://www.imbalanced-learning-regression.com). The website is powered by this package and has a time limit of 15 minutes for each re-sampling task. The source code is open-sourced and can be found here: [FRONTEND](https://github.com/wuwenglei/ImbalancedLearningRegressionDemoUI) and [BACKEND](https://github.com/wuwenglei/ImbalancedLearningRegressionDemo).

## Citation

If you use the code from this repository, please cite the following article:

_Wu, W., Kunz, N., & Branco, P. (2022, September). ImbalancedLearningRegression-A Python Package to Tackle the Imbalanced Regression Problem. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 645-648). Cham: Springer Nature Switzerland._

For bibtex users:

@inproceedings{wu2022imbalancedlearningregression,
title={ImbalancedLearningRegression-A Python Package to Tackle the Imbalanced Regression Problem},
author={Wu, Wenglei and Kunz, Nicholas and Branco, Paula},
booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
pages={645--648},
year={2022},
organization={Springer}
}

Article Link:
https://link.springer.com/chapter/10.1007/978-3-031-26422-1_48

## Features

1. An open-source Python supported version of sampling techniques for Regression, a variation of Nick Kunz's package SMOGN.

2. Supports Pandas DataFrame inputs containing mixed data types.

3. Flexible inputs available to control the areas of interest within a continuous response variable and friendly parameters for re-sampling data.

4. Purely Pythonic, developed for consistency, maintainability, and future improvement, no foreign function calls to C or Fortran, as contained in original R implementation.

## Requirements

1. Python 3
2. NumPy
3. Pandas
4. Scikit-learn

## Installation

```python
## install pypi release
pip install ImbalancedLearningRegression

## install developer version
pip install git+https://github.com/paobranco/ImbalancedLearningRegression.git
```

## Usage

```python
## load libraries
import pandas
import ImbalancedLearningRegression as iblr

## load data
housing = pandas.read_csv(
    ## http://jse.amstat.org/v19n3/decock.pdf
    "https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv"
)

## conduct Random Over-sampling
housing_ro = iblr.ro(
    data = housing,
    y = "SalePrice"
)

## conduct Introduction of Gaussian Noise
housing_gn = iblr.gn(
    data = housing,
    y = "SalePrice"
)
```

## Examples

1. [Random Over-sampling](https://github.com/paobranco/ImbalancedLearningRegression/blob/master/examples/Random%20Over-sampling.ipynb) <br>
2. [Introduction of Gaussian Noise](https://github.com/paobranco/ImbalancedLearningRegression/blob/master/examples/Gaussian_noise.ipynb) <br>
3. [Condensed Nearest Neighbor](https://github.com/paobranco/ImbalancedLearningRegression/blob/master/examples/Condensed%20Nearest%20Neighbour.ipynb) <br>
4. [Edited Nearest Neighbor](https://github.com/paobranco/ImbalancedLearningRegression/blob/master/examples/Edited%20Nearest%20Neighbour.ipynb) <br>

For the examples of other techniques, please refer to [here](https://github.com/paobranco/ImbalancedLearningRegression/tree/master/examples). <br>

## Documentation

The documentation of the package can be found [here](https://imbalancedlearningregression.readthedocs.io/en/latest). <br>

## License

© Paula Branco, 2022. Licensed under the General Public License v3.0 (GPLv3).

## Contributions

ImbalancedLearningRegression is open for improvements and maintenance. Your help is valued to make the package better for everyone.

## Reference

Branco, P., Torgo, L., Ribeiro, R. (2017). SMOGN: A Pre-Processing Approach for Imbalanced Regression. Proceedings of Machine Learning Research, 74:36-50. http://proceedings.mlr.press/v74/branco17a/branco17a.pdf

Branco, P., Torgo, L., & Ribeiro, R. P. (2019). Pre-processing approaches for imbalanced distributions in regression. Neurocomputing, 343, 76-99. https://www.sciencedirect.com/science/article/abs/pii/S0925231219301638

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357. https://www.jair.org/index.php/jair/article/view/10302

Elhassan, T., & Aljurf, M. (2016). Classification of imbalance data using tomek link (t-link) combined with random under-sampling (rus) as a data reduction method. Global J Technol Optim S, 1. https://www.researchgate.net/profile/Mohamed-Shoukri-2/publication/326590590_Classification_of_Imbalance_Data_using_Tomek_Link_T-Link_Combined_with_Random_Under-sampling_RUS_as_a_Data_Reduction_Method/links/5b96a6a0a6fdccfd543cbc40/Classification-of-Imbalance-Data-using-Tomek-Link-T-Link-Combined-with-Random-Under-sampling-RUS-as-a-Data-Reduction-Method.pdf

Hart, P. (1968). The condensed nearest neighbor rule (corresp.). IEEE transactions on information theory, 14(3), 515-516. https://ieeexplore.ieee.org/document/1054155

He, H., Bai, Y., Garcia, E. A., & Li, S. (2008, June). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. In 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence) (pp. 1322-1328). IEEE. https://www.ele.uri.edu/faculty/he/PDFfiles/adasyn.pdf

Kunz, N., (2019). SMOGN. https://github.com/nickkunz/smogn

Menardi, G., & Torelli, N. (2014). Training and assessing classification rules with imbalanced data. Data mining and knowledge discovery, 28(1), 92-122. https://link.springer.com/article/10.1007/s10618-012-0295-5

Tomek, I. (1976). Two modifications of CNN. IEEE Trans. Systems, Man and Cybernetics, 6, 769-772. https://ieeexplore.ieee.org/document/4309452

Torgo, L., Ribeiro, R. P., Pfahringer, B., & Branco, P. (2013, September). Smote for regression. In Portuguese conference on artificial intelligence (pp. 378-389). Springer, Berlin, Heidelberg. https://link.springer.com/chapter/10.1007/978-3-642-40669-0_33

Wilson, D. L. (1972). Asymptotic properties of nearest neighbor rules using edited data. IEEE Transactions on Systems, Man, and Cybernetics, (3), 408-421. https://ieeexplore.ieee.org/abstract/document/4309137
