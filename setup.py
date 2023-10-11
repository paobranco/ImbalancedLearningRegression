from setuptools import setup

setup(
    
    name = 'ImbalancedLearningRegression',
    version = '0.0.1',
    description = 'Python implementations of preprocesssing imbalanced data for regression',
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    author = 'Paula Branco, Wenglei Wu, Alex Chengen Lyu, Lingyi Kong, Gloria Hu',
    author_email = 'pbranco@uottawa.ca, wwu077@uottawa.ca, clyu039@uottawa.ca, lkong073@uottawa.ca, xhu005@uottawa.ca',
    url = 'https://github.com/paobranco/ImbalancedLearningRegression',
    classifiers = [
        
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
        ],
    
    keywords = [
        
        'smote',
        'Gaussian noise',
        'condensed nearest neighbour',
        'edited nearest neighbour',
        'Tomek links',
        'ADASYN',
        'over-sampling',
        'under-sampling',
        'synthetic data',
        'imbalanced data',
        'pre-processing',
        'regression'
    ],
    
    packages = ['ImbalancedLearningRegression'],
    include_package_data = True,
    install_requires = ['numpy', 'pandas', 'tqdm', 'scikit-learn'],
)
