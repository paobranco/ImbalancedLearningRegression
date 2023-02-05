Nearmiss Undersampling
========================================================

Nearmiss is an under-sampling method that....

.. py:function:: nearmiss(data, y, samp_method = "balance", drop_na_col = True, drop_na_row = True, rel_thres = 0.5, rel_method = "auto", rel_xtrm_type = "both", rel_coef = 1.5, rel_ctrl_pts_rg = None, k = 3, n_jobs = 1, k_neighbors_classifier = None)

   
   :param data: Pandas dataframe, the dataset to re-sample.
   :type data: :term:`Pandas dataframe`
   :param str y: Column name of the target variable in the Pandas dataframe.
   :param str samp_method: Method to determine re-sampling percentage. Either ``balance`` or ``extreme``.
   :param bool drop_na_col: Determine whether or not automatically drop columns containing NaN values. The data frame should not contain any missing values, so it is suggested to keep it as default.
   :param bool drop_na_row: Determine whether or not automatically drop rows containing NaN values. The data frame should not contain any missing values, so it is suggested to keep it as default.
   :param float rel_thres: Relevance threshold, above which a sample is considered rare. Must be a real number between 0 and 1 (0, 1].
   :param str rel_method: Method to define the relevance function, either ``auto`` or ``manual``. If ``manual``, must specify ``rel_ctrl_pts_rg``.
   :param str rel_xtrm_type: Distribution focus, ``high``, ``low``, or ``both``. If ``high``, rare cases having small y values will be considerd as normal, and vise versa.
   :param float rel_coef: Coefficient for box plot.
   :param rel_ctrl_pts_rg: Manually specify the regions of interest. See `SMOGN advanced example <https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb>`_ for more details.
   :type rel_ctrl_pts_rg: :term:`2D array`
   :param int k: The number of neighbors considered. Must be a positive integer.
   :param int n_jobs: The number of parallel jobs to run for neighbors search. Must be an integer. See `sklearn.neighbors.KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_ for more details.
   :param k_neighbors_classifier: If users want to define more parameters of KNeighborsClassifier, such as ``weights``, ``algorithm``, ``leaf_size``, and ``metric``, they can create an instance of KNeighborsClassifier and pass it to this method. In that case, setting ``k`` and ``n_jobs`` will have no effect.
   :type k_neighbors_classifier: :term:`KNeighborsClassifier`
   :return: Re-sampled dataset.
   :rtype: :term:`Pandas dataframe`
   :raises ValueError: If an input attribute has wrong data type or invalid value, or relevance values are all zero or all one.

References
----------
[1] I. Mani, I. Zhang. “kNN approach to unbalanced data distributions: a case study involving information extraction,” In Proceedings of workshop on learning from imbalanced datasets, 2003.

[2] 


Examples
--------
.. doctest::

    >>> from ImbalancedLearningRegression import nearmiss
    >>> housing = pandas.read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv")
    >>> housing_repeated_enn = nearmiss(data = housing, y = "SalePrice")
