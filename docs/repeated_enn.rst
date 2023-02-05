Repeated Edited Nearest Neighbor
========================================================

Repeated Edited Nearest Neighbor is an under-sampling method that  utilizes Edited Nearest Neighbor to undersample the majority set in a recurring fashion by removing samples over numerous iterations of ENN. This process continues until no further samples can be removed, or a maximum iteration has been hit.

.. py:function:: repeated_enn(data, y, samp_method = "balance", drop_na_col = True, drop_na_row = True, rel_thres = 0.5, rel_method = "auto", rel_xtrm_type = "both", rel_coef = 1.5, rel_ctrl_pts_rg = None, k = 3, n_jobs = 1, k_neighbors_classifier = None, max_iter = 100)

   
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
   :param max_iter: This specifies the maximum number of iterations of the enn.py function that can be called before returning the undersampled dataset. The default value for this parameter is 100, but if no further samples can be removed before then, the dataset will be returned.
   :return: Re-sampled dataset.
   :rtype: :term:`Pandas dataframe`
   :raises ValueError: If an input attribute has wrong data type or invalid value, or relevance values are all zero or all one.

References
----------
[1] D. Wilson, “Asymptotic Properties of Nearest Neighbor Rules Using Edited Data,” In IEEE Transactions on Systems, Man, and Cybernetrics, vol. 2 (3), pp. 408-421, 1972.
[2] I. Tomek, “An Experiment with the Edited Nearest-Neighbor Rule,” IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6), pp. 448-452, June 1976.

Examples
--------
.. doctest::

    >>> from ImbalancedLearningRegression import enn
    from ImbalancedLearningRegression import repeated_enn
    >>> housing = pandas.read_csv("https://raw.githubusercontent.com/paobranco/ImbalancedLearningRegression/master/data/housing.csv")
    >>> housing_repeated_enn = repeated_enn(data = housing, y = "SalePrice")
