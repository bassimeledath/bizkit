import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import itertools
import plotly.graph_objects as go 

def elbow_method(df, num_c = 10, breakdown = False, max_features = 2):
    """
    takes numerical value columns of dataframe --> user has to clean
    if breakdown is True, max_features matters
    
    Consult SKLearn Documentation for KMeans Algorithm:
    (will run on default KMeans parameters)
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    assert type(df) is pd.DataFrame
    assert df.shape[1] > 1
    assert max_features > 1
    
    iter_nums = np.arange(1, num_c)

    if not breakdown: 
        X1 = df.values
        inertia = [] #Squared Distance between Centroids and data points
        for c in iter_nums:
            model = KMeans(n_clusters = c)
            model.fit(scale(X1))
            inertia.append(model.inertia_)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = iter_nums, y = inertia, title = 'All Features',
                             line = dict(color = 'red', dash = 'dot')))
        fig.update_layout(title='Elbow Plot',
                       xaxis_title='Number of Clusters',
                       yaxis_title='Inertia')
        fig.show()
    else:
        combs = get_combinations(df, max_features)
        group_inertia = get_inertias(df, combs, iter_nums)
        plot_features(group_inertia, iter_nums, combs)

def get_combinations(df, max_features):
    assert max_features <= len(df.columns), 'Max Features must be less than existing dataframe features'
    cols = df.columns
    all_combs = []
    for i in range(2, max_features+1): 
      combs = []
      for subset in itertools.combinations(cols, i): 
        combs.append(subset)
      all_combs.append(combs)
    return all_combs

def get_inertias(df, all_combs, iter_nums):
    group_inertia = []
    for subcomb in all_combs: 
      subgroup_inertias = []
      for comb in subcomb: 
        comb_inertia = []
        for c in iter_nums: 
          model = KMeans(n_clusters = c, algorithm = "elkan")
          model.fit(scale(df[list(comb)].values))
          comb_inertia.append(model.inertia_)
        subgroup_inertias.append(comb_inertia)
      group_inertia.append(subgroup_inertias)
    return group_inertia

def plot_features(group_inertia, iter_nums, all_combs):
    for group in group_inertia: 
      fig = go.Figure()
      for combs in group: 
        fig.add_trace(go.Scatter(x = iter_nums, y = combs, name = str(all_combs[group_inertia.index(group)][group.index(combs)]), line = dict(color = 'purple', dash = 'dot')))
      fig.update_layout(title='Elbow Plot {} Features'.format(group_inertia.index(group) + 2), xaxis_title='Number of Clusters', yaxis_title='Inertia')
      fig.show()

