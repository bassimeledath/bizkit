import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import plotly.figure_factory as ff
import plotly.express as px
import itertools
import plotly.graph_objects as go

class marseg():

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit(self, data):
        """
        
        """
        assert type(data) is pd.DataFrame
        assert data.shape[1] > 1
        
        self.data = data
        self.model = KMeans(n_clusters = self.num_clusters)
        self.model.fit(data.values)
        self.data['label'] = self.model.labels_
        
        print('Model')
        print(self.model)
        print('\n')
        print('Label Value Couts')
        print(self.data['label'].value_counts())
    
    def visualize(self):
      self.visualize_feature_dists()
      if self.data.shape[1] == 4:
        self.visualize_3d()
      self.radarplots()
        
    def visualize_feature_dists(self): 
      all_features = []
      for feature in cluster_data.iloc[:, :3].columns: 
        feature_hdata = []
        for label in range(5): 
          data = cluster_data[feature][cluster_data.label == label]
          feature_hdata.append(data)
        all_features.append(feature_hdata)
    
      group_labels = ['Group 0', 'Group 1', 'Group 2', 'Group 3', 'Group 4']

      for f in all_features:     
        feature_np = np.array(f) 
        fig = ff.create_distplot(feature_np, group_labels, bin_size=.8)
        fig.show()
        
    def visualize_3d(self): 
      cols = self.data.columns
      fig = px.scatter_3d(self.data, x=cols[0], y=cols[1], z=cols[2],
              color='label')
      fig.show()

    def radarplots(self):
      fig1 = go.Figure()
      
      features = self.data.columns[:3]
      groupings = self.data.groupby('label').mean()
      names = ['group{}'.format(i+1) for i in range(self.num_clusters)]
      
      for f in features: 
        fig1.add_trace(go.Scatterpolar(
        r=groupings[f],
        theta=names,
        fill='toself', 
        name=f
        ))
        
      fig1.show()

      fig2 = go.Figure()
      
      groupings_t = groupings.T
      
      for g in range(self.num_clusters): 
        fig2.add_trace(go.Scatterpolar(
          r=groupings_t.iloc[:, g], 
          theta=features, 
          fill='toself', 
          name='group{}'.format(g)
        ))
        
      fig2.show()
