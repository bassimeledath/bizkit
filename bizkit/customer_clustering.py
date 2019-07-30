#dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import itertools

#Hyperparameter Tuning Helper 
def elbow_plots(df, num_c = 10, breakdown = False, max_features = 2):
    """
    Displays elbow plots for various number of features.
    ------------
    df: dataframe containing customer data 
    num_c: number of clusters
    breakdown: whether to show plots for subgroups of features
    max_features: when breakdown is True, max number of features in subgroups
    ------------
    """
    assert type(df) is pd.DataFrame
    assert df.shape[1] > 1
    assert max_features > 1
    
    #elbow_plots Helper Function 1
    def get_combinations(df, max_features):
        """
        Obtain all combinations of features from the data
        ------------
        df: dataframe containing customer data
        max_features: when breakdown is True, max number of features in subgroups
        ------------
        """
        assert max_features <= len(df.columns)
        cols = df.columns
        all_combs = []
        for i in range(2, max_features+1): 
            combs = []
            for subset in itertools.combinations(cols, i): 
                combs.append(subset)
            all_combs.append(combs)
        return all_combs

    #elbow_plots Helper Function 2
    def get_inertias(df, all_combs, iter_nums):
        """
        Obtain losses per iteration
        ------------
        df: dataframe containing customer data 
        all_combs: all combinations of features
        iter_nums: number of iterations; each iteration == cluster number
        ------------
        """
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

    #elbow_plots Helper Function 3
    def plot_features(group_inertia, iter_nums, all_combs):
        """
        Plots inertias over iteration numbers
        ------------
        group_inertia: contains inertias per subgroup 
        iter_nums: number of iterations; each iteration == cluster number
        all_combs: all combinations of features
        ------------
        """
        for group in group_inertia: 
            fig = go.Figure()
            for combs in group: 
                fig.add_trace(go.Scatter(x = iter_nums,
                                     y = combs,
                                     name = str(all_combs[group_inertia.index(group)][group.index(combs)]),
                                     line = dict(color = 'purple', dash = 'dot')))
            fig.update_layout(title='Elbow Plot {} Features'.format(group_inertia.index(group) + 2),
                            xaxis_title='Number of Clusters',
                            yaxis_title='Inertia')
            fig.show()

    iter_nums = np.arange(1, num_c)

    if not breakdown: 
        X1 = df.values
        inertia = []
        for c in iter_nums:
            model = KMeans(n_clusters = c).fit(scale(X1))
            inertia.append(model.inertia_)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = iter_nums,
                                 y = inertia,
                                 name = 'All Features',
                                 line = dict(color = 'blue', dash = 'dot')))
        fig.update_layout(title = 'Elbow Plot',
                          xaxis_title = 'Number of Clusters',
                          yaxis_title = 'Inertia')
        fig.show()
    else:
        combs = get_combinations(df, max_features)
        group_inertia = get_inertias(df, combs, iter_nums)
        plot_features(group_inertia, iter_nums, combs)


    
class marseg():
    """
    Class for fitting KMeans algorithm to data and producing visualizations
    """

    def __init__(self, k):
        """
        Initializes model
        ------------
        k: number of clusters
        ------------
        """
        self.num_clusters = k

    def fit(self, data):
        """
        Fits the KMeans model with numerical customer features
        ------------
        data: dataframe containing feature columns
        ------------
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
        
        return self 
    
    def visualize(self):
        """
        Calls all the visualization functions 
        """
        self.visualize_feature_dists()
        if self.data.shape[1] == 4:
            self.visualize_3d()
        self.radarplots()
        
    def visualize_feature_dists(self):
        """
        For each feature, visualizes distribution of values across clusters
        """
        cluster_data = self.data
        feature_names = cluster_data.iloc[:, :3].columns
        group_labels = ['Group {}'.format(num + 1) for num in range(self.num_clusters)]
        
        for feature in feature_names: 
            feature_hdata = []
            for label in range(self.num_clusters): 
                data = cluster_data[feature][cluster_data.label == label]
                feature_hdata.append(data)
            fig = ff.create_distplot(feature_hdata, group_labels, bin_size = 0.8)
            fig.update_layout(title_text = feature) 
            fig.show()
        
    def visualize_3d(self):
        """
        Produces 3D plot for 3 features
        """
        cols = self.data.columns
        fig = px.scatter_3d(self.data,
                            x=cols[0],
                            y=cols[1],
                            z=cols[2],
                            color='label')
        fig.show()

    def radarplots(self):
        """
        Produces radar plots that compares aggregated values across features
        and across clusters
        """
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
