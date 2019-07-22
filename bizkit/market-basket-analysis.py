# -*- coding: utf-8 -*-
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import d3fdgraph
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#helper function
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1     
      
class MarketBasketAnalysis(object):

  """
  Class for fitting Apriori algorithm for Market Basket Analysis

  Attributes
  ----------
  rules_ : DataFrame
    The rules found by Apriori and their associated metrics

  freq_items_ : DataFrame
    The frequent itemsets found by Apriori and their support
  """
  
  def fit(
      self, 
      dataframe, 
      min_support = 0.01, 
      metric = "confidence"
      min_threshold = 1,
  ):
      """
      Fit the model to a dataset 

      Parameters
      ----------
        dataframe: a pd.DataFrame 
          contains columns 'order_id', 'product_id' and 'quantity'
        min_support: float, optional
          between 0 and 1, mininum threshold for itemset support
        metric: string, optional
          supported evaluation metrics are 'support', 'confidence', 'leverage' and 'conviction'
        min_threshold: float, optional
          minimum threshold for evaluation metric

      Returns
      ----------
      self: MarketBasketAnalysis
        self with new properties like ``rules_``, ``plot_frequency`` and ``plot_network_graph`` 
      """
   
    basket = (dataframe.groupby(['order_id','product_id'])['quantity']
         .sum().unstack().reset_index().fillna(0)
         .set_index('order_id'))

    basket_sets = basket.applymap(encode_units)
    
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: list(x)[0]).astype("unicode")
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0]).astype("unicode")
    rules['consequents'] = rules['consequents'].apply(lambda y: list(y)[0]).astype("unicode")
    
    self.rules_ = rules
    self.freq_items_ = frequent_itemsets
    
    return self
  
  def plot_frequency(self, num_items = 15):
     """
      Plots frequent itemsets according to support as barplot

      Parameters
      ----------
        self: MarketBasketAnalysis object 
        num_items: int, optional
          number of top itemsets plotted
      """

    top_items = self.freq_items_.sort_values('support', ascending = False).reset_index(drop=True).head(num_items)
    
    dims = (10,5)
    fig, ax = plt.subplots(figsize=dims)
    plot = sns.barplot(x='itemsets', y='support', data=top_items)
    plot.set_title('Relative Item Frequency Plot')
    plot.set_xlabel('Itemsets')
    plot.set_ylabel('Item Frequency (Relative)')
    plot.set_xticklabels(plot.get_xticklabels(), rotation=75)
    plt.show()
    
    return None
  
  def plot_network_graph(self, metric = 'lift', collision_scale = 10):
    """
    Plots rules in a network graph 

    Parameters
      ----------
        self: MarketBasketAnalysis object 
        metric: string, optional
          supported evaluation metrics are 'support', 'confidence', 'leverage' and 'conviction'
    """

    df = self.rules_[['antecedents','consequents', metric]]
    d3fdgraph.plot_force_directed_graph(df, collision_scale = collision_scale)
    
    return None