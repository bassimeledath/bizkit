# dependencies
from lifelines import KaplanMeierFitter
from lifelines.statistics import pairwise_logrank_test
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool,Band
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel
import numpy as np
import pandas as pd

# helper functions   
def make_step(x,y):
    """
    Makes a step function using x and y coordinates
    ----------
    x: x coordinates in a list
    y: y coordinates in a list
    -------
    
    """
    add_x = x[1:]
    new_x = [None]*(len(x)+len(add_x))
    new_x[::2] = x
    new_x[1::2] = add_x
    add_y = y[:-1]
    new_y = [None]*(len(x)+len(add_y))
    new_y[::2] = y
    new_y[1::2] = add_y
    return new_x, new_y

def color_negative_red(val):
    color = 'red' if val > 0.05 else 'green'
    return 'color: %s' % color

def color_generator(n):
    np.random.seed(42)
    colors = ['aliceblue', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet'
    , 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan'
    , 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
     'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 
     'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 
     'forestgreen', 'fuchsia', 'gainsboro', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'honeydew', 
     'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 
     'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 
     'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 
     'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 
     'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 
     'moccasin', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 
     'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 
     'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 
     'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 
     'violet', 'wheat', 'yellow', 'yellowgreen']
    return np.random.choice(colors,size=n)

class time_to_event(object):
    """
    Class for fitting the Kaplan-Meier estimate to record time to event
    
    """

    def fit_plot(self,dataframe,durations,group=None,event_observed=None):
        """
        Fit the model to right censored data and plot survival function
        Parameters
        ----------
          dataframe: Pandas dataframe being fit. Should have columns with durations, and optionally group and event observed.
          durations: Time to event for data point. For example, if the event is defined as "when supply for a clothing line is over" and say Women's size Small Tshirt supplies last for 45 days, then duration is 45.
          group: Column of dataframe that represents strata of dataframe (such as clusters or customer segments)
          event_observed: Whether event was observed (True) or not (False).
        Returns
        -------
        Interactive plot showing survival curves
        """
        # Determine where the visualization will be rendered
        output_notebook()  # Render inline in a Jupyter Notebook
        if group is None:
            kmf = KaplanMeierFitter()
            if event_observed is not None:
                kmf.fit(dataframe[durations], dataframe[event_observed])
            else:
                kmf.fit(dataframe[durations])
            
            # Extract kmf estimate values 
            x_temp = kmf.survival_function_.index
            y_temp = kmf.survival_function_['KM_estimate'].values
            
            # Make step transformations
            x,y = make_step(x_temp,y_temp)
            conf_upper = kmf.confidence_interval_survival_function_['KM_estimate_upper_0.95']
            conf_lower = kmf.confidence_interval_survival_function_['KM_estimate_lower_0.95']
            conf_upper = make_step(conf_upper,y_temp)[0]
            conf_lower = make_step(conf_lower,y_temp)[0]
            
            # Getting the kmf results into dataframe format
            df_for_plot = pd.DataFrame()
            df_for_plot['x'] = x
            df_for_plot['y'] = y
            df_for_plot['upper'] = conf_upper
            df_for_plot['lower'] = conf_lower

            # Instantiate a figure() object
            fig = figure(plot_height=400,
            plot_width=600,
            x_axis_label='Timeline',
            y_axis_label='Survival Probability',
            title='Graph',
            )
            fig.multi_line(xs=[df_for_plot['x']], ys=[df_for_plot['y']],
            line_width=4,
            line_color = 'firebrick')

            # confidence intervals
            fig.multi_line(xs=[df_for_plot['x'],df_for_plot['x']], ys=[df_for_plot['upper'],df_for_plot['lower']],
            line_width=2,alpha=0.5,line_dash='dashed',legend='confidence bound')

            #add hovering tooltips
            tooltips = [
            ('Time','$x'),
            ('Survival Probability', '$y')
            ]
            fig.legend.click_policy="hide"
            fig.add_tools(HoverTool(tooltips=tooltips))

            show(fig)
            
        else:
            list_segment_vals_x = []
            list_segment_vals_y = []
            for segment in dataframe[group].unique():
                df_to_fit = dataframe[dataframe[group]==segment]
                kmf = KaplanMeierFitter()
                if event_observed is not None:
                    kmf.fit(df_to_fit[durations], df_to_fit[event_observed])
                else:
                    kmf.fit(df_to_fit[durations])
                x_temp = kmf.survival_function_.index
                y_temp = kmf.survival_function_['KM_estimate'].values
            
                # Make step transformations
                x,y = make_step(x_temp,y_temp)
                list_segment_vals_x.append(x)
                list_segment_vals_y.append(y)
                
            # Instantiate a figure() object
            fig = figure(plot_height=400,
                         plot_width=600,
                         x_axis_label='Timeline',
                         y_axis_label='Survival Probability',
                         title='Graph',
                        )
            count = 0
            colors = color_generator(n=len(dataframe[group].unique()))

            for val_1,val_2 in zip(list_segment_vals_x,list_segment_vals_y):
                fig.line(x=val_1, y=val_2,
                     line_width=2,legend=[str(x) for x in dataframe[group].unique()][count],
                     color=colors[count])
                count+=1

            # add hovering tooltips
            tooltips = [
                        ('Time','$x'),
                        ('Survival Probability', '$y')
                       ]
            fig.add_tools(HoverTool(tooltips=tooltips))
            fig.legend.click_policy="hide"
            show(fig)
        return self
            
    def significance_results(self,dataframe,durations,group,event_observed=None):
        """
        Fit the model to right censored data and plot survival function
        Parameters
        ----------
          dataframe: Pandas dataframe being fit. Should have columns with durations, group and event observed.
          durations: Time to event for data point. For example, if the event is defined as "when supply for a clothing line is over" and say Women's size Small Tshirt supplies last for 45 days, then duration is 45.
          group: Column of dataframe that represents strata of dataframe (such as clusters or customer segments)
          event_observed: Whether event was observed (True) or not (False).
        Returns
        -------
        Table showing significance results at the 0.05 level.
        """
        df_pairwise = pd.DataFrame()
        if event_observed is not None:
            df_pairwise =pairwise_logrank_test(dataframe[durations],dataframe[group],
                                           dataframe[event_observed]).summary[['p']]
        else:
            df_pairwise =pairwise_logrank_test(dataframe[durations],dataframe[group]).summary[['p']]
        df_pairwise.columns = ['p-value']
        df_pairwise = df_pairwise.style.applymap(color_negative_red)
        
        return df_pairwise
            
        