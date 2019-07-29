
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, BoxSelectTool
import pandas as pd
import numpy as np

class anomaly(object):
    """
    Class for fitting Isolation Forest algorithm for anomaly detection

    """

    def __init__(self):
        self.ran_detect_anomaly = False

    def get_columns(self, dataframe, date_col, value_col):

        """
        Get columns from the user
        An alert will be generated if the columns have missing data, and the program stops

        Re-name the columns to be standard
        Convert the date to the format of datetime
        Create a column time_epoch as int64 which is required by the algorithm of Isolation Forest

        Parameters
        -----------------
            dataframe: a pd.dataframe to be fed, a new column of Anomaly will be created to the dataframe after applying the algorithm
            date_col: column of date, series
            value_col: column of the value that the user wants to investigate on the time axis, series

        Return
        ----------
        standardized df['Date'], df['value'], df['timestamp'], df['time_epoch']

        """

        if (date_col.isnull().values.any() == True):
            print('\033[91m' + '\033[1m' + "Alert: Please fill in the missing data before running the program!")
            raise RuntimeError("Alert: Please fill in the missing data before running the program!")

        if (value_col.isnull().values.any() == True):
            print('\033[91m' + '\033[1m' + "Alert: Please fill in the missing data before running the program!")
            raise RuntimeError("Alert: Please fill in the missing data before running the program!")

        df=dataframe        
        df['Date'] = date_col
        df['value'] = value_col
        df['timestamp']=pd.to_datetime(df['Date'])
        df['time_epoch']=df['timestamp'].astype(np.int64)
        return df['Date'], df['value'], df['timestamp'], df['time_epoch']

    def detect_anomaly(self, dataframe, date_col, value_col, outliers_fraction=0.05):  
        """
        Detect anomalies by applying the sklearn IsolationForest algorithm, it returns anomaly scores
        Create a new column named 'anomaly' in the dataframe with the returned anomaly score of 1 (anomaly) and 0 (normal)

        Parameters
            dataframe: a pd.dataframe to be fed, a new column of Anomaly will be created to the dataframe after applying the algorithm
            date_col: column of date, series
            value_col: column of the value that the user wnat to investigate on the time axis, series
            outliers_fraction: user's input of the proportion of outliers they want to investigate, float, default is 0.05

        """
        
        self.get_columns(dataframe, date_col, value_col)
        df=dataframe
        data = df[['time_epoch', 'value']]
        min_max_scaler = StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)
        # train isolation forest 
        model =  IsolationForest(contamination = outliers_fraction)
        model.fit(data)
        # add the data to the main  
        df['anomaly'] = pd.Series(model.predict(data))
        df['anomaly'] = df['anomaly'].map( {1: 0, -1: 1} )

        normal=df[df['anomaly']==0]
        anomaly=df[df['anomaly']==1]
        print("Normal: ", len(normal))
        print("Anomaly: ", len(anomaly))

        self.ran_detect_anomaly = True


    def print_anomaly(self, dataframe, date_col, value_col):
        """
        print anomalies detected by the detect_anomaly function
        If it runs before detect_anomaly(), an error message will be generated
        
        Parameters
        ------------
            dataframe: a pd.dataframe to be fed
            date_col: column of date, series
            value_col: column of the value that the user wnat to investigate on the time axis, series
        
        Return
        ------------
        anomalies presented as pd.dataframe

        """

        if not self.ran_detect_anomaly:
            print('\033[91m' + '\033[1m' + "Alert: print_anomaly was called before detect_anomaly was run. Please run detect_anomaly() first!")
            raise RuntimeError("print_anomaly() was called before detect_anomaly() was run. Please run detect_anomaly() first!")

        df=dataframe 
        self.get_columns(df, date_col, value_col)
        a = pd.DataFrame(df.loc[df['anomaly'] == 1, ['timestamp', 'value', 'time_epoch']]) 
        return a

    def plot_anomaly(self, dataframe, date_col, value_col):

        """
        plot anomalies as red dots on the blue timeseries 
        If it runs before detect_anomaly(), an error message will be generated
        
        Parameters
        ------------
            dataframe: a pd.dataframe to be fed
            date_col: column of date, series
            value_col: column of the value that the user wnat to investigate on the time axis, series

        """

        if not self.ran_detect_anomaly:
            print('\033[91m' + '\033[1m' + "Alert: plot_anomaly was called before detect_anomaly was run. Please run detect_anomaly() first!")
            raise RuntimeError("plot_anomaly was called before detect_anomaly was run. Please run detect_anomaly() first!")

        output_notebook()  # Render inline in a Jupyter Notebook
        df=dataframe 
        a = pd.DataFrame(df.loc[df['anomaly'] == 1, ['timestamp', 'value', 'time_epoch']]) 
        fig=figure(
            plot_height=400,
            plot_width=600,
            x_axis_label='Timeline',
            x_axis_type='datetime',
            y_axis_label='Value',
            title="Time Series With Red Dots Representing Anomalies")

        fig.scatter("timestamp", "value", color="#FF0000", source=a)
        fig.line("timestamp", "value", color="#000080", source=df)

        hover=HoverTool(
            tooltips=[
                ( 'Timestamp', '@timestamp{%Y %m %d}'),
                ( 'Value', '$y'), 
            ],
            formatters={
                'timestamp': 'datetime', 
            },
        )

        fig.add_tools(hover)
        show(fig)

