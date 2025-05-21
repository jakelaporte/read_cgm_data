import pandas as pd
import numpy as np
from datetime import datetime,timedelta,time
import streamlit as st

def return_time_data(series,time0,time1):
    """
    return_time_data - returns glucose values for all days 
        between time0 and time1.
    """
    fmt = '%m/%d/%Y-'+'%H:%M'
    df = series.copy()
    day0 = series.index[0].date().strftime("%m/%d/%Y")
    time = [t.time() for t in df.index]
    df['time']=time
    time0 = datetime.strptime(day0+'-'+time0,fmt).time()
    time1 = datetime.strptime(day0+'-'+time1,fmt).time()
    df = df[(df['time']>=time0) & (df['time']<=time1)]

    return df['glucose']

def return_day_data(data,day):
    """
    return_day_data - returns glucose values for a single day
        or a list of days.
        
    Input: data - dataframe with at least ['day', 'glucose'] as columns
           day - string or list of strings with format ('2025-03-28' or '03/28/2023')
    
    Output: pandas series object with datetime index and glucose readings as values.
    
    Example: return_day_data(df,'03/28/2023')
    
    """
    
    try:
        day = pd.to_datetime(day).date()
        glucose = data.loc[data['day']==day]['glucose']
    except:
        day=[pd.to_datetime(d).date() for d in day]
        glucose = data.loc[data['day'].isin(day)]['glucose']
    return glucose

def read_data(filename,dt_col=1,gl_col=7,
              dt_fmt='%Y-%m-%dT%H:%M:%S',
              header_=0,skip_rows=1,
              units='mg/dL',
              time_delta=5):
    
    c = 1*(units=='mg/dL')+18.018*(units=='mmol/L')
    try:
        """
        output is a dataframe with date time index and glucose as values in a column
        """
        data = pd.read_csv(filename)
        data = data.iloc[:,[dt_col,gl_col]]
        data = data.dropna()
        data['index']=pd.to_datetime(data.iloc[:,0],format=dt_fmt)
        
        data = data.set_index('index')
        cols = data.columns    
        data.index.name = 'datetime'
        data = data[[cols[1]]]
        data.columns = ['glucose']
        data['glucose'] = data['glucose']*c
    except:
        data = read_io_data(filename,dt_col=dt_col,gl_col=gl_col,
                            dt_fmt=dt_fmt,header_=header_,
                            skip_rows=skip_rows,
                            units=units,time_delta=time_delta)  
    data = data[~data.index.duplicated()]
    data = data.sort_index()

    return data


def read_io_data(filename,dt_col=1,gl_col=7,
                 dt_fmt='%Y-%m-%dT%H:%M:%S',
                 header_=0,skip_rows=1,units='mg/dL',
                 time_delta=5):
    """
    filename is a string IO from streamlit, import the data for the 
        glucose readings and the datetime for those readings into a dataframe
        that can be used by read_data.
        
    Input: filename - is actually a stream of IO data from the file 'filename'
           dt_fmt - the datetime format for the datetime data.
           dt_col - the column that the datetime appear.
           gl_col - the column that the glucose value appears.
    """

    from io import StringIO
    start_date = '2000-01-01'
    c = 1*(units=='mg/dL')+18*(units=='mmol/L')
    infile = StringIO(filename.decode("utf-8"))
    lst_data = []
    for j,line in enumerate(infile):
        row = line.split(',')
        lst_data.append(row)
    data={}
    dates = []
    for row in lst_data[header_+skip_rows:]:
        dt = row[dt_col]
        dt = dt.replace('"','')
        try:
            dt = dt.split(".")[0]
        except:
            pass
        try:
            dt = datetime.strptime(dt,dt_fmt)
            val = row[gl_col]
            data[dt]=int(float(val.rstrip())*c)
        except:
            pass
        dates.append(dt)
        val = row[gl_col]
        try:
            data[dt]=int(float(val.rstrip())*c)
        except:
            data[dt]=np.nan

    infile.close()
    data = pd.Series(data)
    data = pd.DataFrame(data,columns = ['glucose'])
    return data


