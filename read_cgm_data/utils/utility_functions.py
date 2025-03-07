import pandas as pd
import numpy as np
from io import StringIO
import streamlit as st

def view_raw_data(iofile,skip_rows = 1,header_ = 0,stop_=None):
    infile = StringIO(iofile.decode("utf-8"))
    str_data = ""
    lst_data = []
    for j,line in enumerate(infile):
        str_data += 'row '+str(j)+'==>>'+line +"\n"
        row = line.split(",")
        for i in range(len(row)):
            try:
                row[i]=int(row[i])
            except:
                try:
                    row[i]=float(row[i])
                except:
                    try:
                        row[i]=row[i].replace('"','')
                    except:
                        pass
        lst_data.append(row)
        if j == stop_:
            break
    data = {}
    header = lst_data[header_]
    for i in range(len(header)):
        data[header[i]]=[]
    for line in lst_data[header_+skip_rows:]:
        for i, col in enumerate(line):
            data[header[i]].append(col)
    try:
         data = pd.DataFrame(data)
    except:
         data=[]
         data.append(header)
         for l in lst_data[skip_rows:]:
            data.append(l)
    return data

def view_list(data,idx=0,color='salmon'):
    num_columns = 0
    for d in data:
        num_columns = max(num_columns,len(d))
    for d in data:
        while len(d)<num_columns:
            d.append(np.nan)
            
    header = ['col'+str(i) for i in range(num_columns)]
    data=pd.DataFrame(data,columns=header)
    data=data.style.applymap(lambda _:f"background-color: {color}",subset=(data.index[idx],))
    return data


# def view_raw_data_(iofile,skip_rows = 1,stop_=None):
#     infile = StringIO(iofile.decode("utf-8"))
#     header = infile.readline().split(',')
#     data={}
#     for i in range(len(header)):
#         data[header[i]]=[]
#     for i in range(skip_rows-1):
#         infile.readline()
#     for j,line in enumerate(infile):
#         row = line.split(',')
#         for i,col in enumerate(row):
#             data[header[i]].append(col)
#         if j == stop_:
#             break
#     return data

def initialize_session():
    st.session_state['cgm_data']=None
    st.session_state['current_file']=None
    st.session_state['skip_rows'] = 1
    st.session_state['date_col'] = None
    st.session_state['glucose_col'] = None
    st.session_state['date_format'] = '%Y-%m-%dT%H:%M:%S'
    st.session_state['pages_dict'] = {":house: Home":"app_launch.py",
                                    ":information_source: Data Structure":"pages/1_read_data.py",
                                    #"Test":"pages/test_page.py",
                                    }
    st.session_state['time_delta'] = 5
    st.session_state['header_row'] = 0
    st.session_state['units']= "mg/dL"
    st.session_state['cohort_stats'] = None
    #st.switch_page("app_launch.py")
    

def session():
    for key in st.session_state.keys():
        st.write(key, st.session_state[key])
    return None

def change_attribute(key1,key2):
    val = st.session_state[key2]
    st.session_state[key1]=val
    return None

def change_date_time(fmt_options,fmt_dates):
    val = st.session_state['dt_selected']
    idx = fmt_options.index(val)
    st.session_state['date_format'] = fmt_dates[idx]
    return None


def display_page_links(pages):
    for key in pages.keys():
        st.sidebar.page_link(pages[key],label=key)
    return None

def add_page_link(page_key,page_link):
    pages = st.session_state['pages_dict']
    pages[page_key]=page_link
    st.session_state['pages_dict']=pages


def remove_page_link(page_key):
    pages = st.session_state['pages_dict']
    if page_key in pages.keys():
        del pages[page_key]
    st.session_state['pages_dict']=pages

def extract_time_period_data(df,period,name,hours,period_idx,deltat=5):
    total = (hours*60)//deltat+1
    minutes = np.arange(0,hours*60+deltat,deltat)
    try:
        ex = df.loc[period[0]:period[1],['imputed']].iloc[-total:]
        ex.index=minutes
        ex.columns = [f'{name}_{period_idx}']
    except:
        ex=df.loc[period[0]:period[1],'imputed']
    
    return ex

