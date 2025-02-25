import streamlit as st
import pandas as pd
import numpy as np
import read_cgm_data as rd
from copy import deepcopy

pages_master = st.session_state['pages_master']
pages_dict = st.session_state['pages_dict']
rd.display_page_links(pages_dict)

options = []

st.subheader("Instructions")
body=''
body+=":red[This application takes `.csv` files with CGM data "
body+="as input and produces variability metrics for that data along "
body+="with graphs to visualize the data.\n It is important that "
body+="the data used has the same structure in terms of columns. The "
body+="`Import Data` section of this tool will use the column number "
body+="of the glucose and the date-time of your data "
body+="selected below. Be sure to select these columns based on "
body+="your data.]  \n\n"
body+="#### Video: [Getting Started](https://youtu.be/S_QklfS5XCw)"
st.markdown(body)
st.divider()
skip_rows = st.session_state['skip_rows']
current_file = st.session_state['current_file']
dt_col = st.session_state['date_col']
gl_col = st.session_state['glucose_col']
time_delta = st.session_state['time_delta']
header_row = st.session_state['header_row']
units_ = st.session_state['units']
if current_file is None:
    current_file = st.file_uploader("Select file to explore.",
                                            type = 'csv')
if current_file is not None:
    st.session_state['current_file'] = deepcopy(current_file)
    df = rd.view_raw_data(current_file.read(),skip_rows,header_=header_row,stop_=15)
    if isinstance(df,list):
        st.write(rd.view_list(df,header_row))
    else:
        st.write(df)
    st.divider()

    header_row = st.sidebar.number_input("Choose the header row:",
                                         min_value = 0,
                                         max_value = 20,
                                         value = header_row,
                                         on_change = rd.change_attribute,
                                         key = 'header_key',
                                         args = ('header_row','header_key'))
    
    skip_rows = st.sidebar.number_input("Number of rows to skip:",
                                min_value = 1,
                                max_value = 20,
                                value = skip_rows,
                                on_change = rd.change_attribute,
                                key = 'skip_key',
                                args=('skip_rows','skip_key'))
    
    time_delta = st.sidebar.radio("Time Delta for this analysis:",
                                  key='time_delta_key',
                                  options=[1,5,15],
                                  on_change = rd.change_attribute,
                                  index = [1,5,15].index(time_delta),
                                  args = ('time_delta','time_delta_key'))
    
    units_ = st.sidebar.radio("Units of Measurement in data:",
                             options = ["mg/dL","mmol/L"],
                             index = ["mg/dL","mmol/L"].index(units_))
    try:
        view_cols = st.columns(2)
        with view_cols[0]:
            dt_bool = [False]*len(df.columns)
            if dt_col is not None:
                dt_bool[dt_col]=True
            st.markdown("#### Select Date-Time Column:")
            time_date_df = pd.DataFrame({'columns':list(df.columns),
                                        'date-time':dt_bool})
            time_date_df = st.data_editor(time_date_df,hide_index=True)
        with view_cols[1]:
            st.markdown("#### Select Glucose Column:")
            gl_bool = [False]*len(df.columns)
            if gl_col is not None:
                gl_bool[gl_col]=True
            glucose_df = pd.DataFrame({'columns':list(df.columns),
                                        'glucose':gl_bool})
            glucose_df = st.data_editor(glucose_df,hide_index=True)
        date_col = np.where(time_date_df.values[:,1]==True)
        glucose_col = np.where(glucose_df.values[:,1]==True)
        fmt_str = st.session_state['date_format']
    except:
        body = ":red[Fix the header row. It is currently assigned to row: "+str(header_row)+".]"
        st.markdown(body)
    try:
        date_col = date_col[0][0]
        glucose_col = glucose_col[0][0]
        cols = df.columns[[date_col,glucose_col]]
        try:
            df[cols[0]] = pd.to_datetime(df[cols[0]],format=fmt_str)
        except:
            st.sidebar.markdown("##### Video [Timestamp](https://youtu.be/gW4RNwWobi4)")
            fmt_str = st.sidebar.text_input("Date-Time Format:",
                                            value=fmt_str)
            df[cols[0]]=(pd.to_datetime(df[cols[0]],format=fmt_str))
        st.write("If you see the data and it looks correct, then click <OK> and continue to Import Data")
        st.write(df[cols])
        st.session_state['date_col']=date_col
        st.session_state['glucose_col']=glucose_col
        st.session_state['date_format']=fmt_str
        st.session_state['time_delta'] = time_delta
        st.session_state['units']=units_
        ok_btn = st.sidebar.button("OK",on_click=rd.add_page_link,
                           args=(":file_cabinet: Import_Data","pages/2_import_data.py"))
    except:
        pass
