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
body+="with graphs to visualize the data.\n This tool will assist the app user to "
body+="set up the file structure so that all of the files can be loaded. It is important that "
body+="the data used have the same structure in terms of the following:] \n"
body+="1) header row (the row that the header row appears on should be consistent) \n"
body+="2) rows to skip (the number of rows that needs to be skipped should be the same) \n"
body+="3) time delta (the difference between observations should have the same time difference) \n"
body+="4) unit of measure (chose between mg/dL or mmol/L) \n"
body+="5) date-time column in the same relative order and glucose column in the same order."

st.markdown(body)
body="#### Video: [Data/File Structure](https://youtu.be/Bsf5e3RWe8Q)"
st.sidebar.markdown(body)
restart=st.sidebar.button("Restart")
if restart:
    rd.initialize_session()
    st.switch_page("app_launch.py")
st.divider()
skip_rows = st.session_state['skip_rows']
current_file = st.session_state['current_file']
dt_col = st.session_state['date_col']
gl_col = st.session_state['glucose_col']
time_delta = st.session_state['time_delta']
header_row = st.session_state['header_row']
units_ = st.session_state['units']
calc_units = st.session_state['calc_units']

if current_file is None:
    current_file = st.file_uploader("Select file to explore.",
                                            type = 'csv')
if current_file is not None:
    st.session_state['current_file'] = deepcopy(current_file)
    df = rd.view_raw_data(current_file.read(),skip_rows=skip_rows,header_=header_row,stop_=15)
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
                             options = ["mg","mmol"],
                             index = ["mg","mmol"].index(units_))
    
    radio_options = ['mg','mmol']
    idx = radio_options.index(calc_units)
    calc_units = st.sidebar.radio("Choose units for calculations:",
                                options=radio_options,
                                index=idx,
                                key = 'radio_units_calc',
                                on_change = rd.change_attribute,
                                args = ('calc_units','radio_units_calc'))
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
        body = ":red[Fix the header row or skip rows. Currently the values are: "
        body += "header="+str(header_row)+"; number of skip rows="+str(skip_rows)+".]"
        st.markdown(body)
    try:
        date_col = date_col[0][0]
        glucose_col = glucose_col[0][0]
        cols = df.columns[[date_col,glucose_col]]
        df.iloc[:,date_col]=df.iloc[:,date_col].apply(lambda x: x.split(".")[0])
        try:
            df[cols[0]] = pd.to_datetime(df[cols[0]])
        except:
            fmt_options = ['m/d/y h:m',
                           'm-d-y h:m','m-d-y h:m:s',
                           'd/m/y h:m',
                           'd-m-y h:m',
                           'y/m/d h:m',
                           'y-m-d h:m','y-m-d h:m:s',
                           'Not In Selection']
            fmt_dates =   ['%m/%d/%Y %H:%M',
                           '%m-%d-%Y %H:%M','%m-%d-%Y %H:%M:%S',
                           '%d/%m/%Y %H:%M',
                           '%d-%m-%Y %H:%M',
                           '%Y/%m/%d %H:%M',
                           '%Y-%m-%d %H:%M','%Y-%m-%d %H:%M:%S',
                           None]
            st.sidebar.markdown("#### Video:[date-time](https://youtu.be/7sbFFIxALvA)")
            select_date = st.sidebar.selectbox("Select a standard date-time.",
                                 options = fmt_options, index = 0,key = 'dt_selected',
                                 on_change=rd.change_date_time,
                                 args=(fmt_options,fmt_dates))
            idx = fmt_options.index(select_date)
            fmt_str = fmt_dates[idx]
        
            fmt_str = st.sidebar.text_input("Date-Time Format:",
                                            value=fmt_str)
            st.sidebar.image('https://images.squarespace-cdn.com/content/v1/5be5c21e75f9ee21b5817cc2/e5a3ad17-463f-463a-98a0-7dfafcd5957b/fig02_datetime.png?format=1000w')
            df[cols[0]]=(pd.to_datetime(df[cols[0]],format=fmt_str))
            st.session_state['date_format']=fmt_str
        st.write("If you see the data and it looks correct, then click <OK> and continue to Import Data")
        st.write(df[cols])
        st.session_state['date_col']=date_col
        st.session_state['glucose_col']=glucose_col
        st.session_state['date_format']=fmt_str
        st.session_state['time_delta'] = time_delta
        st.session_state['units']=units_
        ok_btn = st.sidebar.button(label=":information_source:**OK** ->:file_cabinet: Import",on_click=rd.add_page_link,
                           args=(":file_cabinet: Import_Data","pages/2_import_data.py"))
    except:
        pass
