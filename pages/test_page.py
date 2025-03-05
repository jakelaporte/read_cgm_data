import streamlit as st
import pandas as pd
import numpy as np
import read_cgm_data as rd
from copy import deepcopy

pages_dict = st.session_state['pages_dict']
rd.display_page_links(pages_dict)

## Initialize session parameters
current_file = st.session_state['current_file']
cgm_data = st.session_state['cgm_data']
date_col = st.session_state['date_col']
time_delta = int(st.session_state['time_delta'])
glucose_col = st.session_state['glucose_col']
date_format = st.session_state['date_format']
skip_rows = st.session_state['skip_rows']
header_row = st.session_state['header_row']
units = st.session_state['units']

    
if current_file is None:
    current_file = st.file_uploader("Select file to explore.",
                                            type = 'csv')
    
    if current_file is not None:

        df,period = rd.read_data(filename=current_file.read(),
                            dt_col=1,
                            gl_col=7,
                            dt_fmt=date_format,
                            header_= header_row,
                            skip_rows=skip_rows,
                            units=units,
                            time_delta=time_delta)
        
        st.write(df)
        cgm_data = rd.multiple_CGM(['test'],[df],[period],
                             dt_fmt=date_format,
                             units=units,
                             time_delta=5)
        cgm_data.test_develop('test')


        
# %m/%d/%Y %H:%M
# %d/%m/%Y %H:%M
# %d-%m-%Y %H:%M
