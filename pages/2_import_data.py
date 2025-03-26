import streamlit as st
import read_cgm_data as rd

pages_dict = st.session_state['pages_dict']
pages_master = st.session_state['pages_master']
rd.display_page_links(pages_dict)
st.subheader("Import Data")

## Initialize session parameters
cgm_data = st.session_state['cgm_data']
date_col = st.session_state['date_col']
time_delta = int(st.session_state['time_delta'])
glucose_col = st.session_state['glucose_col']
date_format = st.session_state['date_format']
skip_rows = st.session_state['skip_rows']
header_row = st.session_state['header_row']
units = st.session_state['units']
calc_units = st.session_state['calc_units']
ok_btn = False

if cgm_data is None:

    names = [];dataframes=[];periods = []
    uploaded_files = st.file_uploader("Select .csv files to upload.",
                                    type="csv",
                                    accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        names.append(uploaded_file.name)
        file = uploaded_file.read()
        df,period = rd.read_data(filename=file,
                          dt_col=date_col,
                          gl_col=glucose_col,
                          dt_fmt=date_format,
                          header_=header_row,
                          skip_rows=skip_rows,
                          units=units,
                          time_delta=time_delta)
        #st.write(df)
        dataframes.append(df)
        periods.append(period)

    if len(names)>0:
        first_full_day = st.session_state['ffd_check']
        ### all files are converted to mg/dL, here the calculations are determined
        ### by the button on the load page.
        cgm_data = rd.multiple_CGM(names,
                                dataframes,
                                periods,
                                date_format,
                                units, 
                                time_delta,
                                first_full_day)
        st.session_state['cgm_data'] = cgm_data
        rd.remove_page_link(pages_master[1][0])
        rd.remove_page_link(pages_master[2][0])
        ok_btn = st.sidebar.button("OK",key="sb_ok_pg2_",on_click=rd.add_page_link,
                        args=(pages_master[3][0],pages_master[3][1]))
    else:
        first_full_day = st.sidebar.checkbox("Begin stats on first full day:",
                                         value=False,
                                         key='ffd_check')
        
else:
    st.switch_page("pages/3_explore_data.py")
    
        
        


