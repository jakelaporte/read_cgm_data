import streamlit as st
import read_cgm_data as rd

cgm_data = st.session_state['cgm_data']
pages_master = st.session_state['pages_master']
pages_dict = st.session_state['pages_dict']
calc_units = st.session_state['calc_units']

if len(cgm_data.names)>2:
    rd.add_page_link(pages_master[4][0],pages_master[4][1])
rd.add_page_link(pages_master[5][0],pages_master[5][1])
#update transform self.series
#rd.add_page_link(pages_master[6][0],pages_master[6][1])
    



rd.display_page_links(pages_dict)

name = st.sidebar.selectbox("Choose a file:",
                     options = cgm_data.names,
                     index = 0)




options = ["View Data","Ambulatory Glucose Profile",
           "Glycemia Risk Index","AGP Report",
           "Time In Range Report","Visualize Data",
           "Markov Analysis"
           ]
select = st.pills("Select a tool:",
                  options = options,
                  default = options[0])
if select == options[0]:
    st.subheader("View Data")
    cgm_data.view_df_series(name)
if select == options[1]:
    st.subheader("Ambulatory Glucose Profile")
    cgm_data.ambulatory_glucose_profile(name)
if select ==options[2]:
    st.subheader("Glycemia Risk Index")
    cgm_data.view_gri(name)

if select == options[3]:
    st.subheader("AGP Report")
    cgm_data.agp_report(name)

if select == options[4]:
    st.subheader("Time In Range Report")
    cgm_data.time_in_range_report(name)

if select == options[5]:
    st.subheader("Visualize Data")
    cgm_data.visualize_data(name)

if select == options[6]:
    cgm_data.markov_analysis(name)
