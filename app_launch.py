import streamlit as st
from copy import deepcopy
import numpy as np
import pandas as pd
import read_cgm_data as rd

pages_master = {0:[":house: Home","app_launch.py"],
              1:[":information_source: Data Structure","pages/1_read_data.py"],
              2:[":file_cabinet: Import_Data","pages/2_import_data.py"],
              3:[":man: Explore Data","pages/3_explore_data.py"],
              4:[":woman-woman-boy-boy: Cohort Data","pages/4_compare_data.py"],
              5:[":floppy_disk: Export Data","pages/5_export_data.py"]
}

if 'pages_master' not in st.session_state:
    st.session_state['pages_master'] = pages_master
if 'cgm_data' not in st.session_state:
    st.session_state['cgm_data']=None
if 'current_file' not in st.session_state:
    st.session_state['current_file']=None
if 'skip_rows' not in st.session_state:
    st.session_state['skip_rows'] = 1
if 'date_col' not in st.session_state:
    st.session_state['date_col'] = None
if 'glucose_col' not in st.session_state:
    st.session_state['glucose_col'] = None
if 'date_format' not in st.session_state:
    st.session_state['date_format'] = '%Y-%m-%dT%H:%M:%S'
if 'header_row' not in st.session_state:
    st.session_state['header_row'] = 0
if 'pages_dict' not in st.session_state:
    st.session_state['pages_dict'] = {pages_master[0][0]:pages_master[0][1],
                                    pages_master[1][0]:pages_master[1][1],
                                    #"Test":"pages/test_page.py",
                                    }
if 'time_delta' not in st.session_state:
    st.session_state['time_delta'] = 5
if 'units' not in st.session_state:
    st.session_state['units']="mg/dL"
if 'cohort_stats' not in st.session_state:
    st.session_state['cohort_stats'] = None




pages_dict = st.session_state['pages_dict']
rd.display_page_links(pages_dict)

st.markdown('# GVC-Calc')
body = ""
options = [":orange_book: About",":scroll: Documentation"]
select = st.sidebar.radio(label = "Select:",options=options)

if st.session_state['current_file'] is not None:
    st.sidebar.button("Restart Session",on_click=rd.initialize_session)
if select == options[0]:
    st.subheader("About")
    body = "GV-Calc was developed to assist researchers with Continuous Glucose Montioring (CGM) "
    body += "data by West Point's AI Data Engineering and Machine Learning (AIDE-ML) Center. "
    body += "The focus of the tool is to calculate as many glycemic metrics as possible while "
    body += "documenting the choices that were made during the computation of each. "
    body += "We focus on csv files and assisting the user with the ability to get load their "
    body += "file into the system assuming all of the files are structured the same."
    st.markdown(body)
    st.subheader("Process")
    body = "Structure of the csv file - 1) datetime format, 2) datetime column number, 3) glucose "
    body += "column number, and 4) number of rows to skip before getting to the data. "
    body += "For files that have the same structure, see Figure 1, organize them into a folder and "
    body+= "use the `Data Format` tool to capture the structure of the csv files."
    st.markdown(body)
    img_link = "https://images.squarespace-cdn.com/content/v1/5be5c21e75f9ee21b5817cc2/"
    img_link+="bffaf427-6a80-49c7-acbf-8d64a92cb3cc/structure_csv_cgm_file.png?format=1000w"
    st.image(img_link,
             width = 400,
             caption = "Figure 1: Structure of a CGM File.")
    body = "After the structure is documented, you will be able to download all of the files that "
    body += "have the same structure and use the tool to analyze your data."
    st.markdown(body)
if select == options[1]:
    st.markdown("## Documentation")
    st.markdown("##### :medical_symbol: How should the data be displayed")
    link="- [Statistical Packages and Algorithms for the Analysis of Continuous Glucose Monitoring Data: A Systematic Review]"
    link+="(https://journals.sagepub.com/doi/full/10.1177/19322968231221803) "
    link+="- a review of the CGM statistical packages and algorithms. (2024)"
    st.markdown(link)

    link = "- [Continuous glucose monitoring and metrics for clinical trials: an international consensus statement]"
    link += "(https://www.thelancet.com/journals/landia/article/PIIS2213-8587(22)00319-9/abstract)"
    link += "- recommend the ways CGM data might be used in prospective clinical studies, either "
    link += " as a specified study endpoint or as supportive complementary glucose metrics. (2023)"
    st.markdown(link)


    link = "- [Clinical Targets for Continuous Glucose Monitoring Data Interpretation:"
    link += " Recommendations From the International Consensus on Time in Range]"
    link += "(https://diabetesjournals.org/care/article/42/8/1593/36184/Clinical-Targets-for-Continuous-Glucose-Monitoring)"
    link += "- This article summarizes the Advanced Technologies & Treatments for Diabetes (ATTD) consensus recommendations "
    link += "for relevant aspects of CGM data utilization and reporting among the various diabetes populations. (2019)"
    st.markdown(link)
    
    link = "- [Glucose Variability: A Review of Clinical Applications and Research Developments]"
    link += "(https://www.liebertpub.com/doi/full/10.1089/dia.2018.0092)"
    link += "- states the need for a large accessible database for reference populations to provide a basis for automated "
    link += "interpretation of glucose variability and other features of continuous glucose monitoring records. (2018)"
    st.markdown(link)
    
    link = "- [International Consensus on Use of Continuous Glucose Monitoring]"
    link += "(https://diabetesjournals.org/care/article/40/12/1631/37000/International-Consensus-on-Use-of-Continuous)"
    link += "- assesses that standardized advice about how best to use the new information that CGM data provide is lacking "
    link += "and proceeds to give a consensus recommendation to help understand how CGM results can affect outcomes. (2017)"
    st.markdown(link)

    link = "- [Statistical Tools to Analyze Continuous Glucose Monitor Data]"
    link +="(https://www.liebertpub.com/doi/abs/10.1089/dia.2008.0138) "
    link += "- describes several methods that are pertinent to the analysis of CGM data. (2009)"
    st.markdown(link)

    
    st.markdown("##### :medical_symbol: How are the metrics calculated")

    link = "- [Glycemic Variability Measures]"
    link += "(https://shiny.biostat.umn.edu/GV/README2.pdf)"
    link += "- document written to explain the R code and R shiny app with name Easy GV which is based on "
    link += "the spreadsheet developed at Oxford University by Nathan R. Hill. (2019)"
    st.markdown(link)
    
    link = "- [Mathematical Descriptions of the Glucose Control in Diabetes Therapy. Analysis of the Schlichtkrull “M”-Value]"
    link += "(https://www.thieme-connect.com/products/ejournals/abstract/10.1055/s-2007-979895)"
    link += "- (M-value) (1965)"
    st.markdown(link)

    link = "- [Mean Amplitude of Glycemic Excursions, a Measure of Diabetic Instability]"
    link += "(https://diabetesjournals.org/diabetes/article/19/9/644/3599/Mean-Amplitude-of-Glycemic-Excursions-a-Measure-of)"
    link += "- MAGE (1970)"
    st.markdown(link)

    link = "- [Day-to-day variation of continuously monitored glycaemia: A further measure of diabetic instability]"
    link += "(https://link.springer.com/article/10.1007/BF01218495)"
    link += "- Absolute Mean of Daily Differences (MODD) (1972)"
    st.markdown(link)

    link = "- [“J”-Index. A New Proposition of the Assessment of Current Glucose Control in Diabetic Patients]"
    link += "(https://www.thieme-connect.com/products/ejournals/abstract/10.1055/s-2007-979906)"
    link += "- J-Index (1995)"
    st.markdown(link)

    link = "- [Evaluation of a New Measure of Blood Glucose Variability in Diabetes ]"
    link += "(https://diabetesjournals.org/care/article/29/11/2433/24571/Evaluation-of-a-New-Measure-of-Blood-Glucose)"
    link += "- ADRR (2006)"
    st.markdown(link)




