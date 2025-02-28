from .cgm_object import CGM
import pandas as pd
import streamlit as st

class multiple_CGM(object):
    def __init__(self,names,
                 file_dfs,
                 periods,
                 dt_fmt='%Y-%m-%dT%H:%M:%S',
                 units='mg/dL',
                 time_delta=5):
        self.names = names
        self.files = file_dfs
        self.periods = periods
        self.units = units
        self.time_delta = time_delta

        self.data={}
        df = pd.DataFrame()
        progress_text = ""
        my_progress_bar = st.progress(0,text=progress_text)
        with st.status("Calculating Statistics"):
            for i in range(len(names)):
                my_progress_bar.progress((i+1)/len(names),text=progress_text)
                name = names[i]
                file_df = file_dfs[i]
                periods_ = periods[i]
                st.write(name)
                self.data[name]=CGM(filename=name,
                                    file_df=file_df,
                                    periods=periods_,
                                    dt_fmt=dt_fmt,
                                    units=units,
                                    time_delta=time_delta)
                
                df = pd.concat([df,self.data[name].overall_stats_dataframe()])
        my_progress_bar.empty()
        self.selected_file = self.names[0]
        self.df = df
        # cohort columns come from CGM.stats_functions dictionary
        # Lets the user choose functions that are not vectors to see correlations
        cols = self.data[name].cohort_cols
        self.stats_df=df[cols]
    
    

    def ambulatory_glucose_profile(self,name):
        st.pyplot(self.data[name].plot_agp())
        st.divider()
        st.write(self.data[name].overall_stats_dataframe())
        st.divider()
        daily = st.checkbox(label="Display daily statistics",value=False)
        if daily:
            st.write(self.data[name].stats_by_day())
        else:
            st.write("Daily stats may take time to calculate depending on the number of days.")
        self.selected_file = name

    def test_develop(self,name):
        self.data[name].time_in_range_report()

    def agp_report(self,name):
        options = ["Glucose Statistics and Targets",
                   "Time in Ranges",
                   "AGP",
                   "Daily Glucose Profile"]
        tab1,tab2,tab3,tab4 = st.tabs(options)
        with tab1:
            st.markdown("### :blue-background[GLUCOSE STATISTICS AND TARGETS]")
            self.data[name].plot_agp_report_stats()
        with tab2:    
            st.markdown("### :blue-background[TIME IN RANGES]")
            self.data[name].plot_agp_report()

        with tab3:
            st.markdown("### :blue-background[AMUBULATORY GLUCOSE PROFILE (AGP)]]")
            body = "AGP is a summary of glucose values from the report period, "
            body+="with median (50%) and other percentiles (75%, 95%) shown as "
            body+="if occuring in a single day."

            st.markdown(body)
            self.data[name].agp_plot_only()

        with tab4:
            st.markdown('### :blue-background[DAILY GLUCOSE PROFILES]')
            self.data[name].plot_daily_traces()
        
        self.selected_file = name
        
    def view_df_series(self,name):
        st.write(self.data[name].df)
        st.divider()
        st.write(self.data[name].series)
        st.divider()
        st.write(self.data[name].time_in_range)
        st.divider()
        st.write(self.data[name].periods)
        self.selected_file = name
        
    def view_gri(self,name):
        st.pyplot(self.data[name].plot_gri())
        self.selected_file = name

    def time_in_range_report(self,name):
        self.data[name].time_in_range_report()
        
    def export_data(self,filename):
        df = self.df
        df['idx']=self.names
        df.set_index('idx',inplace=True)
        st.write(df)
        st.sidebar.download_button(label="Download csv",
                           data = df.to_csv().encode('utf-8'),
                           file_name=filename)