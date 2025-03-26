from .cgm_object import CGM
import pandas as pd
import streamlit as st

class multiple_CGM(object):
    def __init__(self,names,
                 file_dfs,
                 periods,
                 dt_fmt='%Y-%m-%dT%H:%M:%S',
                 units='mg/dL',
                 time_delta=5,
                 first_full_day = False):
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
                                    time_delta=time_delta,
                                    first_full_day=first_full_day)
                
                df = pd.concat([df,self.data[name].overall_stats_dataframe()])
        my_progress_bar.empty()
        self.selected_file = self.names[0]
        self.df = df
        # cohort columns come from CGM.stats_functions dictionary
        # Lets the user choose functions that are not vectors to see correlations
        cols = self.data[name].cohort_cols
        self.stats_df=df[cols]

    def create_stats_dataframe(self,units='mg'):
        names = self.names
        stats_df = pd.DataFrame()
        for i in range(len(names)):
            name = names[i]
            stats_df = pd.concat([stats_df,self.data[name].overall_stats_dataframe(units)])
        return stats_df
            
    
    def ambulatory_glucose_profile(self,name,units = 'mg'):
        st.pyplot(self.data[name].plot_agp())
        st.divider()
        st.write(self.data[name].overall_stats_dataframe(units))
        st.divider()
        daily = st.checkbox(label="Display daily statistics",value=False)
        if daily:
            st.write(self.data[name].stats_by_day(units))
        else:
            st.write("Daily stats may take time to calculate depending on the number of days.")
        self.selected_file = name

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
        st.write(self.data[name].periods)
        #st.divider()
        #st.write(self.data[name].stats)
        self.selected_file = name
        
    def view_gri(self,name):
        st.pyplot(self.data[name].plot_gri())
        self.selected_file = name

    def time_in_range_report(self,name):
        self.data[name].time_in_range_report()

    def visualize_data(self,name):
        options = ['Poincare Plot','Time Series']
        tab1,tab2 = st.tabs(options)
        with tab1:
            #options = [td for td in range(self.time_delta,12*self.time_delta+1,self.time_delta)]
            shift_minutes = st.number_input("Time between observations",min_value = self.time_delta,
                            max_value = self.time_delta*12,step = self.time_delta)
            fig=self.data[name].poincare_plot(shift_minutes)
            st.pyplot(fig)
        with tab2:
            impute = st.checkbox("Include Imputed Data.",
                        value = False,)
            fig = self.data[name].time_series_plot(impute=impute)
            st.pyplot(fig)

    def markov_analysis(self,name):
        int1 = st.sidebar.slider("Interval 1",
                          min_value=0,
                          max_value=70,
                          value = 54,
                          step=1)
        int2 = st.sidebar.slider("Interval 2",
                            min_value=int1,
                            max_value=180,
                            value = 70,
                            step=1)
        
        int3 = st.sidebar.slider("Interval 3",
                            min_value=int2,
                            max_value=250,
                            value = 180,
                            step=1)
        
        int4 = st.sidebar.slider("Interval 4",
                            min_value=int3,
                            max_value=300,
                            value = 250,
                            step=1)
        self.data[name].markov_chain_calculation([int1,int2,int3,int4])

        
    def export_data(self,filename,units):
        #df = self.df
        df = self.create_stats_dataframe(units=units)
        df['idx']=self.names
        df.set_index('idx',inplace=True)
        st.write(df)
        st.sidebar.download_button(label="Download csv",
                           data = df.to_csv().encode('utf-8'),
                           file_name=filename)
        
    def test_develop(self,name):
        """
        Using the test_develop method - allows for development of functions in streamlit
        """
        self.markov_analysis(name)

        
        #st.pyplot(self.data[name].time_series_plot(True))
        
        