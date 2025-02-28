import numpy as np
import pandas as pd
#import streamlit as st
from ..utils.difference import difference, difference_m
from ..utils.helper import unique_days
from ..utils.read_data import return_data, all_day_data, return_period_data

def glucose_mean(**kwargs):
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    return data.mean()

def glucose_median(**kwargs):
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    return data.median()

def glucose_std(**kwargs):
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    return data.std()
def glucose_cv(**kwargs):
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    return data.std()/data.mean()

def total_time(**kwargs):
    type_ = kwargs['time_type']
    if type_ == 'daily':
        data = kwargs['data']
        data = data[data.notnull().values]
        return data.index[-1]-data.index[0]
    return kwargs['total_time']

def glucose_N(**kwargs):
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    return len(data.dropna())

def percent_active(**kwargs):
    return 1-kwargs['nans']/kwargs['N']

def time_in_range(**kwargs):
    """
    time in range - assumes equal intervals for all data and simply
        counts the number between lower and upper / total number of 
        observations.
    Input:  data - needs to be a series object with either datetime or 
        minutes after midnight as the index.
            lower - the lower bound to check
            upper - the upper bound to check
    Output: a tuple of floats that represent the 
        (%time below range, %time in range, %time above range)
        
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    lowest = kwargs['lowest']
    lower = kwargs['lower']
    upper = kwargs['upper']
    highest = kwargs['highest']
    data = data[data.notnull().values]
    denom = len(data)
    
    below_54 = len(data[data<lowest])
    below_range = len(data[data<lower])-below_54
    in_range = len(data[(data>=lower) & (data<=upper)])
    above_250 = len(data[data>highest])
    above_range = len(data[data>upper])-above_250
    
    res = np.array([below_54,below_range,in_range,above_range,above_250])
    return res/denom

def calculate_time_in_range(**kwargs):
    """
    This function is specifically for time in range calculations. 
    
    Output: dictonary of range times and corresponding minutes
    """
    data = kwargs['data']; data = data[data.notnull().values]
    day_data = kwargs['day_data']; day_data[day_data.notnull().values]
    night_data = kwargs['night_data']; night_data[night_data.notnull().values]
    time_delta = kwargs['time_delta']
    total_time = len(data)*time_delta
    day_time = len(day_data)*time_delta
    night_time = len(night_data)*time_delta

    
    tir_data = {}
    def tir(td,vals):
        if vals[1]>500:
            return len(td[td>=vals[0]])
        elif vals[0]<10:
            return len(td[td<=vals[1]])
        else:
            return len(td[(td>=vals[0]) & (td<=vals[1])])
    
    ### <54 mg/dL
    tir_data['<54mg/dL'] = {}
    tir_data['<54mg/dL']['all'] = tir(data,[0,53])*time_delta
    tir_data['<54mg/dL']['%all'] =tir_data['<54mg/dL']['all']/total_time
    tir_data['<54mg/dL']['day'] = tir(day_data,[0,53])*time_delta
    tir_data['<54mg/dL']['%day'] = tir_data['<54mg/dL']['day']/day_time
    tir_data['<54mg/dL']['night']=tir(night_data,[0,53])*time_delta
    tir_data['<54mg/dL']['%night']=tir_data['<54mg/dL']['night']/night_time
    
    ### <70 mg/dL
    tir_data['<70mg/dL'] = {}
    tir_data['<70mg/dL']['all'] = tir(data,[0,69])*time_delta
    tir_data['<70mg/dL']['%all'] = tir_data['<70mg/dL']['all']/total_time
    tir_data['<70mg/dL']['day'] = tir(day_data,[0,69])*time_delta
    tir_data['<70mg/dL']['%day'] = tir_data['<70mg/dL']['day']/day_time
    tir_data['<70mg/dL']['night']=tir(night_data,[0,69])*time_delta
    tir_data['<70mg/dL']['%night'] = tir_data['<70mg/dL']['night']/night_time


    ### 54-69 mg/dL
    tir_data['54-69mg/dL'] = {}
    tir_data['54-69mg/dL']['all'] = tir(data,[54,69])*time_delta
    tir_data['54-69mg/dL']['%all'] = tir_data['54-69mg/dL']['all']/total_time
    tir_data['54-69mg/dL']['day'] = tir(day_data,[54,69])*time_delta
    tir_data['54-69mg/dL']['%day'] = tir_data['54-69mg/dL']['day']/day_time
    tir_data['54-69mg/dL']['night']=tir(night_data,[54,69])*time_delta
    tir_data['54-69mg/dL']['%night']=tir_data['54-69mg/dL']['night']/night_time

    ### 70-180 mg/dL
    tir_data['70-180mg/dL'] = {}
    tir_data['70-180mg/dL']['all'] = tir(data,[70,180])*time_delta
    tir_data['70-180mg/dL']['%all'] = tir_data['70-180mg/dL']['all']/total_time
    tir_data['70-180mg/dL']['day'] = tir(day_data,[70,180])*time_delta
    tir_data['70-180mg/dL']['%day'] = tir_data['70-180mg/dL']['day']/day_time
    tir_data['70-180mg/dL']['night']=tir(night_data,[70,180])*time_delta
    tir_data['70-180mg/dL']['%night']=tir_data['70-180mg/dL']['night']/night_time

    ### 70-140 mg/dL
    tir_data['70-140mg/dL'] = {}
    tir_data['70-140mg/dL']['all'] = tir(data,[70,140])*time_delta
    tir_data['70-140mg/dL']['%all']=tir_data['70-140mg/dL']['all']/total_time
    tir_data['70-140mg/dL']['day'] = tir(day_data,[70,140])*time_delta
    tir_data['70-140mg/dL']['%day']=tir_data['70-140mg/dL']['day']/day_time
    tir_data['70-140mg/dL']['night']=tir(night_data,[70,140])*time_delta
    tir_data['70-140mg/dL']['%night']=tir_data['70-140mg/dL']['night']/night_time

    ### 70-120 mg/dL
    tir_data['70-120mg/dL'] = {}
    tir_data['70-120mg/dL']['all'] = tir(data,[70,120])*time_delta
    tir_data['70-120mg/dL']['%all']=tir_data['70-120mg/dL']['all']/total_time
    tir_data['70-120mg/dL']['day'] = tir(day_data,[70,120])*time_delta
    tir_data['70-120mg/dL']['%day']=tir_data['70-120mg/dL']['day']/day_time
    tir_data['70-120mg/dL']['night']=tir(night_data,[70,120])*time_delta
    tir_data['70-120mg/dL']['%night']=tir_data['70-120mg/dL']['night']/night_time

    ### >120 mg/dL
    tir_data['>120mg/dL'] = {}
    tir_data['>120mg/dL']['all'] = tir(data,[121,1000])*time_delta
    tir_data['>120mg/dL']['%all']=tir_data['>120mg/dL']['all']/total_time
    tir_data['>120mg/dL']['day'] = tir(day_data,[121,1000])*time_delta
    tir_data['>120mg/dL']['%day']=tir_data['>120mg/dL']['day']/day_time
    tir_data['>120mg/dL']['night']=tir(night_data,[121,1000])*time_delta
    tir_data['>120mg/dL']['%night']=tir_data['>120mg/dL']['night']/night_time

    ### >140 mg/dL
    tir_data['>140mg/dL'] = {}
    tir_data['>140mg/dL']['all'] = tir(data,[141,1000])*time_delta
    tir_data['>140mg/dL']['%all']=tir_data['>140mg/dL']['all']/total_time 
    tir_data['>140mg/dL']['day'] = tir(day_data,[141,1000])*time_delta
    tir_data['>140mg/dL']['%day']=tir_data['>140mg/dL']['day']/day_time
    tir_data['>140mg/dL']['night']=tir(night_data,[141,1000])*time_delta
    tir_data['>140mg/dL']['%night']=tir_data['>140mg/dL']['night']/night_time

    ### >180 mg/dL
    tir_data['>180mg/dL'] = {}
    tir_data['>180mg/dL']['all'] = tir(data,[181,1000])*time_delta
    tir_data['>180mg/dL']['%all']=tir_data['>180mg/dL']['all']/total_time
    tir_data['>180mg/dL']['day'] = tir(day_data,[181,1000])*time_delta
    tir_data['>180mg/dL']['%day'] = tir_data['>180mg/dL']['day']/day_time
    tir_data['>180mg/dL']['night']=tir(night_data,[181,1000])*time_delta
    tir_data['>180mg/dL']['%night']=tir_data['>180mg/dL']['night']/night_time

    ### 180-250 mg/dL
    tir_data['181-250mg/dL'] = {}
    tir_data['181-250mg/dL']['all'] = tir(data,[181,250])*time_delta
    tir_data['181-250mg/dL']['%all']=tir_data['181-250mg/dL']['all']/total_time
    tir_data['181-250mg/dL']['day'] = tir(day_data,[181,250])*time_delta
    tir_data['181-250mg/dL']['%day'] = tir_data['181-250mg/dL']['day']/day_time
    tir_data['181-250mg/dL']['night']=tir(night_data,[181,250])*time_delta
    tir_data['181-250mg/dL']['%night']=tir_data['181-250mg/dL']['night']/night_time

    ### >250 mg/dL
    tir_data['>250mg/dL'] = {}
    tir_data['>250mg/dL']['all'] = tir(data,[251,1000])*time_delta
    tir_data['>250mg/dL']['%all']=tir_data['>250mg/dL']['all']/total_time
    tir_data['>250mg/dL']['day'] = tir(day_data,[251,1000])*time_delta
    tir_data['>250mg/dL']['%day']=tir_data['>250mg/dL']['day']/day_time
    tir_data['>250mg/dL']['night']=tir(night_data,[251,1000])*time_delta
    tir_data['>250mg/dL']['%night']=tir_data['>250mg/dL']['night']/night_time

    return tir_data


def conga(**kwargs):
    """
    conga - continuous overall net glycemic action (CONGA) McDonnell paper 
            and updated by Olawsky paper
            
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            h - number of hours to shift
            type_ - "paper" or "easy" easy is the update.
            
    Output: CONGA(h) as a float.
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    type_ = kwargs['type']
    h = kwargs['conga_h']
    time_delta = kwargs['time_delta']
    sample_rate = time_delta #in minutes      
    samples_per_hour = 60//sample_rate
    ## shift method moves values back and forward
    ## line1 moves the values an hour ahead back so 
    ## they can be used below
    line1 = data.shift(-samples_per_hour*h).dropna()
    delta = difference(data.dropna(),h)
    if type_ == 'paper':
        congah = delta.std()
        return congah
    if type_ == 'easy':
        k = len(delta)
        d_star = (abs(delta)).sum()/k
        congah = np.sqrt(((line1-d_star)**2).sum()/(k-1))
        return congah
    return None

def lability_index(**kwargs):
    """
    lability_index - for glucose measurement at time Xt, Dt = difference
        of glucose measurement k minutes prior.
    Input:  data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            k - length of time in minutes (5 min increments) used to find patterns
    Output: LI as a float.
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    k = kwargs['li_k']
    Dt = difference_m(data,k)
    try: #if there are too few data values for the data given
        li = (Dt**2).sum()/(len(Dt))
    except:
        li = np.nan
    return li
    
def mean_absolute_glucose(**kwargs):
    """
    mean_absolute_glucose - Hermanides (2009) paper
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
    Output: MAG as a float.
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    total_hours = (data.index[-1]-data.index[0]).total_seconds()/3600
    data = data[~data.isnull().values].values
    diff = np.abs(data[1:]-data[:-1])
    return diff.sum()/(total_hours)

def glycemic_variability_percentage(**kwargs):
    """
    glycemic_variability_percentage - Peyser paper length of curve / length
                    straight line with no movement (time[final]-time[initial])
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
    Output: GVP as a float percentage.
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    data = data[~data.isnull().values]
    time = data.index
    data = data.values
    t2 = [((time[i+1]-time[i]).total_seconds()/60)**2 for i in range(len(time)-1)] 
    y2 = (data[1:]-data[:-1])**2
    seg = np.array(t2+y2)
    L = np.array([np.sqrt(a) for a in seg]).sum()
    L0 = np.sqrt(t2).sum()
    return (L/L0-1)*100

def j_index(**kwargs):
    """
    j_index - calculates J-index 

    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
    Output: J-index as a float.
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    unit = kwargs['unit']
    if unit == 'mg':
        return (data.mean()+data.std())**2/1000
    if unit =="mmol":
        return (18**2)*(data.mean()+data.std())**2/1000
    return None

def low_high_blood_glucose_index(**kwargs):
    """
    low_high_blood_glucose_index - calculates the blood glucose index 
                with three sets of indices.
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            type_- "paper", "easy", or "update" Default = "paper"
            unit - "mg" or "mmol" Default: "mg"
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    type_ = kwargs['type']
    unit = kwargs['unit']
    n = len(data)
    data = data[~data.isnull().values].values
    f = 1
    c = 1
    if unit == 'mg':
        f = 1.509*(np.log(data)**1.084-5.381)
    if unit == 'mmol':
        f = 1.509*(np.log(18*data)**1.084-5.381)
    if type_ == 'update':
        c = 22.77
    if type_ == 'paper':
        c = 10
    if type_ == 'easy':
        c = 10
    rl = np.array([c*r**2 if r<0 else 0 for r in f])
    rh = np.array([c*r**2 if r>0 else 0 for r in f])
    if type_ != 'easy':
        nl = n
        nh = n
    else:
        nl=(rl>0).sum()
        nh=(rh>0).sum()
    return rl.sum()/nl, rh.sum()/nh

def glycemic_risk_assessment_diabetes_equation(**kwargs):
    """
    GRADE - or glycemic risk assessment diabetes equation

    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            type_ - "paper" or "easy" Default: "paper"
            unit - "mg" or "mmol" Default: "mg"

    Output: GRADE as 4 numbers ================================= 
            (1) GRADE or mean/median of conversion, 
            (2) GRADE for values < 70.2(mg) or 3.9(mmol), 
            (3) GRADE for values between 70.2(mg) or 3.9(mmol) 
                                    and 140.4(mg) or 7.8(mmol),
            (4) GRADE for values above 140.4(mg) or 7.8(mmol)
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    type_ = kwargs['type']
    unit = kwargs['unit']
    g = data[~data.isnull().values].values
    c1,c2 = 3.9,7.8
    if unit == 'mg':
        g = g/18
    if type_=='paper':
        c = 0.16
    if type_ == 'easy':
        c = 0.15554147
    h_log = lambda x,c: 425*((np.log10(np.log10(x))+c)**2)
    h_min = lambda x: x*(x<50)+50*(x>=50)
    h = lambda x,c: h_min(h_log(x,c))
    h_i = h(g,c)

    # separate glucose values into categories based on value
    gl = g[g<c1]
    gm = g[(c1<g)&(g<c2)]
    gh = g[g>c2]

    # run each group of glucose values through the functions
    hl = h(gl,c)
    hm = h(gm,c)
    hh = h(gh,c)
    h_sum = h_i.sum()
    if type_ == 'easy':
        grade = np.median(h_i)
    if type_ == 'paper':
        grade = h_i.mean()
    ans = np.array([grade,hl.sum()/h_sum,hm.sum()/h_sum,hh.sum()/h_sum])
    return ans

def mean_amplitude_of_glycemic_excursions(**kwargs):
    """
    MAGE (Olawsky 2019)
    mean_amplitude_of_glycemic_excursions - MAGE mean of differences that are
        large compared to daily value.
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    data = data[~data.isnull().values]
    days = kwargs['days']
    time_delta = kwargs['time_delta']
    E = []
    for day in days:
        # g - glucose values for day=day
        g = all_day_data(day,**kwargs)
        # s - standard deviation for glucose on day=day
        s = g.std()
        # D - glucose values differenced (5 minutes)
        D = difference_m(g,time_delta)
        # test if abs(d) > standard deviation for the day
        for d in D:
            if abs(d)>s:
                E.append(d)
    ## Use numpy array to sort / find mean of data
    if len(E)>0:
        E = np.array(E)
        mage_plus = E[E>0].mean()
        mage_minus = E[E<0].mean()
    else:
        mage_plus = mage_minus = np.nan
    return mage_plus,mage_minus

def mean_of_daily_differences(**kwargs):
    """
    MODD - or mean of daily differences
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            type_ - algorithm to use - either "paper" or "easy" 
    Output: MODD as a float
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    type_ = kwargs['type']
    time_delta = kwargs['time_delta']

    data = data[~data.isnull().values]
    
    days = kwargs['days']
    if len(data)>1440/time_delta:
        delta = difference(data,24)
        if type_ == 'paper':
            return (abs(delta)).sum()/len(delta)
        if type_ == 'easy':
            delta = delta[delta != delta.max()]
            return (abs(delta)).sum()/(len(delta))
    else:
        return np.nan
    
def average_daily_risk_range(**kwargs):
    """
    average_daily_risk_range - adrr_paper - returns ADRR based on actual days.
        adrr_easy - returns average daily risk range as calculated using
                the algorithm from easyGV. It differs from the algorithm
                in this calculation because our datetime is used to pull 
                data from each day instead of using the first time as a 
                reference and using the next 24 hours.

    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
    Output: ADRR as three values - sum of low and high, low risk rate, high risk rate.
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    unit = kwargs['unit']
    type_ = kwargs['type']
    data = data[~data.isnull().values]
    days = unique_days(data) #use unique days in data, not days in params
    if type_=='paper':
        if len(days)>=1:
            d = 1
            if unit == 'mmol':
                d=18
            f = lambda x: 1.509*(np.log(d*x)**1.084-5.381)

            fx=f(data)
            rfl = lambda x: 10*x**2 if x<0 else 0 
            rfh = lambda x: 10*x**2 if x>0 else 0

            df = pd.DataFrame(fx.values,index=data.index,columns=['fx'])
            df['rl']=df['fx'].apply(rfl)
            df['rh']=df['fx'].apply(rfh)

            LR = np.zeros(len(days))
            HR = np.zeros(len(days))

            for i,day in enumerate(days):
                rh_data = return_data(df,'rh',day)
                rl_data = return_data(df,'rl',day)
                LR[i]=max(rl_data)
                HR[i]=max(rh_data)
            adrr_m = (LR+HR).mean()
            adrr_l = LR.mean()
            adrr_h = HR.mean()
        else:
            return np.nan
        return adrr_m,adrr_l,adrr_h
    if type_ == 'easy':
        d = 1
        if unit == 'mmol':
            d=18
        f = lambda x: 1.509*(np.log(d*x)**1.084-5.381)
        
        fx=f(data)
        rfl = lambda x: 10*x**2 if x<0 else 0 
        rfh = lambda x: 10*x**2 if x>0 else 0
        df = pd.DataFrame(fx.values,index=data.index,columns=['fx'])
        df['rl']=df['fx'].apply(rfl)
        df['rh']=df['fx'].apply(rfh)
        LR = []
        HR = []
        for i in range(len(data)//288+1):
            d = df.iloc[i*288:(i+1)*288]
            LR.append(d['rl'].max())
            HR.append(d['rh'].max())
        LR = np.array(LR)
        HR = np.array(HR)
        return (LR+HR).mean(),LR.mean(),HR.mean()

  

def m_value(**kwargs):
    """
    m_value - calculates the M-value for a glucose 
                time series. 
    Input: data - pandas Series with index as a datetime,
                    values are glucose 
                    readings associated with those times.
            type_ - calculates either the algorithm 
                    from the "paper" or "easy"
            index - the value used as the index, 
                    default is 120
            unit - "mg" for milligrams per deciliter 
                    or "mmol" for milimoles per
                    liter. Default is "mg".
    Output:
        M-value as a float or None if type_ is not correct.

    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    type_ = kwargs['type']
    unit = kwargs['unit']
    index = kwargs['m_index']
    data = data[~data.isnull().values]
    if unit == 'mmol':
        data = 18*data
    m_star_abs = np.abs((10*np.log10(data/index))**3)
    w = (data.max()-data.min())/20
    if type_=='paper':
        return m_star_abs.mean()+w
    if type_=='easy':
        return m_star_abs.mean()
    return None

def glucose_management_indicator(**kwargs):
    """
    glucose_management_indicator - Bergenstal (2018), formerly 
        referred to as eA1C, or estimated A1C which is a measure 
        converting mean glucose from CGM data to an estimated 
        A1C using data from a population and regression.
        
    Input: data - pandas Series with index as a datetime, 
            values are glucose readings associated with those times.
            unit - "mg" for milligrams per deciliter or "mmol" 
            for milimoles per
                    liter. Default is "mg".
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    unit = kwargs['unit']
    data = data[~data.isnull().values]
    if unit == 'mmol':
        data = 18*data
    if unit == 'mg':
        return 3.31+0.02392*data.mean()
    return None

def interquartile_range(**kwargs):
    """
    IQR - inter-quartile range 75th percentile - 25th percentile. 
        Danne (2017) had this calculation in one of the figures. 
    """
    calc = kwargs['calculation']
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']
    unit = kwargs["unit"]
    data = data[~data.isnull().values]
    if unit == 'mmol':
        data = 18*data
    q75,q25 = np.percentile(data.values,[75,25])
    return q75-q25

def glycemic_risk_index(**kwargs):
    """
    Glycemic Risk Indicator - (Klonoff 2023)
        This risk index is a three number and letter result which represents a composite metric for
        the quality of the glycemia from a CGM. 
        
    Input - time in range vector representing [x1,x2,n,y2,y1] the percent time in each category
            [g<54,54<=g<70,70<=g<180,180<g<250,g>250]
    """
    #data = kwargs['data']
    tir = time_in_range(**kwargs)
    tir = np.round(tir*100,1)
    x1,x2,_,y2,y1 = tir
    f = lambda x1,x2:x1+0.8*x2
    g = lambda y1,y2:y1+0.5*y2
    h = lambda x1,x2,y1,y2: 3*f(x1,x2)+1.6*g(y1,y2)
    x = f(x1,x2)
    y = g(y1,y2)
    gri = h(x1,x2,y1,y2)
    
    return gri,x,y

def auc(**kwargs):
    """
    area under the curve with no threshold

    Input: data - pandas series with inde as a datetime and values as glucose
                  readings associated with those times.
    """
    type_=kwargs['type']
    calc = kwargs['calculation']

    if 'wake' in type_:
        calc='day'
    if 'sleep' in type_:
        calc = 'night'
    if 'all' in 'type':
        calc = 'all'
    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']

    data = data[~data.isnull().values]
    ans = 0
    total_hours = 0
    for i in range(len(data)-1):
        d1 = data.iloc[i]
        d2 = data.iloc[i+1]
        # convert time to hours
        dt = (data.index[i+1]-data.index[i]).seconds/3600
        total_hours += dt
        ans += (d1+d2)/2*dt
    ## glucose*hour / day ... the denominator here is number of days in total_minutes
    ## total minutes * 1 hour/60 minutes * 1 day/24 hours => minutes & hours
    return int(ans/(total_hours/(24)))

def auc_thresh(**kwargs):
    """
    auc - area under the curve with a threshold value - converts to a 
        glucose*min/day value for area above or below the threshold value.
        
    Input: data - pandas Series with index as a datetime, values are glucose 
                    readings associated with those times.
            thresh - a value for the threshold for the calculation above or below the val.
                    default value = 100
            above - boolean to calculate above or below the threshold.
    """
    type_=kwargs['type']
    calc = kwargs['calculation']
    thresh = kwargs['thresh']
    above = kwargs['above']

    if 'wake' in type_:
        calc='day'
    if 'sleep' in type_:
        calc = 'night'
    if 'all' in 'type':
        calc = 'all'
    if '+' in type_:
        above=True
    if '-' in type_:
        above=False  

    if calc == 'all':
        data = kwargs['data']
    elif calc == 'day':
        data = kwargs['day_data']
    elif calc == 'night':
        data = kwargs['night_data']

    
    data = data[~data.isnull().values]
    ans = 0
    total_hours = 0
    if above:
        for i in range(len(data)-1):
            d1 = data.iloc[i]
            d2 = data.iloc[i+1]
            if d1>=thresh and d2>=thresh:
                ## dt in minutes
                dt = (data.index[i+1]-data.index[i]).seconds/3600
                ans += ((d1-thresh)+(d2-thresh))/2*dt
                total_hours+=dt
                ## from paper, overly complicated is equivalent to above eqn
                #ans2 += ((min(d1-thresh,d2-thresh)*dt)+abs(d2-d1)*(dt/2))
        if total_hours == 0:
            return 0
        return int(ans/(total_hours/(24)))
    else:
        for i in range(len(data)-1):
            d1 = data.iloc[i]
            d2 = data.iloc[i+1]
            if d1<=thresh and d2<=thresh:
                dt = (data.index[i+1]-data.index[i]).seconds/3600
                ans += ((thresh-d1)+(thresh-d2))/2*dt
                total_hours+=dt
        if total_hours == 0:
            return 0
        return int(ans/(total_hours/(24)))

#### Needs updating ###############################################
def auc_5(day,**kwargs):
    """
    this function is used to calculate auc above thresh for multi-day
        times that are not continuous (i.e. for times between midnight and 6am)
        and assumes 5 minutes between all times.
        
    thresh - is the value that is currently in the object for the threshold value
    """
    thresh = kwargs['thresh']
    time_delta = kwargs['time_delta']

    if day=='day':
        data = kwargs['day_data']
    elif day == "night":
        data = kwargs['night_data']
    elif day == "all":
        data = kwargs['data']
    days = kwargs['days']
    data=data[~data.isnull().values].values
    total_minutes = time_delta*len(data)
    ans = 0
    for i in range(len(data)-1):
        d1 = data[i]
        d2 = data[i+1]
        if d1>=thresh and d2>=thresh:
            dt=time_delta
            ans+=((d1-thresh)+(d2-thresh))/2 * dt
    return int((ans/60)/len(days))
