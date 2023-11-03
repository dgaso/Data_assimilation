# -*- coding: utf-8 -*-
# Copyright (c) 2021 Wageningen-UR
# Deborah Gaso Melgar and Allard de Wit, Febrary 2021

#from: https://github.com/ajwdewit/pcse_notebooks/blob/master/08_data_assimilation_with_the_EnKF.ipynb
import sys, os.path
this_dir = os.getcwd()
up_dir = os.path.dirname(this_dir)
if not up_dir in sys.path:
    sys.path.append(up_dir)

import copy
import datetime as dt
import pcse
from pcse.engine import Engine
from pcse.base import ParameterProvider
from pcse.fileinput import ExcelWeatherDataProvider, PCSEFileReader

from runfiles import config
from campbell_diaz.model import CampbellDiazModel

import yaml
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib
matplotlib.style.use("ggplot")
import matplotlib.pyplot as plt
print("This notebook was built with:")
print("python version: %s " % sys.version)
print("PCSE version: %s" %  pcse.__version__)
import rasterio

##Runs EnKF with multiple variables: LAI, TDM and yield###

def read_sentinel2(col, row):
    #IMP:images before sowing or after harvest give error in fun run_one_pixel
             
    variable = 'ci'
    flist = sorted([fname for fname in os.listdir(config.s2_dir)
                    if fname.endswith('.tif') and variable in fname])
    ci = []
    for fname in flist:
        im = rasterio.open(os.path.join(config.s2_dir, fname))
        imarray = im.read(1)
        ci.append(imarray)
    ci = np.array(ci)

    lai= ci ** config.CI_coefficient / config.CI_offset
    LAItimeseries = lai[:, row, col]
    STDtimeseries = LAItimeseries * 0.1
    
    dates=[]
    for i in flist:
        #date_str=(i[7:11]) + '/' + (i[11:13]) + '/' + (i[13:15]) #img from ESA
        date_str=(i[0:4]) + '/' + (i[4:6]) + '/' + (i[6:8]) #img from GEE
        format_str = '%Y/%m/%d' # The format
        datetime_obj = dt.datetime.strptime(date_str, format_str)
        d=(datetime_obj.date())        
        dates.append(d)

    observations_for_DA=[]
    for d, lai, errlai in zip(dates, LAItimeseries, STDtimeseries):
        observations_for_DA.append((d, {"LAI":(lai, errlai)}))
    
    return observations_for_DA, dates, LAItimeseries, STDtimeseries

def read_polaris(col,row):
   
    flist = sorted([fname for fname in os.listdir(config.polaris_dir)
                    if fname.endswith('.tif') ])        
    soil_map = []
    for fname in flist:
        im = rasterio.open(os.path.join(config.polaris_dir, fname))
        imarray = im.read(1)
        soil_map.append(imarray)
    soil_map = np.array(soil_map)

    return soil_map

def make_agromanagement(year):
    """Returns the proper crop agromanagement dates for given campaign year

    :param year: The harvest year of the cropping season
    :return: the PCSE agromanagement structure
    """
    campaign_dates = config.agromanagement[year]
    agromanagement = """
    - {campaign_start_date}:
        CropCalendar:
            crop_name: Soybean 
            variety_name: Soybean 
            crop_start_date: {crop_start_date}
            crop_start_type: sowing
            crop_end_date: {crop_end_date}
            crop_end_type: harvest
            max_duration: 300
        TimedEvents:
        StateEvents:
        """
    agro = yaml.safe_load(agromanagement.format(**campaign_dates))
    return agro

def run_model(year,col,row):
    
    agro=make_agromanagement(year)

    #check model
    # Weather data for Uruguay
    wdp = ExcelWeatherDataProvider(config.weather_fname)
    
    # Read soil map and override params:field capacity and wilting point
    soil_map=read_polaris(col, row)
    
    ## Parameter sets for crop, soil and site
    ## Model parameters
    cropd = PCSEFileReader(config.crop_fname)    
    soild = PCSEFileReader(config.soil_fname)
    soild.update({"FCP":soil_map[0,row,col], "PWPP":soil_map[1,row,col] })    
    params = ParameterProvider(cropdata=cropd, soildata=soild,sitedata={})
    model = CampbellDiazModel(params, wdp, agro)
    model.run_till_terminate()
    output=model.get_output()
    df = pd.DataFrame(model.get_output()).set_index("day")
#    df
   
    # Size of the ensemble    
    np.random.seed(10000)
    
    # A container for the parameters that we will override
    override_parameters = {}
    
    override_parameters["WUE"] = np.random.normal(4,0.3 ,(config.ensemble_size)) #0.2
    override_parameters["RDMAX"] = np.random.normal(1.2,0.1 ,(config.ensemble_size)) #0.1
    override_parameters["FNTR"] = np.random.normal(3,0.3 ,(config.ensemble_size)) #0.3
    override_parameters["initLAI"] = np.random.normal(0.15,0.04 ,(config.ensemble_size))#0.04
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    # Show the sample that was drawn
    for ax, (par, distr) in zip(axes.flatten(), override_parameters.items()):
        ax.hist(distr)
        ax.set_title(par)

    #Initializing the ensemble
    ensemble = []
    for i in range(config.ensemble_size):
        p = copy.deepcopy(params)
        for par, distr in override_parameters.items():
            p.set_override(par, distr[i])
        member = CampbellDiazModel(p, wdp, agro)
        ensemble.append(member)
            
    return ensemble

    
def run_EnKF(year, col, row, silent=False):
    
    observations_for_DA, dates, LAItimeseries, STDtimeseries=read_sentinel2(col, row)
    ensemble=run_model(year,col,row)
     
    variables_for_DA = ["LAI","TSTEM","TSEED"]
    observed_states = ["LAI"]
 
   
    #Implementing the data assimilation itself
    show_output = False
    
    for i in range(len(observations_for_DA)):
        
        #2.4.1 Move the ensemble forward in time until the first analysis step
        day, obs = observations_for_DA.pop(0)
#        print(day,obs)
#        print("day", day)
#        print ("obs", obs)
        for member in ensemble:
            member.run_till(day)
        #print("%s observations left!" % len(observations_for_DA))
            
        #Retrieve the states from the ensemble
        collected_states = []
        for member in ensemble:
            t = {}
            for state in variables_for_DA:
                t[state] = member.get_variable(state)
            collected_states.append(t)
        df_A = pd.DataFrame(collected_states)
        #print("collected_states", collected_states)
        
        A = np.matrix(df_A).T
        #df_A if show_output else None
         
        #Compute covariance within ensemble states
        P_e = np.matrix(df_A.cov())
        #print("Pe", P_e)
        #df_A.cov() if show_output else None
        #import pdb; pdb.set_trace()
        
        #Calculate perturbed observations and their mean and covariance
        perturbed_obs = []
        observations = ["LAI"]
        for state in observed_states:
            (value, std) = obs[state]
            #print("v",value,std)
            d = np.random.normal(value, std, (config.ensemble_size))
            perturbed_obs.append(d)
        df_perturbed_obs = pd.DataFrame(perturbed_obs).T
        df_perturbed_obs.columns = observations
        D = np.matrix(df_perturbed_obs).T
        R_e = np.matrix(df_perturbed_obs.cov())
        #df_perturbed_obs if show_output else None    
         
        #Apply the Kalman filter equations
        # Here we compute the Kalman gain
        #H = np.identity(len(obs))
        H = np.matrix([1.,0.,0.])
        K1 = P_e * (H.T)
        K2 = (H * P_e) * H.T
        K = K1 * ((K2 + R_e).I)
        #K if show_output else None   
        
        # Here we compute the analysed states
        Aa = A + K * (D - (H * A))
        df_Aa = pd.DataFrame(Aa.T, columns=variables_for_DA)
        #df_Aa if show_output else None    
        
        valueSTEM = df_A['TSTEM']
        valueSEED = df_A['TSEED']

        #2.4.6. Update the ensemble to reflect the new state
        for member, new_states in zip(ensemble, df_Aa.itertuples()):
            #if np.isnan(new_states.LAI):
                #import pdb; pdb.set_trace()
            member.set_variable("LAI", new_states.LAI)
            member.set_variable("TSTEM", new_states.TSTEM)
            member.set_variable("TSEED", new_states.TSEED)
                        
#        if i==7:            
#            import pdb; pdb.set_trace()
             
    #2.4.7 Repeat steps 2.4.1 to 2.4.6 for the next analysis step (observation)
    # IMP!!! We now need to repeat the above steps for each observation available and until the cell under 2.4.1 reports 0 observations left!
            
    #4.8 Run all the ensemble members until the end of the growing season
    for member in ensemble:
        member.run_till_terminate()    
        
    # 3. Visualizing output from the ensemble with the EnKF
    #    3.1 Retrieve output from the ensemble
        
    results = []
    for member in ensemble:
        member_df = pd.DataFrame(member.get_output()).set_index("day")
        results.append(member_df)
        
        collected_yield=[]
        for i in range (len(results)):
            harvested_yield=results[i]["YIELD"][results[i].index[-1]]
            collected_yield.append(harvested_yield)
    
    #3.2 Visualize ensemble with EnKF        
    fig, ax = plt.subplots(figsize=(10,10))
    for member_df in results:
        member_df["TSTEM"].plot(style="k:", ax=ax)  
    ax.set_title("TSTEM (kg m-2)")
    fig.autofmt_xdate()  

    fig, ax = plt.subplots(figsize=(10,10))
    for member_df in results:
        member_df["TSEED"].plot(style="k:", ax=ax)   
    ax.set_title("TSEED (kg m-2)")
    fig.autofmt_xdate()  

    fig, ax = plt.subplots(figsize=(10,10))
    for member_df in results:
        member_df["YIELD"].plot(style="k:", ax=ax)   
    ax.set_title("YIELD (kg m-2)")
    fig.autofmt_xdate()  

    fig, ax = plt.subplots(figsize=(10,10))
    for member_df in results:
        member_df["LAI"].plot(style="k:", ax=ax)  
    ax.errorbar(dates, LAItimeseries, yerr= STDtimeseries, fmt="o")
    ax.set_title("LAI (m2.m2)")
    fig.autofmt_xdate()  
    
    return collected_yield 

if __name__ == "__main__":

    collected_yield=run_EnKF(year=2020, col=14, row=20)
    


   


