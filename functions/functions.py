# -*- coding: utf-8 -*-
"""
@author: Oskar Lindberg
E-mail: oskar.lars.lindberg@gmail.com
"""
################################################################################
# Import packages and define path
################################################################################
import numpy as np
import os
import pickle 
import pandas as pd
from scipy.special import expit,logit
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from ranewable import Ra

PATH = os.getcwd()

################################################################################
# Function to load site information
################################################################################
def load_sites():
    '''
    This functions load information about the sites that is used for forecasting.
    
    Returns
    -------
    site_info : Dataframe with information about the sites.
    '''
    site_info = pd.read_excel(os.path.join(PATH, 'data', 'info_sites.xlsx'))
    site_info['features'] = site_info['features'].str.split(',')
    return site_info

################################################################################
# Function to save parameter dictionary
################################################################################
def save_params(param_dict):
    '''
    This function saves a dictionary of parameters that contains information 
    about the variables used when forecasting.
    
    Parameters
    ----------
    param_dict: A dictionary containing parameters and information to be used when forecasting.
    '''
    os.makedirs(os.path.join(PATH, 'parameters'), exist_ok=True) # If folder does not exist, create it
    pickle.dump(param_dict, open(os.path.join(PATH, 'parameters', 'parameter_dictionary.pkl'), 'wb'))

################################################################################
# Function to save parameter dictionary
################################################################################
def load_params():
    '''
    This function loads a dictionary of parameters that contains information 
    about the variables used when forecasting.
    
    Returns
    -------
    param_dict: A dictionary containing parameters and information to be used when forecasting.
    '''
    parameter_path = os.path.join(PATH, 'parameters', 'parameter_dictionary.pkl')
    with open(parameter_path, 'rb') as file:
        param_dict = pickle.load(file)
    return param_dict

################################################################################
# Function to compile production data on correct format
################################################################################
def compile_power_production(site_name):
    '''
    This function compiles nproduction values so that they are on the correct form.
    
    Parameters
    ----------
    site_name: Name of power production site.

    Returns
    -------
    df: A DataFrame with datetime index and a single aggregated power production column.

    '''
    
    df = pd.read_csv(os.path.join(PATH, 'data', 'energy_production', site_name, site_name + '.csv'), sep = '\t')
    df = df.dropna(axis=0)

    df['datetime'] = pd.to_datetime(df['datetime'], format = '%Y-%m-%d %H:%M:%S%z')

    df = df.set_index('datetime')

    df = df.sum(axis=1).to_frame() # If the park consists of several power production units (e.g., wind turbines, PV inverters etc.)
    df = df.dropna(axis=1)

    df.columns = ['production']
    
    foldername = os.path.join(PATH, 'data', 'energy_production', site_name)  
    filename = "{}_{}_{}.txt".format('compiled', 'production', site_name + '.txt')  
    
    df.to_csv(os.path.join(foldername,filename), sep = '\t')
    
    return df
    
################################################################################
# Function to compile weather data and production data
################################################################################
def compile_data(site_info):        
    '''
    This function compiles numerical weather predictors and production values that
    is used to train and test forecasting models.
    
    Parameters
    ----------
    site_info: Dataframe with information about the sites.

    Returns
    -------
    modelling_table: Dictionary of Dataframes consisting of the numerical weather predictors 
                     and production values to be used for training and testing. Each key in
                     the dictionary corresponds to a site.
    '''
    # Create a dictionary, where each key corresponds to a site in site_info.
    modelling_table = {}
    
    for _,site in site_info.iterrows():
        # Read NWP data
        foldername = os.path.join(os.getcwd(), 'data', 'nwp', 'MetNo_HIRESMEPS', site['site_name'])  
        filename = "{}_{}_{}.txt".format(site['site_name'], 'MetNo_HIRESMEPS', 'compiled')  
        nwp_data = pd.read_csv(os.path.join(foldername,filename), sep="\t", index_col = [0,1], parse_dates = [0,1])
        nwp_data = nwp_data.loc[:,site['features']] # Get the features of interest
        
        # Read production data
        foldername = os.path.join(os.getcwd(), 'data', 'energy_production', site['site_name'])  
        filename = "{}_{}_{}.txt".format('compiled', 'production', site['site_name'])  
        if os.path.isfile(os.path.join(foldername,filename)):
            energy_data = pd.read_csv(os.path.join(foldername,filename), sep="\t",index_col = [0], parse_dates = [0])
        else:
            energy_data = compile_power_production(site_name = site['site_name'])
        
        modelling_table[site['site_name']] = nwp_data.join(energy_data, on="valid_time") # Join the NWP data and production data into a single Dataframe
        modelling_table[site['site_name']].dropna(axis = 0, inplace = True) # Remove NaN
        
    return modelling_table

################################################################################
# Function to preprocess training data
################################################################################
def preprocess_training(modelling_table, site_info):
    '''
    This function preprocess the data used for training and testing (combination of 
    features and observation pairs).
    
    Parameters
    ----------
    modelling_table: Dataframe with feature variable(s) and a target variable with the name 'production'.
    site_info: Dataframe with information about the sites.

    Returns
    -------
    modelling_table: Updated 'modelling_table'

    '''
    for _, site in site_info.iterrows():
        site_name = site['site_name']
        modelling_table[site_name] = modelling_table[site_name].dropna(axis=0)
        modelling_table[site_name]['production'] = modelling_table[site_name]['production'] / site['max_power'] # Normalize with maximum power to get production between [0,1]

        # Get complete data        
        vc = modelling_table[site_name].index.get_level_values('ref_time').value_counts() # Count number of instances of each reference time
        idates = vc[vc >= 56].sort_index().index # Get issue times with complete data 
        modelling_table[site_name] = modelling_table[site_name][modelling_table[site_name].index.get_level_values('ref_time').isin(idates)] # Extract reference times that are complete
        
        # To be consistent, use 56 hourly forecasts (reference times varies between 56-59 in the database)
        new_valid_time = [pd.date_range(start=dt, periods = 56, freq='h').tolist() for dt in modelling_table[site['site_name']].index.get_level_values('ref_time').unique()]
        new_valid_time = [item for sublist in new_valid_time for item in sublist]

        new_ref_time = [[dt]*56 for dt in modelling_table[site['site_name']].index.get_level_values('ref_time').unique()]
        new_ref_time = [item for sublist in new_ref_time for item in sublist]

        new_index = pd.MultiIndex.from_arrays([new_ref_time, new_valid_time], names=('ref_time', 'valid_time'))

        # Reindex the DataFrame to fill missing datetimes and set NaN values to 0
        modelling_table[site['site_name']] = modelling_table[site['site_name']].reindex(new_index, fill_value=0)
                
        if site['type'] == 'wind':
            # Pre-process the wind power production time series using the logit function
            modelling_table[site_name]['production'] =  np.where(modelling_table[site_name]['production'] <= 0.0001, 0.0001, modelling_table[site_name]['production'])
            modelling_table[site_name]['production'] = logit(modelling_table[site_name]['production'])
            
        if site['type'] == 'solar': 
            ra =  Ra(longitude=site['longitude'],
                      latitude=site['latitude'],
                      altitude=10,
                      dc_capactity_modules=site['max_power'], # DC capacity of modules
                      dc_capactity_inverters=site['max_power_dc'], # DC capacity of inverters
                      orientation=180,
                      tilt=30)
        
            # Calculate the solar position
            solar_pos = ra.calculate_solpos(modelling_table[site_name].index.get_level_values('valid_time')) 
            
            # Only train/forecast when the sun is above the horizon
            bools = (solar_pos['zenith'] < 90).tolist()
            modelling_table[site_name] = modelling_table[site_name].loc[bools,]
            
            # Add HoD and DoY features
            modelling_table[site_name]['HourOfDay'] =  modelling_table[site_name].index.get_level_values('valid_time').hour # Get hours of day
            modelling_table[site_name]['DayOfYear'] =  modelling_table[site_name].index.get_level_values('valid_time').dayofyear # Get day of year
                    
        modelling_table[site['site_name']].sort_values([('ref_time'), ('valid_time')], inplace = True)

    return modelling_table

################################################################################
# Function to preprocess operational forecast data
################################################################################
def preprocess_features(X, site_info):
    '''
    This function preprocess the data used for testing (solely features) that is
    used to generate operational forecasts.
    
    Parameters
    ----------
    X: Feature variables (e.g., weather forecasts from a weather model).
    site_info: Dataframe with information about the sites.

    Returns
    -------
    X: Preprocessed feature variables.

    '''
    for _, site in site_info.iterrows():
        site_name = site['site_name']
        
        X[site_name] = X[site_name][:56] # To be consistent, use 56 hourly forecasts (reference times varies between 56-59 in the database)
        
        if site['type'] == 'solar':
            ra =  Ra(longitude=site['longitude'],
                      latitude=site['latitude'],
                      altitude=10,
                      dc_capactity_modules=site['max_power'], # DC capacity of modules
                      dc_capactity_inverters=site['max_power_dc'], # DC capacity of inverters
                      orientation=180,
                      tilt=30)
        
            # Calculate the solar position
            solar_pos = ra.calculate_solpos(X[site_name].index.get_level_values('valid_time')) 

            # Only forecast when the sun is above the horizon
            bools = (solar_pos['zenith'] < 90).tolist()
            X[site_name] = X[site_name].loc[bools,] 

            # Add HoD and DoY features
            X[site_name]['HourOfDay'] =  X[site_name].index.get_level_values('valid_time').hour # Get hours of day
            X[site_name]['DayOfYear'] =  X[site_name].index.get_level_values('valid_time').dayofyear # Get day of year
    return X

################################################################################
# Post process forecasts and observation pairs
################################################################################
def postprocess(df,site_info):
    '''
    This function postprocess forecasts and observations (depending on what
    is used as inputs).
    
    Parameters
    ----------
    df: Dataframe that should be post-processed (e.g., observed power production or forecasts).
    site_info: Dataframe with information about the sites (single row).

    Returns
    -------
    df : Post-processsed Dataframe.
    '''
    
    if site_info['type'] == 'solar':
        ra =  Ra(longitude=site_info['longitude'],
                  latitude=site_info['latitude'],
                  altitude=10,
                  dc_capactity_modules=site_info['max_power'], # DC capacity of modules
                  dc_capactity_inverters=site_info['max_power_dc'], # DC capacity of inverters
                  orientation=180,
                  tilt=30)
        
        # Calculate the solar position
        solar_pos = ra.calculate_solpos(df.index.get_level_values('valid_time')) 

        # Assuming no production when the sun is below the horizon
        df.loc[solar_pos['zenith'].values>90,:] = 0 
        
    # Since the wind production was normalized with the logit function, we use the inverse logit (expit) to get the 'actual' production
    if site_info['type'] == 'wind':
        df = expit(df)

    # Since the production was normalized prior to forecasting, it should vary between [0,1] 
    df[df>1] = 1
    df[df<0] = 0 
    
    # Create a new datetime index with the desired range (since solar forecasts were not generated with zenith angles above 90 degrees)
    new_valid_time = [pd.date_range(start=dt, periods = 56, freq='h').tolist() for dt in df.index.get_level_values('ref_time').unique()]
    new_valid_time = [item for sublist in new_valid_time for item in sublist]

    new_ref_time = [[dt]*56 for dt in df.index.get_level_values('ref_time').unique()]
    new_ref_time = [item for sublist in new_ref_time for item in sublist]

    new_index = pd.MultiIndex.from_arrays([new_ref_time, new_valid_time], names=('ref_time', 'valid_time'))

    # Reindex the DataFrame to fill missing datetimes and set NaN values to 0
    df = df.reindex(new_index, fill_value=0)

    # Go from normalized to nominal production
    # df = pd.DataFrame(df * site_info['max_power'])

    # Sort values
    df.sort_values([('ref_time'), ('valid_time')], inplace = True)

    return df

################################################################################
# Function to cluster horizons
################################################################################
def cluster_horizons(df, train_all_horizons = True):
    '''
    This function clusters horizons for the case of when one model should be used to 
    forecast a predefined number of clusters of forecast horizons.
    
    Parameters
    ----------
    df: Dataframe with multiindex of forecast reference and valid times.
    train_all_horizons : If all forecast horizons should be trained in one go (True) or separately (False) (optional - default is True).

    Returns
    -------
    clustered_data: Dictionary consisting of the clustered horizons ('all' or clusters of forecast horizons on the format '{:03.0f}-{:03.0f}')

    '''
    clustered_data = {}
    if train_all_horizons:
        clustered_hrs = 'all'
        clustered_data[clustered_hrs] = df
    else:
        nr_clusters = 14 # Define number of clusters
        lead_times = df.index.get_level_values('valid_time') - df.index.get_level_values('ref_time') # Get forecast horizons
        unique_lead_times = lead_times.unique().sort_values() # Get unique forecast horizons
        nr_timesteps_in_cluster = math.ceil(len(unique_lead_times) / nr_clusters) # Number of forecast horizons in each cluster
        for cluster in range(nr_clusters):
            lead_time_cluster = unique_lead_times[(cluster * nr_timesteps_in_cluster) : (cluster + 1) * nr_timesteps_in_cluster] # Get cluster of lead times
            df_temp = df[lead_times.isin(lead_time_cluster)] # Create cluster
            if not lead_time_cluster.empty:
                clustered_data['{:03.0f}-{:03.0f}'.format(lead_time_cluster[0].total_seconds() / 3600, lead_time_cluster[-1].total_seconds() / 3600,)] = df_temp
    return clustered_data



################################################################################
# Function to plot probabilitic (quantile) forecasts
################################################################################ 
def plot_quantile_forecasts(fcs, site_info, quantiles = np.linspace(5,95,19,dtype = int).tolist()):
    '''
    This function plots the operational quantile forecasts. 
    
    Parameters
    ----------
    fcs: Dictionary of Dataframes of quantile forecasts, where each column corresponds to a quantile q.
         Each key in the dictionary corresponds to a site.
    site_info: Dataframe with information about the sites.
    quantiles: List of quantiles that the forecast corresponds to. (optional - default is quantiles from 5 to 95 in steps of 6).


    '''
    n_i = int(len(quantiles)/2) # Number of intervals
    c = np.arange(1, n_i + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])

    fig,ax = plt.subplots(site_info.shape[0], sharex=True, figsize=(8,6))

    for i,site in site_info.iterrows(): 
        site_name = site['site_name']  
        for j in range(n_i):
            # if j == 3:
            #     ax[i].plot(fcs[site_name][f"q{quantiles[j]}"].values ,'r--', linewidth = 0.5)
            #     ax[i].plot(fcs[site_name][f"q{quantiles[len(quantiles)-j-1]}"].values, 'r--', linewidth = 0.5, label = '60% int.')
            # if j == 7:
            #     ax[i].plot(fcs[site_name][f"q{quantiles[j]}"].values ,'r:', linewidth = 0.5)
            #     ax[i].plot(fcs[site_name][f"q{quantiles[len(quantiles)-j-1]}"].values, 'r:', linewidth = 0.5, label = '20% int.')
                
            if site_info.shape[0] == 1:
                ax.fill_between(np.arange(fcs[site_name].shape[0]), 
                                    fcs[site_name][f"q{quantiles[j]}"].values, 
                                    fcs[site_name][f"q{quantiles[len(quantiles)-j-1]}"].values,
                                    color=cmap.to_rgba(j + 1), linewidth=0.0)#, alpha=alphas[j])
            else:
                ax[i].fill_between(np.arange(fcs[site_name].shape[0]), 
                                    fcs[site_name][f"q{quantiles[j]}"].values, 
                                    fcs[site_name][f"q{quantiles[len(quantiles)-j-1]}"].values,
                                    color=cmap.to_rgba(j + 1), linewidth=0.0)#, alpha=alphas[j])
        
        if site_info.shape[0] == 1:
            ax.plot(fcs[site_name].median(axis=1).values, 'k--', label = 'Median')
    
            ax.set_ylim([0,site['max_power']])
            ax.set_title(site_name.capitalize())
        else:
            ax[i].plot(fcs[site_name].median(axis=1).values, 'k--', label = 'Median')
    
            ax[i].set_ylim([0,site['max_power']])
            ax[i].set_title(site_name.capitalize())

    plt.legend(loc="upper left", frameon=False)
    plt.xlabel('Prognos [h]') 

    plt.figtext(0.05, 0.95, fcs[site_name].index.get_level_values('ref_time')[0].strftime('%Y-%m-%d %H:%M:%S'), 
                ha='left', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

################################################################################
# Function to verify probabilitic forecasts
################################################################################ 
def pinball(y,q,alpha):
    return (y-q)*alpha*(y>=q) + (q-y)*(1-alpha)*(y<q)

def pinball_score(df):
    '''
    This function calculates that average Pinball loss score across all quantile.

    Parameters
    ----------
    df: Dataframes of quantile forecasts, where each column corresponds to a quantile q.
    '''
    score = list()
    for qu in range(10,100,10):
        score.append(pinball(y=df["production"],
            q=df[f"q{qu}"],
            alpha=qu/100).mean())
    return sum(score)/len(score)


