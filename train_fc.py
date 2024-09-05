# -*- coding: utf-8 -*-
"""
@author: Oskar Lindberg
E-mail: oskar.lars.lindberg@gmail.com
"""
import os
import numpy as np
import sys

# PATH = os.path.dirname(os.path.realpath(__file__)) # Windows
PATH = '/Users/oskarlindberg/Documents/GitHub/49421---Probabilistic-forecasting' # macOS
sys.path.insert(1, os.path.join(PATH, 'functions'))

os.chdir(PATH)

import functions as fn
import fc_utils
import nwp_utils

param_dict = {
             # General parameters
             'PATH': PATH, # Path to this script
             
             # NWP parameters
             'download-historical-nwp': True, # Boolean: If NWP should be downloaded or not
             'start_date': '2020-01-01', # Start date on the format '%Y-%m-%d'
             'end_date': '2020-12-31', # End date on the format '%Y-%m-%d'
             'ref-times': [0], # List of reference time(s) of interest
             
             # Forecasting parameters
             'algorithm': 'QRF',
             'train-vs-import': True, # Boolean: If the forecast models should be trained or not      
             'train-all-horizons': True, # Boolean: if all horizons should be trained in one go or separately
             'quantiles': np.linspace(5,95,19,dtype = int).tolist(), # Quantiles to forecast 
             'backtesting': True,
             'visualize-live-forecasts': True, # If operational forecasts (generated from auto_forecaster.py) should be plotted or not
             }

# Get site information
site_info = fn.load_sites()#.iloc[0,:].to_frame().transpose()

# Save parameters dictionary
fn.save_params(param_dict)

# If historical weather forecasts should be downloaded or not
if param_dict['download-historical-nwp']: 
    nwp_utils.download_historical_MetNo_HIRESMEPS(site_info = site_info,
                                                  start_date = param_dict['start_date'], 
                                                  end_date = param_dict['end_date'], 
                                                  ref_times = param_dict['ref-times'])
    
# If the forecasting models should be trained
if param_dict['train-vs-import']:    
    for ref_time in param_dict['ref-times']:
        fc_utils.train_fc_model(site_info = site_info,
                                ref_time = [ref_time],
                                algorithm = param_dict['algorithm'],
                                quantiles = param_dict['quantiles'],
                                train_all_horizons = param_dict['train-all-horizons'],
                                )

if param_dict['backtesting'] and not param_dict['train-vs-import']:
    fcs, obs = fc_utils.backtesting(site_info = site_info,
                                    ref_time = [0],
                                    algorithm = param_dict['algorithm'],
                                    quantiles = param_dict['quantiles'],
                                    train_all_horizons = param_dict['train-all-horizons'],
                                    training_set_ratio = 0.5
                                    )
    