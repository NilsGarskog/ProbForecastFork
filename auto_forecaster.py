# -*- coding: utf-8 -*-
"""
@author: Oskar Lindberg
E-mail: oskar.lars.lindberg@gmail.com
"""
import os
import sys 

# PATH = os.path.dirname(os.path.realpath(__file__)) # Windows
PATH = '/Users/oskarlindberg/Documents/GitHub/49421---Probabilistic-forecasting' # macOS
sys.path.insert(1, os.path.join(PATH, 'functions'))

os.chdir(PATH)

import fc_utils
import functions as fn

# Load pre-defined parameters
param_dict = fn.load_params()

# Get site information
site_info = fn.load_sites()

# Generate the most updated forecasts
fcs = fc_utils.live_forecaster(site_info = site_info,
                                algorithm = param_dict['algorithm'],
                                quantiles = param_dict['quantiles'],
                                train_all_horizons = param_dict['train-all-horizons'],
                                only_fetch_new = False
                                )    

# If forecasts should be visualized or not
if (param_dict['visualize-live-forecasts']) and (fcs != None):
    fn.plot_quantile_forecasts(fcs = fcs, 
                               site_info = site_info, 
                               quantiles = param_dict['quantiles'])
