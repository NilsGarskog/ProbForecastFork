# -*- coding: utf-8 -*-
"""
@author: Oskar Lindberg 
E-mail: oskar.lars.lindberg@gmail.com
"""
################################################################################
# Import packages
################################################################################
import numpy as np
import pandas as  pd
import joblib as joblib
import os
from quantile_forest import RandomForestQuantileRegressor
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import nwp_utils
import functions as fn

################################################################################
# Quantile Random Forest (QRF) forecasting algorithm
################################################################################
class QRF:
    def __init__(self, 
                 site_name,
                 quantiles = np.linspace(5,95,19,dtype = int).tolist(),
                 ):

        self.site_name = site_name
        self.quantiles = quantiles
        self.algorithm = 'qrf'
        
    def train_qrf_model(self, train_X, train_y, save_model = True):
        '''
        This function trains a QRF model.
        
        Parameters
        ----------
        train_X: Dataframe of of feature variable(s)
        train_y: Dataframe of target variable
        save_model: If the trained model should be saved or not (optional - default is True)

        Returns
        -------
        model: Trained QRF model
        '''
        # Define model parameters
        model = RandomForestQuantileRegressor(n_estimators=100, 
                                              max_features=train_X.shape[1], 
                                              min_samples_leaf=10,
                                              random_state=1)
        # Train the model
        model.fit(train_X, train_y)
        
        if save_model:
            ref_times = sorted(train_X.index.get_level_values('ref_time').hour.unique().tolist())
            
            PATH_MODELS = os.path.join('.','models', self.algorithm, self.site_name) # Path to store the file
            os.makedirs(PATH_MODELS, exist_ok=True) # If folder doesn't exist, then create it.
            
            filename = "{}_{}_{}_{}_{}_{}.{}".format(self.site_name, self.algorithm, "ref_time", "_".join(map(str, ref_times)), "horizon", self.horizon, "sav") # Define filename
            joblib.dump(model, os.path.join(PATH_MODELS, filename)) # Save the model
                
        return model
    
    def forecast_qrf(self, test_X, model = None, ref_times = None):
        '''
        This function forecasts using a QRF model.

        Parameters
        ----------
        test_X: Dataframe of of feature variable(s)
        model: Trained QRF model (optional - default is None)

        Returns
        -------
        df_fc: Dataframe of probabilistic forecasts where the columns correspond to the quantile(s).

        '''
        # If model is not provided to the function, try to load it
        if model == None: 
            if ref_times == None:
                target_ref_time = sorted(test_X.index.get_level_values('ref_time').hour.unique().tolist())
                
                # Get the model with the closest corresponding reference time
                if len(target_ref_time) == 1:
                    avaliable_ref_times = [0, 6, 12, 18]
                    ref_times = [min(avaliable_ref_times, key=lambda x: min(abs(x - target_ref_time[0]), abs(target_ref_time[0] - x - 24)))] # Update to the closest avaliable trained forecasting model based on ref_time
                else:
                    ref_times = target_ref_time
                    
            PATH_MODELS = os.path.join('.','models', self.algorithm, self.site_name) # Path where the file is stored
            filename = "{}_{}_{}_{}_{}_{}.{}".format(self.site_name, self.algorithm, "ref_time", "_".join(map(str, ref_times)), "horizon", self.horizon, "sav") # Define filename
            
            try:
                model = joblib.load(os.path.join(PATH_MODELS, filename)) # Load model                
            except:
                print("ERROR\n No trained model were found in directory:\n {}".format(os.path.join(PATH_MODELS, filename))) # If model does not exist, it will not be able to forecast
                return
            
        # Forecast using the provided/trained model
        fc = model.predict(test_X, quantiles=[q / 100 for q in self.quantiles]) 
    
        # Store the forecasts in a dataframe
        df_fc = pd.DataFrame(data = fc, 
                             index = test_X.index,
                             columns = [f"q{quantile}" for quantile in self.quantiles])
        return df_fc

################################################################################
# Quantile Regression (QR) forecasting algorithm
################################################################################
class QR:
    def __init__(self, 
                 site_name,
                 quantiles = np.linspace(5,95,19,dtype = int).tolist(),
                 ):

        self.site_name = site_name
        self.quantiles = quantiles
        self.algorithm = 'qr'
        
    def train_qr_model(self, train_X, train_y, save_model = True):
        '''
        This function trains a QR model.

        Parameters
        ----------
        train_X: Dataframe of of feature variable(s)
        train_y: Dataframe of target variable
        save_model: If the trained model should be saved or not (optional - default is True)

        Returns
        -------
        model: Trained QR model
        '''
        model = sm.QuantReg(train_y, train_X)
        model_fit = {}
        
        for q in self.quantiles:
            model_fit[f"q{q}"] = model.fit(q=q/100, max_iter=20000)
                
        if save_model:
            ref_times = sorted(train_X.index.get_level_values('ref_time').hour.unique().tolist())

            PATH_MODELS = os.path.join('.','models', self.algorithm, self.site_name) # Path to store the file
            os.makedirs(PATH_MODELS, exist_ok=True) # If folder doesn't exist, then create it.
            
            filename = "{}_{}_{}_{}_{}_{}.{}".format(self.site_name, self.algorithm, "ref_time", "_".join(map(str, ref_times)), "horizon", self.horizon, "sav") # Define filename
            joblib.dump(model_fit, os.path.join(PATH_MODELS, filename)) # Save the model
                
        return model
    
    def forecast_qr(self, test_X, ref_times = None, model = None):
        '''
        This function forecasts using a QR model.

        Parameters
        ----------
        test_y : Dataframe of of feature variable(s)
        model : Trained QRF model (optional - default is None)

        Returns
        -------
        df_fc : Dataframe of probabilistic forecasts where the columns correspond to the quantile(s).

        '''
        # If model is not provided to the function, try to load it
        if model == None: 
            if ref_times == None:
                target_ref_time = sorted(test_X.index.get_level_values('ref_time').hour.unique().tolist())
                
                # Get the model with the closest corresponding reference time
                if len(target_ref_time) == 1:
                    avaliable_ref_times = [0, 6, 12, 18]
                    ref_times = [min(avaliable_ref_times, key=lambda x: min(abs(x - target_ref_time[0]), abs(target_ref_time[0] - x - 24)))] # Update to the closest avaliable trained forecasting model based on ref_time
                else:
                    ref_times = target_ref_time
                    
            PATH_MODELS = os.path.join('.','models', self.algorithm, self.site_name) # Path where the file is stored
            filename = "{}_{}_{}_{}_{}_{}.{}".format(self.site_name, self.algorithm, "ref_time", "_".join(map(str, ref_times)), "horizon", self.horizon, "sav") # Define filename
            
            try:
                model = joblib.load(os.path.join(PATH_MODELS, filename)) # Load model                
            except:
                print("ERROR\n No trained model were found in directory:\n {}".format(os.path.join(PATH_MODELS, filename))) # If model does not exist, it will not be able to forecast
                return
            
        # Forecast using the provided/trained model
        fc_q = []

        for q in self.quantiles:
            fc_q.append(model[f"q{q}"].predict(test_X)) 
            fc = np.vstack(fc_q).T # List to NumPy array

        # Store the forecasts in a dataframe
        df_fc = pd.DataFrame(data = fc, 
                             index = test_X.index,
                             columns = [f"q{quantile}" for quantile in self.quantiles])
        return df_fc


################################################################################
# Train forecast model(s)
################################################################################
def train_fc_model(site_info,
                   ref_time,
                   algorithm = 'QRF',
                   quantiles = np.linspace(5,95,19,dtype = int).tolist(),
                   train_all_horizons = True,
                   ):
    '''
    This function trains models for a pre-defined forecasting algorithm.
    
    Parameters
    ----------
    site_info: Dataframe with information about the sites.
    ref_time: List of reference time that should be used when training, e.g., [0] or [0,6] etc.
    algorithm: Algorithm that should be trained  (optional - default is 'QRF')
    quantiles : List of quantiles that should be forecasted (optional - default is 5 to 95 in steps of 5)
    train_all_horizons : If all forecast horizons should be trained in one go (True) or separately (False) (optional - default is True).

    '''
    # Compiled energy production and weather forecasts
    print('\nCompiling data...')
    modelling_table = fn.compile_data(site_info = site_info)
    print('data was compiled.')

    # Pre process that data
    print('\nPreprocessing data...')
    modelling_table = fn.preprocess_training(modelling_table = modelling_table, 
                                             site_info = site_info)
    print('data was preprocessed.')

    for _,site in site_info.iterrows(): 
        site_name = site['site_name']   
        
        # Get the combination of reference times of interest for training
        bol = [False]*modelling_table[site['site_name']].shape[0]
        
        for h in ref_time:
            bol = [a or b  for a,b, in zip(bol, (modelling_table[site['site_name']].index.get_level_values('ref_time').hour == h).tolist())]
            
        modelling_table[site['site_name']] = modelling_table[site['site_name']].iloc[bol,:]
       
        print(f'\nTraining {algorithm} model for site {site_name}...')
        train_X = modelling_table[site_name].drop('production', axis=1) # Get feature variables
        train_y = modelling_table[site_name]['production'] # Get target variables
        
        # Cluster horizons
        train_X = fn.cluster_horizons(train_X, train_all_horizons)
        train_y = fn.cluster_horizons(train_y, train_all_horizons)
            
        if algorithm == 'QRF':
            model = QRF(site_name = site_name,
                        quantiles = quantiles)
            
            for ky in train_X.keys():
                model.horizon = ky      
                _ = model.train_qrf_model(train_X = train_X[ky], 
                                          train_y = train_y[ky],
                                          save_model = True)  
        elif algorithm == 'QR':
            model = QR(site_name = site_name,
                       quantiles = quantiles)
            
            for ky in train_X.keys():
                model.horizon = ky      
                _ = model.train_qr_model(train_X = train_X[ky], 
                                          train_y = train_y[ky],
                                          save_model = True)  
        
        elif algorithm == 'QRF+QR':
            model_qrf = QRF(site_name = site_name,
                            quantiles = quantiles)
        
            model_qr = QR(site_name = site_name,
                          quantiles = quantiles)
            
            for ky in train_X.keys():
                model_qrf.horizon = ky  
                model_qr.horizon = ky      

                ii = int(train_X[ky].shape[0] / 1.3) # Define how much data that should be used for training the QR model
                model_fit = model_qrf.train_qrf_model(train_X = train_X[ky].iloc[:ii,], 
                                                  train_y = train_y[ky].iloc[:ii,],
                                                  save_model = True)  
                
                fcs = model_qrf.forecast_qrf(train_X[ky][ii:],
                                             model = model_fit)
                
                _ = model_qr.train_qr_model(train_X = fcs, 
                                            train_y = train_y[ky][ii:],
                                            save_model = True)  
                
        print(f'{algorithm} model for site {site_name} was trained.')
 
################################################################################
# Train forecast model(s)
################################################################################
def backtesting(site_info,
                ref_time,
                algorithm = 'QRF',
                quantiles = np.linspace(5,95,19,dtype = int).tolist(),
                train_all_horizons = True,
                training_set_ratio = 0.5
                ):
    
    # Compiled energy production and weather forecasts
    print('\nCompiling data...')
    modelling_table = fn.compile_data(site_info = site_info)
    print('data was compiled.')

    # Pre process that data
    print('\nPreprocessing data...')
    modelling_table = fn.preprocess_training(modelling_table = modelling_table, 
                                             site_info = site_info)
    print('data was preprocessed.')

    fcs = {}
    obs = {}
    
    for _,site in site_info.iterrows(): 
        
        site_name = site['site_name']   
        
        fcs[site_name] = pd.DataFrame()
        obs[site_name] = pd.DataFrame()
        
        # Get the combination of reference times of interest for training
        bol = [False]*modelling_table[site['site_name']].shape[0]
        
        for h in ref_time:
            bol = [a or b  for a,b, in zip(bol, (modelling_table[site['site_name']].index.get_level_values('ref_time').hour == h).tolist())]
            
        modelling_table[site['site_name']] = modelling_table[site['site_name']].iloc[bol,:]
        
        print(f'\nTraining {algorithm} model for site {site_name}...')
        
        N = modelling_table[site['site_name']].shape[0]
        
        # Divide into training and testing sets
        train_X = modelling_table[site_name].drop('production', axis=1).iloc[:int(N*training_set_ratio),:] # Get feature variables
        train_y = modelling_table[site_name]['production'].iloc[:int(N*training_set_ratio),] # Get target variables
        
        test_X = modelling_table[site_name].drop('production', axis=1).iloc[int(N*training_set_ratio):,] # Get feature variables
        test_y = modelling_table[site_name]['production'].iloc[int(N*training_set_ratio):,] # Get target variables
        
        # Cluster horizons
        train_X = fn.cluster_horizons(train_X, train_all_horizons)
        train_y = fn.cluster_horizons(train_y, train_all_horizons)
            
        test_X = fn.cluster_horizons(test_X, train_all_horizons)
        test_y = fn.cluster_horizons(test_y, train_all_horizons)
        
        if algorithm == 'QRF':
            model = QRF(site_name = site_name,
                        quantiles = quantiles)
            
            for ky in train_X.keys():
                model.horizon = ky      
                trained_model = model.train_qrf_model(train_X = train_X[ky], 
                                                      train_y = train_y[ky],
                                                      save_model = False)  
                
                fcs[site_name] = pd.concat([fcs[site_name], model.forecast_qrf(test_X[ky], model = trained_model)], axis = 0)
                obs[site_name] = pd.concat([obs[site_name], test_y[ky].to_frame()], axis = 0)
                
        elif algorithm == 'QR':
            model = QR(site_name = site_name,
                       quantiles = quantiles)
            
            for ky in train_X.keys():
                model.horizon = ky      
                _ = model.train_qr_model(train_X = train_X[ky], 
                                          train_y = train_y[ky],
                                          save_model = False)  
        
        elif algorithm == 'QRF+QR':
            model_qrf = QRF(site_name = site_name,
                            quantiles = quantiles)
        
            model_qr = QR(site_name = site_name,
                          quantiles = quantiles)
            
            for ky in train_X.keys():
                model_qrf.horizon = ky  
                model_qr.horizon = ky      

                ii = int(train_X[ky].shape[0] / 1.3) # Define how much data that should be used for training the QR model
                model_fit = model_qrf.train_qrf_model(train_X = train_X[ky].iloc[:ii,], 
                                                  train_y = train_y[ky].iloc[:ii,],
                                                  save_model = False)  
                
                fcs = model_qrf.forecast_qrf(train_X[ky][ii:],
                                             model = model_fit)
                
                _ = model_qr.train_qr_model(train_X = fcs, 
                                            train_y = train_y[ky][ii:],
                                            save_model = False)  
        
        # Post processing the forecasts
        fcs[site_name] = fn.postprocess(df = fcs[site_name],
                                        site_info = site)
        # Post processing the observations
        obs[site_name] = fn.postprocess(df = obs[site_name],
                                        site_info = site)
    print(f'{algorithm} model for site {site_name} was trained and forecasted.')    
    return fcs,obs
 
def live_forecaster(site_info,
                    algorithm = 'QRF',
                    quantiles = np.linspace(5,95,19,dtype = int).tolist(),
                    train_all_horizons = True,
                    only_fetch_new = True
                    ):
    '''
    This function is used to generate the most updated weather forecasts.

    Parameters
    ----------
    site_info: Dataframe with information about the sites.
    algorithm: Algorithm that should be trained (optional - default is 'QRF')
    quantiles : List of quantiles that should be forecasted (optional - default is 5 to 95 in steps of 5)
    train_all_horizons : If all forecast horizons should be trained in one go (True) or separately (False) (optional - default is True).
    only_fetch_new: Boolean if the function should check if current forecasts are already generated and, if not, download them  (optional - default is True).

    Returns
    -------
    fcs: 

    '''
    
    # Define variables for class MetNoAPI
    MetNo = nwp_utils.MetNoAPI(
                               latitudes = site_info['latitude'].tolist(),
                               longitudes = site_info['longitude'].tolist(),
                               site_names = site_info['site_name'].tolist(),
                               features = site_info['features'].tolist()
                               )
    
    # Get latest weather forecasts
    print('Downloading the latest weather forecasts...\n')
    if only_fetch_new:
        features = MetNo.automatic_fetching() # To check if current forecasts are already generated and, if not, download them
    else:
        features = MetNo.get_latest_MetNo_HIRESMEPS() # To get the current forecasts (regardless if they have been generate beforehand)

    # In the case of fetch_new == True and the XML file is already updated, the features will be returned as 'updated'
    if features == 'updated':
        return print('The most updated forecasts are already generated.') # In that case, exit the function
    
    # Preprocess the features
    preprocessed_features = fn.preprocess_features(X = features, 
                                                   site_info = site_info)
             
    # The functions are stored as a dictionary, where each key corresponds to a site.
    fcs = {}
    
    for _,site in site_info.iterrows():

        site_name = site['site_name']
        
        print(f'Forecasting {site_name.capitalize()}...\n')

        # Cluster the forecast horizons
        test_X = fn.cluster_horizons(df = preprocessed_features[site_name], 
                                     train_all_horizons = train_all_horizons)

        if algorithm == 'QRF':            
            model = QRF(site_name = site_name,
                        quantiles = quantiles)
            
            for ky in test_X.keys():
                model.horizon = ky      
                fcs[site_name] = model.forecast_qrf(test_X[ky])
        
        if algorithm == 'QR':            
            model = QR(site_name = site_name,
                       quantiles = quantiles)
            
            for ky in test_X.keys():
                model.horizon = ky      
                fcs[site_name] = model.forecast_qr(test_X[ky])
        
        elif algorithm == 'QRF+QR':
            model_qrf = QRF(site_name = site_name,
                            quantiles = quantiles)
        
            model_qr = QR(site_name = site_name,
                          quantiles = quantiles)
            
            for ky in test_X.keys():
                model_qrf.horizon = ky  
                model_qr.horizon = ky  
                fcs[site_name] = model_qrf.forecast_qrf(test_X[ky])
                fcs[site_name] = model_qr.forecast_qr(fcs[site_name])
        
        # Post processing the forecasts
        fcs[site_name] = fn.postprocess(df = fcs[site_name],
                                        site_info = site)
   
    print('All sites were forecasted!\n')

    return fcs
