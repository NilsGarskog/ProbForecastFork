# -*- coding: utf-8 -*-
"""
@author: Oskar Lindberg
E-mail: oskar.lars.lindberg@gmail.com
"""

import numpy as np
import pandas as pd

import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain

class Ra():

    def __init__(self,
                 longitude,
                 latitude,
                 altitude,
                 dc_capactity_modules,
                 dc_capactity_inverters,
                 orientation,
                 tilt,
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model='pvwatts',
                 ac_model='pvwatts',
                 aoi_model='no_loss',
                 spectral_model='no_loss',
                 temperature_model='sapm',
                 losses_model='no_loss'):

        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.dc_capactity_modules = dc_capactity_modules
        self.dc_capactity_inverters = dc_capactity_inverters
        self.orientation = orientation
        self.tilt = tilt
        
        temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']
        
        self.location = Location(longitude=self.longitude,
                                 latitude=self.latitude,
                                 tz='UTC',
                                 altitude=self.altitude)
        self.system = PVSystem(surface_tilt=self.tilt,
                               surface_azimuth=self.orientation,
                               module_parameters={'pdc0': dc_capactity_modules, 'gamma_pdc': -0.004}, 
                               inverter_parameters={'pdc0': dc_capactity_inverters},
                               temperature_model_parameters=temp_params['open_rack_glass_polymer'])
        
        self.set_modelchain(clearsky_model=clearsky_model,
                          transposition_model=transposition_model,
                          solar_position_method=solar_position_method,
                          airmass_model=airmass_model,
                          dc_model=dc_model,
                          ac_model=ac_model,
                          aoi_model=aoi_model,
                          spectral_model=spectral_model,
                          temperature_model=temperature_model,
                          losses_model=losses_model)


    def set_modelchain(self,
                       clearsky_model='ineichen',
                       transposition_model='haydavies',
                       solar_position_method='nrel_numpy',
                       airmass_model='kastenyoung1989',
                       dc_model='pvwatts',
                       ac_model='pvwatts',
                       aoi_model='no_loss',
                       spectral_model='no_loss',
                       temperature_model='sapm',
                       losses_model='no_loss'):

        self.modelchain = ModelChain(self.system,
                                     self.location,
                                     clearsky_model=clearsky_model,
                                     transposition_model=transposition_model,
                                     solar_position_method=solar_position_method,
                                     airmass_model=airmass_model,
                                     dc_model=dc_model,
                                     ac_model=ac_model,
                                     aoi_model=aoi_model,
                                     spectral_model=spectral_model,
                                     temperature_model=temperature_model,
                                     losses_model=losses_model)


    def calculate_solpos(self, index):
        solpos = self.location.get_solarposition(index)
        return solpos

    def calculate_clearsky(self, index, freq_str=None):
        clearsky = self.location.get_clearsky(index)
        clearsky.index = clearsky.index
        return clearsky

    def weather_from_tcc(self, tcc, freq_str=None):
        weather = self.model.cloud_cover_to_irradiance(tcc, how='clearsky_scaling')
        return weather

    def weather_from_ghi(self, ghi, freq_str=None):
        solpos = self.calculate_solpos(ghi.index)
        weather = pvlib.irradiance.erbs(np.ravel(ghi), np.ravel(solpos['zenith']), ghi.index)
        weather['ghi'] = ghi
        poa = self.modelchain.run_model(weather=weather).results.total_irrad
        weather['poa'] = poa['poa_global']
        return weather

    def calculate_power_clearsky(self, index):
        clearsky = self.calculate_clearsky(index)
        power_clearsky = self.modelchain.run_model(weather=clearsky).results.ac
        return power_clearsky

    def calculate_power(self, weather):
        power_clearsky = self.calculate_power_clearsky(weather.index)
        power = self.modelchain.run_model(weather=weather).results.ac

        # If power is greater than clearsky use clearsky
        idx_clearsky = power > power_clearsky
        power.loc[idx_clearsky] = power_clearsky.loc[idx_clearsky]

        # If power is negative set to zero
        idx_negative = power < 0
        power.loc[idx_negative] = 0

        return power