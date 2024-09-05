# -*- coding: utf-8 -*-
"""
@author: Oskar Lindberg
E-mail: oskar.lars.lindberg@gmail.com
"""

import netCDF4 
import pyproj
import numpy as np
import datetime
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

PATH = os.getcwd()

################################################################################
# Class to download the NWP from MetNo's High-resolution MEPS
################################################################################
class MetNoAPI:
      
    base_url = 'https://thredds.met.no/thredds/dodsC/'
    
    def __init__(
                  self,
                  latitudes,
                  longitudes,
                  site_names,
                  features
                  ):
        
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.site_names = site_names
        self.features = features
        
    def query_MetNo_HIRESMEPS(self, url):
        '''
        This function downloads NWP from the thredds server.
        
        Parameters
        ----------
        url: URL to the dataset
        
        Returns
        -------
        metno_hires: NWP forecasts as a multiindex (ref_time and valid_time) Dataframe.
        '''
        
        try:
            file = netCDF4.Dataset(url,"r") # Read netCDF file from server
            
            horizons = np.arange(0,file.variables["time"][:].shape[0],1).tolist() # Get forecast horizons
            
            proj = pyproj.Proj(file.variables["projection_lcc"].proj4) # Define projection
        
            # See information of the projection by: print(ncfile.variables["projection_lambert"])
            crs = pyproj.CRS.from_cf(
                                    {
                                        "grid_mapping_name": "lambert_conformal_conic",
                                        "standard_parallel": [63.3, 63.3],
                                        "longitude_of_central_meridian": 15.0,
                                        "latitude_of_projection_origin": 63.3,
                                        "earth_radius": 6371000.0,
                                    }
                                    )
            
            # Transformer to project from ESPG:4368 (WGS:84) to their lambert_conformal_conic
            proj = pyproj.Proj.from_crs(4326, crs, always_xy=True)
                       
            # Find nearest neighbour from x and y coordinates
            x = file.variables['x'][:]
            y = file.variables['y'][:]
            
            # Define dictionary where each site corresponds to a key in the dictionary
            metno_hires = {}

            for i in range(len(self.latitudes)):
                # Compute projected coordinates of lat/lon points
                X,Y = proj.transform(self.longitudes[i], self.latitudes[i])
                
                Ix = np.argmin(np.abs(x - X))
                Iy = np.argmin(np.abs(y - Y))
                
                # Get reference times of forecasts
                ref_time = file.variables["forecast_reference_time"][:]
                ref_time = pd.to_datetime(datetime.datetime.utcfromtimestamp(ref_time+0), utc=True) 
                ref_times = [ref_time]*(len(horizons))
                df_ref_times = pd.DataFrame(data=ref_times,columns=['ref_time'])
                
                # Forecast valid times
                valid_time = file.variables["time"][:]
                valid_time = np.vstack([pd.to_datetime(datetime.datetime.utcfromtimestamp(np.asarray(i)+0), utc=True) for i in valid_time])
                df_valid_time = pd.DataFrame(data=valid_time,columns=['valid_time'])
                
                df = pd.concat([df_ref_times, df_valid_time], axis = 1)
            
                # Wind speed at 10 meters
                if 'wind_speed_10m' in self.features[i]:
                    ws_10m_wind = file.variables["wind_speed_10m"][horizons,Iy,Ix] # unit: m/s
                    df_temp = pd.DataFrame(data=ws_10m_wind,columns=['wind_speed_10m'])
                    df = df.join(df_temp)
                
                # Wind direction at 10 meters
                if 'wind_direction_10m' in self.features[i]:
                    wd_10m_wind = file.variables["wind_direction_10m"][horizons,Iy,Ix] # unit: Degrees
                    df_temp = pd.DataFrame(data=wd_10m_wind,columns=['wind_direction_10m'])
                    df = df.join(df_temp)
                
                # Wind speed of gust at 10 meters
                if 'wind_speed_of_gust' in self.features[i]:
                    wg_10m_wind = file.variables["wind_speed_of_gust"][horizons,Iy,Ix] # unit: m/s
                    df_temp = pd.DataFrame(data=wg_10m_wind,columns=['wind_speed_of_gust'])
                    df = df.join(df_temp)
                
                # GHI at surface
                if 'ghi' in self.features[i]:
                    ghi_accumulated = file.variables['integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time'][:,Iy,Ix] / 3600 # unit: Ws/m2
                    ghi = np.diff(ghi_accumulated,axis=0,prepend=0) # Prepend with 0 because it is the integral of shortwave flux.
                    ghi = ghi[horizons] # Select the forecast period.
                    df_temp = pd.DataFrame(data=ghi,columns=['ghi'])
                    df = df.join(df_temp)
            
                # Cloud area fraction
                if 'cloud_area_fraction' in self.features[i]:
                    frac = file.variables["cloud_area_fraction"][horizons,Iy,Ix] # unit: %
                    df_temp = pd.DataFrame(data=frac,columns=['cloud_area_fraction'])
                    df = df.join(df_temp)
                
                # Temperature at 2 meters
                if 'air_temperature_2m' in self.features[i]:
                    temperatures = file.variables["air_temperature_2m"][horizons,Iy,Ix] # unit: degrees K
                    df_temp = pd.DataFrame(data=temperatures, columns=['air_temperature_2m'])
                    df = df.join(df_temp)
            
                # Relative humidity at 2 meters
                if 'relative_humidity_2m' in self.features[i]:
                    rel = file.variables["relative_humidity_2m"][horizons,Iy,Ix] # unit: %
                    df_temp = pd.DataFrame(data=rel, columns=['relative_humidity_2m'])
                    df = df.join(df_temp)
                    
                # Precipitation
                if 'precipitation_amount' in self.features[i]:
                    prec = file.variables["precipitation_amount"][horizons,Iy,Ix] # unit: %
                    df_temp = pd.DataFrame(data=prec, columns=['precipitation_amount'])
                    df = df.join(df_temp)
                
                df.set_index(['ref_time', 'valid_time'], inplace = True)
                metno_hires[self.site_names[i]] = df
                
            return metno_hires

        except OSError as error:
            print(error)
            metno_hires = {}
            return metno_hires
    
    def sort_historical_MetNo_HIRESMEPS(self, ref_times):
        '''
        This function sort weather forecasts.

        Parameters
        ----------
        ref_times: List of reference times, e.g., [0] or [0,6,12,18].
        '''
        
        for site_name in self.site_names:
            print(f"\nSorting site: {site_name}")
            df = pd.DataFrame()
            for ref_time in tqdm(ref_times):
                foldername = os.path.join(PATH, 'data', 'nwp', 'MetNo_HIRESMEPS', site_name, '{:02d}Z'.format(ref_time))
                filenames = os.listdir(foldername)
        
                for filename in filenames:
                    df_temp = pd.read_csv(os.path.join(foldername, filename), sep = '\t', parse_dates = [0,1])
                    df = pd.concat([df, df_temp], axis = 0, ignore_index = True)
            df = df.set_index(['ref_time','valid_time']) # Create multiindex consisting of reference time and valid time
        
            df.sort_values(['ref_time', 'valid_time'], inplace = True)            
            foldername = os.path.join(PATH, 'data', 'nwp', 'MetNo_HIRESMEPS', site_name)  
            filename = "{}_{}_{}.txt".format(site_name, 'MetNo_HIRESMEPS', 'compiled')  
            df.to_csv(os.path.join(foldername,filename), sep="\t")
            print(f"\n{site_name} was sorted and saved.")

    def get_historical_MetNo_HIRESMEPS(self, start_date, end_date, ref_times):
        '''
        This function downloads historical forecasts from the MetNo Hires model from a specified start date, end date
        and reference time(s). The function also sorts the downloaded forecasts.
        
        Parameters
        ----------
        start_date: Start date as a string on the format "%Y-%m-%d".
        end_date: End date as a string on the format "%Y-%m-%d".
        ref_times: List of reference times to download, e.g., [0] or [0,6,12,18]

        '''
        
        strtdt = datetime.datetime.strptime(start_date, "%Y-%m-%d") 
        enddt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start=strtdt, end=enddt,freq='D')
        
        for date in tqdm(dates):
            for ref_time in ref_times:
                file_name = "met_forecast_1_0km_nordic_{:04d}{:02d}{:02d}T{:02d}Z.nc".format(date.year, date.month, date.day, ref_time)
                url = f"{self.base_url}/metpparchive/{date.year:04d}/{date.month:02d}/{date.day:02d}/{file_name}"
                metno_hires = self.query_MetNo_HIRESMEPS(url)
                
                for site_name in self.site_names:
                    foldername = os.path.join(PATH, 'data', 'nwp', 'MetNo_HIRESMEPS', site_name, '{:02d}'.format(ref_time) + "Z")
                    filename = date.strftime("%Y%m%dT%H%M")+".txt"
                
                    os.makedirs(foldername, exist_ok=True) # If folder does not exist, create it
                    
                    if metno_hires != {}: # If missing from database, the dataframe will be empty
                        df = metno_hires[site_name]
                        df.to_csv(os.path.join(foldername,filename), sep="\t")

        self.sort_historical_MetNo_HIRESMEPS(ref_times = ref_times)

    def get_latest_MetNo_HIRESMEPS(self):
        '''
        This function downloads the latest forecasts.
        '''
        url = f"{self.base_url}/metpplatest/met_forecast_1_0km_nordic_latest.nc"
        metno_hires = self.query_MetNo_HIRESMEPS(url)
        return metno_hires
            
    def load_rss(self):
        '''
        This function gets and saves the latest xml-file using the URL of RSS feed.
        '''
        
        url = 'https://thredds.met.no/thredds/catalog/metpplatest/catalog.xml' # URL of RSS feed    
        resp = requests.get(url) # Creating HTTP response object from the given URL

        # Saving the XML file
        os.makedirs(os.path.join(PATH, 'xml'), exist_ok=True) # If folder does not exist, create it

        with open(os.path.join(PATH, 'xml', 'metpplatest_newsfeed.xml'), 'wb') as f:
            f.write(resp.content)

    def parse_xml(self, xmlfile):
        '''
        This function parse an XML file and obtain the time stamp of the most current updated file.

        Parameters
        ----------
        xmlfile : An XML-file from: 'https://thredds.met.no/thredds/catalog/metpplatest/catalog.xml'

        Returns
        -------
        modified_date: Time during which the latest file was updated.

        '''
        tree = ET.parse(xmlfile) # Create element tree object
        root = tree.getroot() # Get root element
        namespace = {'ns': 'http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0'} # Define the namespace used in the XML
        dataset_xpath = ".//ns:dataset[@name='met_forecast_1_0km_nordic_latest.nc']/ns:date[@type='modified']" # Find the dataset with the specified name
        
        modified_date_element = root.find(dataset_xpath, namespaces=namespace) # Use XPath to extract the modified date
        
        if modified_date_element is not None:
            modified_date = modified_date_element.text
            return modified_date
        else:
            print("Dataset not found in the XML file.")

    def automatic_fetching(self):
        '''
        This function fetches the most current weather forecasts in the case of the database being updated since the last run.

        '''
        
        fetch_nwp = False
        if 'metpplatest_newsfeed.xml' not in os.listdir(os.path.join(PATH, 'xml')): # If a file does not exist the first time, a forecast should be generated
            fetch_nwp = True
            self.load_rss() # Load RSS and update existing XML file
        else:
            previous_update = self.parse_xml(os.path.join(PATH, 'xml', 'metpplatest_newsfeed.xml')) # Parse the current XML file
            self.load_rss() # Get the most updated xml-file
            latest_update = self.parse_xml(os.path.join(PATH, 'xml', 'metpplatest_newsfeed.xml'))  # Parse the updated XML file
            
            if (latest_update != previous_update):
                fetch_nwp = True
        if fetch_nwp == True:
            return self.get_latest_MetNo_HIRESMEPS()
        else:
            return 'updated'
    
def download_historical_MetNo_HIRESMEPS(site_info, start_date, end_date, ref_times = [0,6,12,18]):
    '''
    This function downloads historical MetNo NWPs.
    
    Parameters
    ----------
    site_info: Dataframe with information about the sites.
    start_date: Start date on the string-format '%Y-%m-%d'
    end_date: End date on the string-format '%Y-%m-%d'
    ref_times: List of reference time(s) to download (optional - default is [0,6,12,18])
    '''   
    
    MetNo = MetNoAPI(
                     latitudes = site_info['latitude'].tolist(),
                     longitudes = site_info['longitude'].tolist(),
                     site_names = site_info['site_name'].tolist(),
                     features = site_info['features']
                     )
    
    MetNo.get_historical_MetNo_HIRESMEPS(start_date = start_date, 
                                         end_date = end_date, 
                                         ref_times = ref_times)
    
    
    