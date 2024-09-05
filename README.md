
# Probabilistic forecasting
This repository is a handbook to generate probabilistic power production forecasts. The repository contains utilities to generate operational probabilistic forecasts of power production using the numerical weather predictor (NWP) MetCoOp Ensemble Prediction System (MEPS) and statistical post-processing methodologies.

## Prepare your Python environment
This example was developed using Anaconda. You can use the file ```environment.yml``` create an environment with the same packages and version of Python. In the Anaconda prompt, run:

```bash
conda env create -f environment.yml
conda activate forecasting_env
```

## Update sites
Update the file ```info_sites.xlsx``` with the sites that you want to forecast.

## Add power production data 
Add a csv-file containing the electrical energy production from each of your site(s) provided in ```info_sites.xlsx```. The csv-file must contain one column of datetime index and at least one column with power production (if multiple columns are present, they are aggregated). Put the file in the folder: ```~/data/energy_production/{site_name}/{site_name}.csv```

## Train forecasting models
The script ```train_fc.py``` is used to download NWP forecasts, train the forecasting models and perform backtesting.

## Generate live forecasts
After the forecasting models are trained, they could be used to generate forecasts using the most up to date NWP forecasts using the ```auto_forecaster.py``` script.
