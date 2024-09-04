
# Probabilistic forecasting
This repository is a handbook to generate probabilistic forecasts. The handbook was created within the project "Increased utilisation of the grid with combined solar- and wind power parks" (no. 49421-1), financed by the Swedish Energy Agency. The repository contains utilities to generate operational probabilistic forecasts of power production using the numerical weather predictor (NWP) MetCoOp Ensemble Prediction System (MEPS) and statistical post-processing methodologies.

## Prepare your Python environment
This example was developed using anaconda. You can use the file ```environment.yml``` create an environment with the same packages and version of Python. In anaconda prompt, run

```bash
conda env create -f environment.yml
conda activate forecasting_env
```

## Update sites
Update the file ```info_sites.xlsx``` with the sites that you want to forecast.

## Train forecasting models
The script ```train_fc.py``` is used to train the forecasting models.

## Generate live forecasts
After the forecasting models are trained, they could be used to generate forecasts using the most up to date NWP forecasts using the ```auto_forecaster.py``` script.
