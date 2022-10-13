# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 12:50:12 2022
author: Wenting Wang
School of Electrical Engineering and Automation
Harbin Institute of Technology
email: wangwenting3000@gmail.com

"""

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# irradiance-to-power conversion using model chain
# source: https://www.aeprenewables.com/jacumba-solar/
# plant name: Jacumba Solar Farm
# EIA Plant ID: 60947
# Latitude: 32.6193, Longitude: -116.130
# Capacity_AC: 20 MW
# Capacity_dc: 28 MW
# Region: CAISO
# Tilt: 25°
# Azimuth: 180°
# Year: 2017,2018,2019,2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# download McClear 

# Time reference: Universal time (UTC)
# More information at: http://www.soda-pro.com/web-services/radiation/cams-mcclear
# Latitude: 32.6193, Longitude: -116.130 (Jacumba Solar Farm)
# Columns:
# 1. Observation period (ISO 8601)
# 2. TOA. Irradiation on horizontal plane at the top of atmosphere (Wh/m2)
# 3. Clear sky GHI. Clear sky global irradiation on horizontal plane at ground level (Wh/m2)
# 4. Clear sky BHI. Clear sky beam irradiation on horizontal plane at ground level (Wh/m2)
# 5. Clear sky DHI. Clear sky diffuse irradiation on horizontal plane at ground level (Wh/m2)
# 6. Clear sky BNI. Clear sky beam irradiation on mobile plane following the sun at normal incidence (Wh/m2)

# read McClear of Jacumba
metadata_McClear = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/McClear_Jacumba.csv',sep=';')
# extract two columns: "Observation period" and "Clear sky GHI"
# "Observation period": beginning/end of the time period with the format "yyyy-mm-ddTHH:MM:SS.S/yyyy-mm-ddTHH:MM:SS.S" (ISO 8601)
McClear = metadata_McClear[["Observation period","Clear sky GHI","Clear sky DHI","Clear sky BNI"]]
# set the beginning of the time period as the index.
begin_time_McClear = pd.date_range(start='2017-01-01 00:00:00', end='2020-12-31 23:45:00', freq='15min')
McClear.index = begin_time_McClear
# aggregate time series into 1 hour
# dataframe "McClear_agg_1h_raw" is to aggregate 00:00:00, 00:15:00, 00:30:00, 00:45:00. as 00:00:00
McClear_agg_1h_raw = McClear.resample("1h").sum()
# In the ECMWF dataset, the time is stamped at the end of the hour.
# Thus, the aggregate value of four period, namely, 00:00:00, 00:15:00, 00:30:00, and 00:45:00, is stamped at 01:00:00
McClear_agg_1h_raw.index = McClear_agg_1h_raw.index + pd.Timedelta("1h")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 30min in advance
# we also want to aggregate "00:30:00, 00:45:00, 01:00:00, and 01:15:00" as the 01:00:00
McClear_advance_30min = McClear.copy() 
McClear_advance_30min.index = McClear_advance_30min.index - pd.Timedelta("30min")
McClear_agg_1h_advance_30min = McClear_advance_30min.resample("1h").sum()
# In the ECMWF dataset, the time is stamped at the end of the hour.
McClear_agg_1h_advance_30min.index = McClear_agg_1h_advance_30min.index + pd.Timedelta("1h")

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Truncated valid period: 2017-01-01 00:01:00 ~ 2020-12-31 23:00:00
McClear_agg_1h_raw = McClear_agg_1h_raw["2017-01-01 01:00:00" : "2020-12-31 23:00:00"]
McClear_agg_1h_advance_30min = McClear_agg_1h_advance_30min["2017-01-01 01:00:00" : "2020-12-31 23:00:00"]


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# read ECMWF ensemble forecasts at Jacumba
metadata_ENS_Jacumba = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/Jacumba_ENS.csv', index_col=1)
metadata_ENS_Jacumba = metadata_ENS_Jacumba.iloc[:,1:]
# update index type
ENS_Jacumba_time = pd.date_range(start='2017-01-01 01:00:00', end='2020-12-31 23:00:00', freq='1h')
metadata_ENS_Jacumba.index = ENS_Jacumba_time
# ECMWF time stamp: e.g., "02:00:00" stands for period "01:00:00 ~ 02:00:00"



# calculate member beta-median (beta=-1)

# clear sky index

# define a new dataframe "data_target1" to store clear sky index
metadata_ENS_Jacumba_kappa = pd.DataFrame(columns=[f'EC_kappa_{x+1}' for x in range(50)], index=metadata_ENS_Jacumba.index)


for i in range(50):
    metadata_ENS_Jacumba_kappa.iloc[:,i] = metadata_ENS_Jacumba.iloc[:,i] / McClear_agg_1h_raw['Clear sky GHI']

# FIND zenith > 85 degree
# data_irr.loc[data_irr['Solar Zenith Angle'] > 85,'NSRDB_GHI'] = np.nan

# metadata_ENS_Jacumba_kappa = metadata_ENS_Jacumba_kappa.replace(0,np.nan)
metadata_ENS_Jacumba_kappa = metadata_ENS_Jacumba_kappa.dropna()

median_cdf = pd.DataFrame(index=metadata_ENS_Jacumba_kappa.index,columns=['median-1'])

for i in range(len(metadata_ENS_Jacumba_kappa)):
    # define a new dataframe "interim_target" to store the ensemble irradiance of each row
    interim_target = np.array(metadata_ENS_Jacumba_kappa.iloc[i,:50])
    interim_target = pd.DataFrame(data=interim_target, index=range(50),columns=['interim_irr'])
    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth = 0.01, kernel='gaussian')
    # fit the kernel density model on the ensemble data
    kde_fit = kde.fit(interim_target.iloc[:])
    # the range of irradiance, and "x_sample" stands for the sample spaces 
    x_sample = np.linspace(0.01, 2.5, 250)[:, np.newaxis]
    # compute the log-likelihood of each sample under the model
    log_dens = kde.score_samples(x_sample)
    # probability density
    dens = np.exp(log_dens)
    # a = (dens*0.001).sum()
    beta = -2
    
    fy_beta = x_sample**beta
    fy1 = pd.DataFrame(data=fy_beta,columns=['fy_beta'])
    density = pd.DataFrame(data=dens,columns=['density_raw'])
    density['yfy'] = fy1['fy_beta']*density['density_raw']
    density['fy_beta'] = fy1['fy_beta']
    # integrate for the total area under the scaled pdf
    # this area is the so-called "proportion" in Gneiting's paper
    area = 0.01*density['yfy'].sum()
    # density function of the new vaiable (i.e, the value of yf(y)/area, which integrates to 1)
    density['new_yfy'] = density['yfy'] / area
    # aa = (density['new_yfy']*0.001).sum()
    # use cumulative sum to find out at which index the area accumulated is closest to 0.5
    cdf_target = np.cumsum(0.01*density['new_yfy'])
    x_sample = pd.DataFrame(data=x_sample, columns=['samples'])
    media=0.5
    aim_index = (cdf_target-media).abs().argsort()[0]
    median_cdf.iloc[i,0] = x_sample.iloc[aim_index,0]

metadata_ENS_Jacumba_kappa['member_median_neg_1'] = median_cdf['median-1']




ENS_median_neg_1_Jacumba = pd.DataFrame(columns=['member_median_neg_1'], index=metadata_ENS_Jacumba.index)
ENS_median_neg_1_Jacumba['member_median_neg_1'] = metadata_ENS_Jacumba_kappa['member_median_neg_1']

# Using McClear to advance the current ECMWF forecasts by half an hour. define a new dataframe "ENS_mean_Jacumba_p"
# index stands for ECMWF time stamp: e.g., 01:30:00 is period "00:30:00 ~ 01:30:00"
ENS_median_neg_1_Jacumba_p = pd.DataFrame(columns=['member_median_neg_1'], index=ENS_median_neg_1_Jacumba.index)
ENS_median_neg_1_Jacumba_p['member_median_neg_1'] = ENS_median_neg_1_Jacumba['member_median_neg_1'] * McClear_agg_1h_advance_30min['Clear sky GHI']
ENS_median_neg_1_Jacumba_p.index = ENS_median_neg_1_Jacumba_p.index - pd.Timedelta("30min")
ENS_median_neg_1_Jacumba_p = ENS_median_neg_1_Jacumba_p['member_median_neg_1'].astype('float64')

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# download ECMWF HRES forecasts
# Actually, index in dataframe "metadata_HRES" is ECMWF time stamp. e.g., 2017-01-01 01:00:00 is period 2017-01-01 00:00:00 ~ 2017-01-01 01:00:00
# Note that we moved all the ensemble forecasts forward by half an hour. 
# Unfortunately, we have no way to align the time stamp of the ENS, which is pushed back half an hour, with the time stamp of the HRES.
metadata_HRES = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/ECMWF_HRES.csv', index_col=0)
# extract three variables, namely, "u10", "v10", "t2m"
weather_HRES = metadata_HRES[['u10','v10','t2m']]
# calculate wind speed by "u10" and "v10". (m/s)
# Eq.(1) in https://doi.org/10.1016/j.solener.2021.12.011
weather_HRES.insert(2,"wind_speed", np.sqrt((weather_HRES.u10)**2 + (weather_HRES.v10)**2))


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# zenith angle
# latitude, longitude
lat, lon = 32.6193, -116.13
# Times
# time of zenith 01:00:00 <---> time of ECMWF 01:30:00  
zenith_time = pd.date_range(start='2017-01-01 01:00:00', end='2020-12-31 23:00:00', freq='1h', tz='UTC')
# the position of the Sun
# spa_python: the solar positionig algorithm (SPA) is commonly regarded as the most accurate one to date.
position = pvlib.solarposition.spa_python(time=zenith_time, latitude=lat, longitude=lon)
# the position of the Sun is described by the solar azimuth and zenith angles.
zenith = position.zenith
zenith_angle = pd.DataFrame(columns=['ECMWF_time','zenith'],index=zenith.index)
zenith_angle['zenith'] = zenith
zenith_angle['ECMWF_time'] = zenith_angle.index - pd.Timedelta("30min")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------- 
# define a temporary dataframe
temp = ENS_median_neg_1_Jacumba_p.copy()
temp.index = zenith_time
# join "zenith" and "member_mean" in same dataframe named "data_target"
# data_target: index--->NSRDB, e.g., (2017-01-01 01:00:00+00:00) is period (2017-01-01 00:30:00+00:00 ~ 2017-01-01 01:30:00+00:00)
# data_target: ECMWF_time---->ECMWF, e.g., (2017-01-01 01:30:00) is period (2017-01-01 00:30:00+00:00 ~ 2017-01-01 01:30:00+00:00)
data_target = zenith_angle.copy()
data_target['member_median_neg_1'] = temp

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# check zenith versus clear sky solar irradiance
McClear_agg_1h_advance_30min.index = zenith_time 
McClear_agg_1h_advance_30min['zenith'] = zenith_angle['zenith']
# Any day is selected, and the user can change it at will
d = McClear_agg_1h_advance_30min.iloc[25:49,:]
# If on a clear day the rising path (morning) of the plot does not coincide with the falling path (afternoon) of the curve, 
# it is advisable to reexamine the assumptions on the time stamp.
plt.plot(d['zenith'], d['Clear sky GHI'])

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# estimate DNI and DHI from EC_GHI
# Separation modeling aims at splitting the beam and diffuse radiation components from the global one.
# When estimating global titled irradiance (GTI), both GHI and DHI are required.
# Estimate DNI and DHI from GHI using the DISC model.
# The DISC algorithm converts global horizontal irradiance to direct normal irradiance through empirical relationships between the global and direct clearness indices. 
irradiance = pvlib.irradiance.disc(ghi=data_target.member_median_neg_1, solar_zenith=data_target.zenith, datetime_or_doy=data_target.index)
irradiance['dhi'] = data_target['member_median_neg_1'] - np.cos(data_target['zenith']/180*np.pi)*irradiance['dni']

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# weather: input variable to model chain
# ghi: global horizontal irradiance; dhi: diffuse horizontal irradiacne; dni: direct normal irradiance; wind_speed: wind speed; temp_air: temperature
weather = pd.DataFrame(columns=['ghi','dhi','dni','wind_speed','temp_air'], index=data_target.index)
# ghi (W/m2)
weather['ghi'] = data_target['member_median_neg_1']
# dhi (W/m2)
weather['dhi'] = irradiance['dhi']
# dni (W/m2)
weather['dni'] = irradiance['dni']
# wind_speed (m/s)
weather['wind_speed'] = weather_HRES['wind_speed']
# air_temp (℃)
weather['temp_air'] = weather_HRES['t2m']


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# download modeled generation estimates (regard as real PV AC power)
# source: https://data.openei.org/submissions/4503
real_PV = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/model chain/data/60947.csv', index_col=0)
real_PV.index = pd.date_range(start='2017-01-01 08:00:00', end='2021-01-01 07:00:00', freq='1h', tz='UTC')


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PV model
# PVWatts 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Determine extraterrestrial radiation from day of year.
# 截取2020年
position2020 = position["2020-01-01 00:00:00" : "2020-12-31 23:00:00"]

weather2020 = weather["2020-01-01 00:00:00" : "2020-12-31 23:00:00"]
dni_extra = pvlib.irradiance.get_extra_radiation(datetime_or_doy=weather.index, method='nrel', epoch_year=2020)
weather2020['dni_extra'] = dni_extra
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Transposition model
# Determine total in-plane irradiance and its beam, sky diffuse and ground reflected components, using the perez model.
# I_{tot} = I_{beam} + I_{skydiffuse} + I_{ground}
transposition_irradiance = pvlib.irradiance.get_total_irradiance(surface_tilt=25, 
                                                                 surface_azimuth=180, 
                                                                 solar_zenith=position2020.apparent_zenith, 
                                                                 solar_azimuth=position2020.azimuth,
                                                                 dni=weather2020.dni,
                                                                 ghi=weather2020.ghi, 
                                                                 dhi=weather2020.dhi,
                                                                 dni_extra=weather2020.dni_extra, 
                                                                 model='perez')



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# the angle of incidence of the solar vector on a surface. This is the angle between the solar vector and the surface normal. 
# Input all angles in degrees.
aoi = pvlib.irradiance.aoi(surface_tilt=25, 
                           surface_azimuth=180, 
                           solar_zenith=position2020.apparent_zenith, 
                           solar_azimuth=position2020.azimuth)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reflection Loss model
# Fresnel:
# refraction index of the PV module cover material, npv=1.526 for normal, npv=1.3 for anti-reflection coated glass.
npv = 1.526
weather2020['aoi'] = aoi
weather2020['theta_RefractiveAngle'] = (np.arcsin(np.sin(weather2020.aoi/180*np.pi)/npv))/np.pi*180
    
Rd = (np.sin((weather2020.theta_RefractiveAngle-weather2020.aoi)/180*np.pi))**2 / (np.sin((weather2020.theta_RefractiveAngle+weather2020.aoi)/180*np.pi))**2 + \
     (np.tan((weather2020.theta_RefractiveAngle-weather2020.aoi)/180*np.pi))**2 / (np.tan((weather2020.theta_RefractiveAngle+weather2020.aoi)/180*np.pi))**2
     
R0 = ((npv-1)/(npv+1))**2    
# the physical relative transmittance for beam radiation (tau_b) is only based on Fresnel equations:
tau_b = (1-Rd/2)/(1-R0)

# tau_b using pvlib 
tau_b_pvlib = pvlib.iam.physical(aoi=weather2020.aoi)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Xie:
nt = 1.4585 # the refeaction index of the pyranometer cover, which is usually a fused silica dome with nt=1.4585
tilt_angle = 25 # tilt angle
S = tilt_angle/180*np.pi # radians
w = (npv*(nt+1)**2)/(nt*(npv+1)**2) * (2.77526E-9 + 3.74953*npv - 5.18727*(npv)**2 + 3.41186*(npv)**3 - 1.08794*(npv)**4 + 0.13606*(npv)**5)
# the relative transmittance for diffuse radiation (tau_d):
tau_d = 2*w/(np.pi*(1+np.cos(S)))*\
    (30/7*np.pi - 160/21*S - 10/3*np.pi*np.cos(S) + 160/21*np.cos(S)*np.sin(S) - \
     5/3*np.pi*np.cos(S)*(np.sin(S))**2 + 20/7*np.cos(S)*(np.sin(S))**3 - 5/16*np.pi*(np.sin(S))**4 + 16/105*np.cos(S)*(np.sin(S))**5)

# the relative transmittance for ground-reflected radiation (tau_g):
tau_g = 40*w/(21*(1-np.cos(S))) - tau_d*(1+np.cos(S))/(1-np.cos(S))
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# absorbed radiation (G'c):
# G'_c = tau_b*B_c + tau_d*D_c + tau_g*D_g
absorbed_radiation = tau_b*transposition_irradiance.poa_direct + tau_d*transposition_irradiance.poa_sky_diffuse + tau_g*transposition_irradiance.poa_ground_diffuse


cell_temperature = pvlib.temperature.pvsyst_cell(absorbed_radiation, weather2020.temp_air, weather2020.wind_speed)
PVWatts_DC = pvlib.pvsystem.pvwatts_dc(absorbed_radiation, cell_temperature, pdc0=28000000, gamma_pdc=-0.00711, temp_ref=25.0)
PVWatts_AC = pvlib.inverter.pvwatts(PVWatts_DC, 20000000/0.96, eta_inv_nom=0.96, eta_inv_ref=0.99)

# cell_temperature = pvlib.temperature.pvsyst_cell(transposition_irradiance.poa_global, weather2020.temp_air, weather2020.wind_speed)
# PVWatts_DC = pvlib.pvsystem.pvwatts_dc(transposition_irradiance.poa_global, cell_temperature, pdc0=28000000, gamma_pdc=-0.00711, temp_ref=25.0)
# PVWatts_AC = pvlib.inverter.pvwatts(PVWatts_DC, 20000000/0.96, eta_inv_nom=0.96, eta_inv_ref=0.99)




Results = pd.DataFrame(columns=['PV_AC','SAM_gen'], index=weather2020.index)

Results['PV_AC'] = PVWatts_AC

# unit MW
Results['PV_AC'] = Results['PV_AC']/1000000
Results['SAM_gen'] = real_PV['SAM_gen']

# insert zenith angle into dataframe "Results"

zenith_angle = zenith_angle["2020-01-01 00:00:00" : "2020-12-31 23:00:00"]
Results.insert(2, "zenith", zenith_angle.zenith)
Results.insert(3, "utc_time", pd.date_range(start='2020-01-01 00:00:00', end='2020-12-31 23:00:00', freq='1h'))
# Delete "PV_AC" and "SAM_gen" with zenith angle greater than 85 degrees.
Results.loc[Results['zenith'] > 85,'PV_AC'] = np.nan
Results.loc[Results['zenith'] > 85,'SAM_gen'] = np.nan
Results = Results.replace(0,np.nan)
Results = Results.dropna()
# RMSE nRMSE
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(Results.SAM_gen, Results.PV_AC, squared=False)

mean_measurements = Results.SAM_gen.mean()

nRMSE = rmse/mean_measurements



# MAE nMAE
# error metric: mean absolute error
def mae(y_true, y_pred):
    """
    Mean absolute error
    
    Parameters
    ----------
    y_true: array
        observed value
    y_pred: array
        forecasts 
    """
    return np.mean(np.abs((y_pred - y_true)))
mae_mean = mae(Results.SAM_gen, Results.PV_AC)


# MAPE
def mape(y_true, y_pred):
    """
    Absolute percentage error
    
    Parameters
    ----------
    y_true: array
        observed value
    y_pred: array
        forecasts 
    """  
    return ( np.mean( np.abs( (y_pred - y_true)/y_true ) ) ) * 100


mape = mape(Results.SAM_gen, Results.PV_AC)

# MRE
def re(y_true, y_pred):
    """
    Relative error
    
    Parameters
    ----------
    y_true: array
        observed value
    y_pred: array
        forecasts 
    """  
    return np.mean( np.abs( (y_pred - y_true)/y_pred ) )

mre = re(Results.SAM_gen, Results.PV_AC)



def rmspe(y_true, y_pred):
    """
    Root mean squared percentage error
    Parameters
    -----------
    y_ture: array
        observed value
    y_pred
        forecasts
    """
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100


rmspe = rmspe(Results.SAM_gen, Results.PV_AC)

