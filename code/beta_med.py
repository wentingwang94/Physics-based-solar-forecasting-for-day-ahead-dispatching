# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:09:24 2022

@author: 81095
"""


import pandas as pd
import numpy as np
import pvlib
from scipy import special
import scipy.stats as stats
from sklearn import preprocessing
from scipy.misc import derivative
from scipy.optimize import Bounds
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KernelDensity
from statsmodels.distributions.empirical_distribution import ECDF

#---------------------------------------------------------------------------------------------------------------
# load data
metadata = pd.read_csv('C:/WWT/论文/ECMWF_ENS/supplementary material/beta_meidan/data/BON_ENS.csv', index_col=0)
# find zenith > 85 degree
metadata.loc[metadata['Solar Zenith Angle'] > 85,'NSRDB_GHI'] = np.nan
metadata = metadata.dropna()


# clear sky index
# define a new dataframe "data_target1" to store clear sky index
data_kappa = pd.DataFrame(columns=[f'EC_kappa_{x+1}' for x in range(50)], index=metadata.index)
for i in range(50):
    data_kappa.iloc[:,i] = metadata.iloc[:,i] / metadata['NSRDB_ClearskyGHI']


#---------------------------------------------------------------------------------------------------------------
# beta_median function
def beta_med(data_target, beta):
    median_cdf = pd.DataFrame(index=data_kappa.index,columns=['median'])
    for i in range(len(data_target)):
        # define a new dataframe "interim_target" to store the ensemble irradiance of each row
        interim_target = np.array(data_target.iloc[i,:50])
        interim_target = pd.DataFrame(data=interim_target, index=range(50),columns=['interim_irr'])
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth = 0.01, kernel='gaussian')
        # fit the kernel density model on the ensemble data
        kde_fit = kde.fit(interim_target.iloc[:])
        # the range of irradiance, and "x_sample" stands for the sample spaces 
        x_sample = np.linspace(0.01, 1.3, 130)[:, np.newaxis]
        # compute the log-likelihood of each sample under the model
        log_dens = kde.score_samples(x_sample)
        # probability density
        dens = np.exp(log_dens)
        # a = (dens*0.01).sum()
        # beta = 1
    
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
    return median_cdf

#---------------------------------------------------------------------------------------------------------------
# mean function
def ens_mean(data_target):
    ens_mean = pd.DataFrame(index=data_kappa.index,columns=['mean'])
    for i in range(len(data_target)):
        # define a new dataframe "interim_target" to store the ensemble irradiance of each row
        interim_target = np.array(data_target.iloc[i,:50])
        interim_target = pd.DataFrame(data=interim_target, index=range(50),columns=['interim_irr'])
        # instantiate and fit the KDE model
        kde = KernelDensity(bandwidth = 0.01, kernel='gaussian')
        # fit the kernel density model on the ensemble data
        kde_fit = kde.fit(interim_target.iloc[:])
        # the range of irradiance, and "x_sample" stands for the sample spaces 
        x_sample = np.linspace(0.01, 1.3, 130)[:, np.newaxis]
        # compute the log-likelihood of each sample under the model
        log_dens = kde.score_samples(x_sample)
        # probability density
        dens = np.exp(log_dens)
        # a = (dens*0.01).sum()
        density = pd.DataFrame(data=dens,columns=['density_raw'])
        # cdf_target = np.cumsum(0.01*density['new_yfy'])
        x_sample = pd.DataFrame(data=x_sample, columns=['samples'])
        aim_mean = 0.01*(density['density_raw'] * x_sample['samples']).sum()
        ens_mean.iloc[i,0] = aim_mean
    return ens_mean

#----------------------------------------------------------------------------------------------------------------
# clear sky -- solar irradiance
beta_med_0 = beta_med(data_kappa, 0)
beta_med_1 = beta_med(data_kappa, 1)
beta_med_n2 = beta_med(data_kappa, -2)
member_mean = ens_mean(data_kappa)

data_kappa['beta_med_kappa_0'] = beta_med_0['median']
data_kappa['beta_med_kappa_1'] = beta_med_1['median']
data_kappa['beta_med_kappa_n2'] = beta_med_n2['median']


data_kappa['NSRDB_GHI'] = metadata['NSRDB_GHI']
data_kappa['NSRDB_ClearskyGHI'] = metadata['NSRDB_ClearskyGHI']

data_kappa['beta_med_0'] = data_kappa['beta_med_kappa_0'] * data_kappa['NSRDB_ClearskyGHI']
data_kappa['beta_med_1'] = data_kappa['beta_med_kappa_1'] * data_kappa['NSRDB_ClearskyGHI']
data_kappa['beta_med_n2'] = data_kappa['beta_med_kappa_n2'] * data_kappa['NSRDB_ClearskyGHI']

# member_mean
data_kappa['mean_kappa'] = data_kappa.iloc[:,0:50].mean(axis=1)
data_kappa['member_mean'] = data_kappa['mean_kappa'] * data_kappa['NSRDB_ClearskyGHI']

#--------------------------------------------------------------------------------------------------------------
# score function
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

#----------------------------------------------------------------------------------------------------------------
# MAE
mae_mean = mean_absolute_error(data_kappa['NSRDB_GHI'], data_kappa['member_mean'])
mae_median = mean_absolute_error(data_kappa['NSRDB_GHI'], data_kappa['beta_med_0'])
mae_median1 = mean_absolute_error(data_kappa['NSRDB_GHI'], data_kappa['beta_med_1'])
mae_median_n2 = mean_absolute_error(data_kappa['NSRDB_GHI'], data_kappa['beta_med_n2'])

# RMSE
rmse_mean = mean_squared_error(data_kappa['NSRDB_GHI'], data_kappa['member_mean'], squared=False)
rmse_median = mean_squared_error(data_kappa['NSRDB_GHI'], data_kappa['beta_med_0'], squared=False)
rmse_median1 = mean_squared_error(data_kappa['NSRDB_GHI'], data_kappa['beta_med_1'], squared=False)
rmse_median_n2 = mean_squared_error(data_kappa['NSRDB_GHI'], data_kappa['beta_med_n2'], squared=False)

# RE
re_mean = re(data_kappa['NSRDB_GHI'], data_kappa['member_mean'])
re_median = re(data_kappa['NSRDB_GHI'], data_kappa['beta_med_0'])
re_median1 = re(data_kappa['NSRDB_GHI'], data_kappa['beta_med_1'])
re_median_n2 = re(data_kappa['NSRDB_GHI'], data_kappa['beta_med_n2'])

# RMSPE
rmspe_mean = rmspe(data_kappa['NSRDB_GHI'], data_kappa['member_mean'])
rmspe_median = rmspe(data_kappa['NSRDB_GHI'], data_kappa['beta_med_0'])
rmspe_median1 = rmspe(data_kappa['NSRDB_GHI'], data_kappa['beta_med_1'])
rmspe_median_n2 = rmspe(data_kappa['NSRDB_GHI'], data_kappa['beta_med_n2'])




