#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 19:42:27 2022

@author: umair
"""

import numpy as np
import errors as err
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import seaborn as sns

def linfunc(x, a, b):
    """ Function for fitting
    x: independent variable
    a, b: parameters to be fitted
    """
    y = a*x + b
    return y

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

def read_data(name):
    '''
    this function take file name and after reading the file return original data and transposed data
    Parameters
    ----------
    name : contain the name of dataset.

    Returns
    -------
    data : return a dataframe that contain indicators of world ecnomic.

    '''
    data=pd.read_csv(name,encoding= 'unicode_escape',skip_blank_lines=True, na_filter=True)
    return data,data.T
    
def data_exploration(data,country,series):
    '''
    this function take parameters and then return the series data that is required
    Parameters
    ----------
    data : is dataframe that contain data indicators for cleaning.
    country : contain the name of country that need to be taken into consideration.
    series : contain the name of series that is going to b used for analysis.

    Returns
    -------
    series_data : return the series data that we need from the whole dataset.

    '''
    country_data=data[data['Country Name']==country]
    series_data=country_data[country_data["Series Name"]==series]
    return series_data
def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled
def norm_df(df, first=0, last=None):
    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df
def data_fitting(data,country,indicator,col_name):
    '''
    this function take four parameters and perform exponentional function in curve fit for good fitting 
    Parameters
    ----------
    data : is dataframe that contain data indicators for cleaning.
    country : contain the name of country that need to be taken into consideration.
    indicator : contain the name of series that is going to b used for analysis.
    col_name : this variable contain the name of column to b used in the dafaframe.

    Returns
    -------
    None.

    '''
    gdp_data=data_exploration(data,country,indicator)
    gdp_data=gdp_data.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    g=gdp_data.transpose()
    g.columns=[col_name]
    g=g.reset_index()
    g=g.rename(columns={"index": "Year"})
    g=g.dropna()
    g["Year"] = pd.to_numeric(g["Year"])
    

    param, covar = opt.curve_fit(logistic, g["Year"], g[col_name],p0=(3e12, 0.03,2000.0))
    sigma = np.sqrt(np.diag(covar))
    g["fit"] = logistic(g["Year"], *param)
    g["trial"] = logistic(g["Year"], 3e12, 0.10, 1990)
    g.plot("Year", [col_name, "fit"])
    plt.title(country,fontsize=18)
    plt.xlabel('year',fontsize=18)
    plt.ylabel(col_name,fontsize=18)
    plt.show()
    year = np.arange(1960, 2031)
    low, up = err.err_ranges(year, logistic, param, sigma)
    forecast = logistic(year, *param)

    plt.figure()
    plt.plot(g["Year"],g[col_name], label=col_name)
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year",fontsize=18)
    plt.title(country,fontsize=18)
    plt.ylabel(col_name,fontsize=18)
    plt.legend(),
    plt.show()
    print("Forcasted population")
    low, up = err.err_ranges(2030, logistic, param, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2030:", mean, "+/-", pm)
    print("2030 between ", low, "and", up)
    low, up = err.err_ranges(2040, logistic, param, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2040:", mean, "+/-", pm)
    print("2040 between ", low, "and", up)
    low, up = err.err_ranges(2050, logistic, param, sigma)
    mean = (up+low) / 2.0
    pm = (up-low) / 2.0
    print("2050:", mean, "+/-", pm)
    print("2050 between ", low, "and", up)
    
def clustering_data(data,country,indicator,col_name,second_indicator,indicator_name):
    '''
    this function take six parameters and then train kmean model and perform clustring on data and then perform plot
    
    Parameters
    ----------
    data : is dataframe that contain data indicators for cleaning.
    country : contain the name of country that need to be taken into consideration.
    indicator : contain the name of series that is going to b used for analysis.
    col_name : this vaiable contain the name of dataframe.
    second_indicator : this variable contain the name of second indicator.
    indicator_name : this variable contain the name of second column name.

    Returns
    -------
    None.

    '''
    gdp_data=data_exploration(data,country,indicator)
    gdp_data=gdp_data.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    second_data=data_exploration(data,country,second_indicator)
    second_data=second_data.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    g=gdp_data.transpose()
 
    g.columns=[col_name]
    g[indicator_name]=second_data.T
    print(g)
    g=g.reset_index()
    g=g.rename(columns={"index": "Year"})
    g=g.dropna()
    g[indicator_name] = pd.to_numeric(g[indicator_name])
    g = g[[col_name,indicator_name]].copy()
    
    # normalise dataframe and inspect result
    # normalisation is done only on the extract columns. .copy() prevents
    # changes in df_fit to affect df_fish. This make the plots with the
    # original measurements
    g = norm_df(g)
    print(g.describe())
    plt.figure()
    plt.plot(g[indicator_name], g[col_name], "+")
    plt.xlabel(indicator_name)
    plt.ylabel(col_name)
    plt.show()
    df_ex = g[[indicator_name, col_name]].copy()
    max_val = df_ex.max()
    min_val = df_ex.min()
    df_ex = (df_ex - min_val) / (max_val - min_val)
    # print(df_ex)
    # set up the clusterer for number of clusters
    ncluster = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_ex) # fit done on x,y pairs
    labels = kmeans.labels_
    # print(labels) # labels is the number of the associated clusters of (x,y)points
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_ 
    print(cen)
    # calculate the silhoutte score
    print(skmet.silhouette_score(df_ex, labels))
    # plot using the labels to select colour
   
    print(cen)
    df_cen = pd.DataFrame(cen, columns=[indicator_name, col_name])
    print(df_cen)
    df_cen = df_cen * (max_val - min_val) + max_val
    df_ex = df_ex * (max_val - min_val) + max_val
    
    print(df_cen)
    # plot using the labels to select colour
    plt.figure(figsize=(10.0, 10.0))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for l in range(ncluster): # loop over the different labels
        plt.plot(df_ex[labels==l][indicator_name], df_ex[labels==l][col_name], \
        "o", markersize=3, color=col[l],label="Cluster"+str(l))
        #
        # show cluster centres
    plt.plot(df_cen[indicator_name], df_cen[col_name], "dk", markersize=10)
    plt.xlabel(indicator_name,fontsize=18)
    plt.ylabel(col_name,fontsize=18)
    plt.title(country,fontsize=18)
    plt.legend(loc="upper left")
    plt.show()
    
def correlation_graph(data,country):
    '''
    this function take two parameters and then plot heat map correlation of different indicators
    Parameters
    ----------
    data : is dataframe that contain data indicators for cleaning.
    country : contain the name of country that need to be taken into consideration.

    Returns
    -------
    None.

    '''
    co_burning=data_exploration(data,country,"Emission Totals - Emissions (CO2eq) (AR5) - Burning - Crop residues")
    co_crop=data_exploration(data,country,"Emission Totals - Emissions (CO2eq) (AR5) - Crop Residues")
    gdp=data_exploration(data,country,"GDP (current US$)")
    gdp=gdp.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    co_burning=co_burning.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    co_crop=co_crop.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    indicators_data=pd.DataFrame()
    indicators_data["CO2-Residues"]=co_crop.T
    indicators_data["Burning - Crop residues"]=co_burning.T
    indicators_data["GDP"]=gdp.T
    print(indicators_data)
    ax = plt.axes()
   
    sns.heatmap(indicators_data.corr(),annot=True, ax = ax)
    ax.set_title(country)
    plt.show()
    
    
    
    
    
def clustering_normalized_data(data,country,indicator,col_name,second_indicator,indicator_name):
    '''
    in this function we cluster the normalized data and show in the scatter plot
    Parameters
    ----------
   data : is dataframe that contain data indicators for cleaning.
   country : contain the name of country that need to be taken into consideration.
   indicator : contain the name of series that is going to b used for analysis.
   col_name : this vaiable contain the name of dataframe.
   second_indicator : this variable contain the name of second indicator.
   indicator_name : this variable contain the name of second column name.

    Returns
    -------
    None.

    '''
    gdp_data=data_exploration(data,country,indicator)
    gdp_data=gdp_data.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    second_data=data_exploration(data,country,second_indicator)
    second_data=second_data.drop(["Country Name","Country Code","Series Name","Series Code"], axis='columns')
    g=gdp_data.transpose()
 
    g.columns=[col_name]
    g[indicator_name]=second_data.T
    print(g)
    g=g.reset_index()
    g=g.rename(columns={"index": "Year"})
    g=g.dropna()
    g[indicator_name] = pd.to_numeric(g[indicator_name])
    g = g[[col_name,indicator_name]].copy()
    
    # normalise dataframe and inspect result
    # normalisation is done only on the extract columns. .copy() prevents
    # changes in df_fit to affect df_fish. This make the plots with the
    # original measurements
    g = norm_df(g)
    print(g.describe())
    for ic in range(2, 7):
        # set up kmeans and fit
        kmeans = cluster.KMeans(n_clusters=ic)
        kmeans.fit(g)
        # extract labels and calculate silhoutte score
        labels = kmeans.labels_
        print (ic, skmet.silhouette_score(g, labels))
        
    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(g)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))
    
    # l-th number from the colour table.
    plt.scatter(g[col_name], g[indicator_name], c=labels, cmap="Accent")
    # colour map Accent selected to increase contrast between colours
    # show cluster centres
    for ic in range(3):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel(col_name)
    plt.ylabel(indicator_name)
    plt.title(str("Clustrig Data Indicator of ")+country)
    plt.show()
    
   
    


if __name__ == "__main__":
    data,transpose_data=read_data("data.csv")
    fig = plt.figure()
   
    data_fitting(data,"Pakistan","GDP (current US$)","GDP")
    data_fitting(data,"Pakistan","Emission Totals - Emissions (CO2eq) (AR5) - Burning - Crop residues","CO2 Burning - Crop residues")
    clustering_data(data,"Pakistan","GDP (current US$)","gdp","Emission Totals - Emissions (CO2eq) (AR5) - Crop Residues","Crop Residues")
    
    data_fitting(data,"India","GDP (current US$)","GDP")
    data_fitting(data,"India","Emission Totals - Emissions (CO2eq) (AR5) - Burning - Crop residues","CO2 Burning - Crop residues")
    clustering_data(data,"India","GDP (current US$)","gdp","Emission Totals - Emissions (CO2eq) (AR5) - Crop Residues","Crop Residues")
    correlation_graph(data,"Pakistan")
    correlation_graph(data,"India")
    clustering_normalized_data(data,"India","GDP (current US$)","gdp","Emission Totals - Emissions (CO2eq) (AR5) - Crop Residues","Crop Residues")
    
   


