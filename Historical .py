#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:18:36 2020

@author: fanfan
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#import data
returns = pd.read_csv('Returns_Clean.csv')

#assume a window size
w=1000

#create an empty table for mean
returns_table=returns[['Dates']].copy()

#save all the historial data
returns_historical = returns.drop(columns = ['Dates'])

#calculate rolling window mean
for col in returns_historical.columns:
    returns_table[col] = returns_historical[col].rolling(w).mean()

#remove date and NaN from mean table 
mean_table=returns_table.drop(returns_historical.index[0:w-1])
mean_table=mean_table.drop(returns_historical.index[-1])
mean_table=mean_table.drop(columns = ['Dates'])

#remove NaN from historical data 
returns_historical = returns_historical.drop(returns_historical.index[0:w])

#reset the index
mean_table = mean_table.reset_index(drop=True)
returns_historical = returns_historical.reset_index(drop=True)

#get the error 
new_df=mean_table-returns_historical

#get the error square
error_square=pd.DataFrame()
for i in range(0,958):
    for j in range(0,100):
        error_square.set_value(i,j,(new_df.iloc[i,j])**2)
        
#column mean  of error   
error_square_mean=error_square.mean(1)

#squared 
squared_error_mean = np.sqrt(np.array(error_square_mean))

data_final = pd.DataFrame()
data_final["Dates"] = returns["Dates"][1000:]
data_final["RMSE"] = squared_error_mean

#plot
#change date type 
data_final["Dates"] = pd.to_datetime(pd.Series(data_final['Dates']), format="%Y%m%d")
plt.scatter(data_final["Dates"], data_final["RMSE"])
plt.xlabel('Date')
plt.ylabel('RMSE')




        