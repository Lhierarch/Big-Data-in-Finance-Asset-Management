# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:14:21 2020

@author: Xinyu Zhang
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import math 
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#load data
returns = pd.read_csv('Returns_Clean.csv')
flows = pd.read_csv('Flows_Clean.csv')
#rename column name of the trading flows
flows.columns = [str(col) + '_flows' for col in flows.columns]
#put all data into the dataframe
returns_lag1=returns.iloc[:,1:101].shift(1)
returns_lag2=returns.iloc[:,1:101].shift(2) 
returns_lag3=returns.iloc[:,1:101].shift(3)
returns_lagged=returns.join(returns_lag1.rename(columns=lambda x:str(x)+"_lag1"))
returns_lagged=returns_lagged.join(returns_lag2.rename(columns=lambda x:str(x)+"_lag2"))
returns_lagged=returns_lagged.join(returns_lag3.rename(columns=lambda x:str(x)+"_lag3"))   
flows_lag1=flows.iloc[:,1:101].shift(1)
flows_lag2=flows.iloc[:,1:101].shift(2)
flows_lag3=flows.iloc[:,1:101].shift(3)
flows_lagged=flows.join(flows_lag1.rename(columns=lambda x:str(x)+"_lag1"))
flows_lagged=flows_lagged.join(flows_lag2.rename(columns=lambda x:str(x)+"_lag2"))
flows_lagged=flows_lagged.join(flows_lag3.rename(columns=lambda x:str(x)+"_lag3"))
data = pd.concat([returns_lagged, flows_lagged], axis=1)
#delete the second Dates column
del data["Dates_flows"]
#as we have 3 lages, regression will start from 4th period
data = data.iloc[3:,]
#creat a list to store all column names
name_list = list(returns.columns)
name_list.remove("Dates")

#delete dates column in returns
del returns["Dates"]
#create a list to store MSE for each window size
MSE_w = []
#choose window size for linear regression
for w in range(30,200):
    #create a empty dataframe with column names only
    prediction_w = pd.DataFrame(columns=name_list)
    for name in name_list:
        rolling_beta = RollingOLS(data[name], data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3"]], 
                     window=w).fit()
        pred_list = []
        for i in range(w,1955):
            para = np.array(rolling_beta.params.iloc[i-1,:])
            value = np.array(data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3"]].iloc[i])
        #inner product of parameters and values in order to predict
            pred = np.dot(para, value)
            pred_list.append(pred)
        prediction_w[name] = pred_list
    returns_compare_w = returns.iloc[-(1955-w):,:]
    pred_error_w = pd.DataFrame()
    #put square errors into the new dataframe
    for i in range(0,1955-w):
        for j in range(0,100):    
            pred_error_w.set_value(i, j, (prediction_w.iloc[i,j] - returns_compare_w.iloc[i,j])**2)
    #name each columns with stock tickers
    pred_error_w.columns = name_list
    #calculate time series mean then cross section mean
    m = pred_error_w.mean()
    mean = m.mean()
    #store mean in the list
    MSE_w.append(mean)
#find the index of the minimum value in the MSE list
ind = MSE_w.index(min(MSE_w))
#find the best window size for regression
window_size = ind + 30
print("The best window size for regression is {}".format(window_size))

#create an empty table for mean
returns_table=returns[['Dates']].copy()
#save all the historial data
returns_historical = returns.drop(columns = ['Dates'])
#choosing window size for historical mean
MSE_w_HM = []
for w in range(30,200):
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
    for i in range(0,1958-w):
        for j in range(0,100):
            error_square.set_value(i,j,(new_df.iloc[i,j])**2)
    row_mean = error_square.mean(axis = 1)
    mean = row_mean.mean()
    MSE_w_HM.append(mean)
ind_hm = MSE_w_HM.index(min(MSE_w))
#find the best window size for regression
window_size_HM = ind_hm + 30
print("The best window size for regression is {}".format(window_size_HM))  
        
    






#calculate historical mean
#assume a window size
w = 200
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




#create a empty dataframe to store predictions
prediction = pd.DataFrame(columns=name_list)
for name in name_list:
    rolling_beta = RollingOLS(data[name], data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3"]], 
                     window=1000).fit()
    pred_list = []
   # pred = rolling_beta.predict(data[[name+"_lag1",name+"_lag2", name+"_lag3", 
    #                 name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3"]])
    for i in range(1000,1955):
        para = np.array(rolling_beta.params.iloc[i-1,:])
        value = np.array(data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3"]].iloc[i])
        pred = np.dot(para, value)
        pred_list.append(pred)
    prediction[name] = pred_list

returns_compare = returns.iloc[-955:,:]
#comput prediction error
pred_error = pd.DataFrame()#(columns=name_list)
for i in range(0,955):
    for j in range(0,100):    
        pred_error.set_value(i, j, (prediction.iloc[i,j] - returns_compare.iloc[i,j])**2)
#name each columns with stock tickers
pred_error.columns = name_list
#comput means of each column
RMSE_TS = np.sqrt(np.array(pred_error.mean(axis = 1)))
#create a dataframe to plot
data_plot = pd.DataFrame()
data_plot["Dates"] = data["Dates"][-955:]
data_plot["RMSE"] = RMSE_TS
#change date type 
data_plot["Dates"] = pd.to_datetime(pd.Series(data_plot['Dates']), format="%Y%m%d")
#plot
plt.scatter(data_plot["Dates"], data_plot["RMSE"])
plt.xlabel('Date')
plt.ylabel('RMSE')

#maching learning 
#compute correlation matrix
corr_matrix = returns.corr()
#order correlations between stocks
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False))

#seperate training and test data
training_data = data.iloc[:1555,:]
test_data = data.iloc[-400:,:]
#acquire two highest correlations for each stock 
dic1 = {}
dic2 = {}
for name in name_list:
    a = corr_matrix[name].nlargest(3)
    dic1.update({name: a.index[1]})
    dic2.update({name: a.index[2]})
#set parameters for grid cross-validation
tscv = TimeSeriesSplit(max_train_size=None, n_splits=5)
param_grid = [{'lasso__alpha': np.logspace(100, 0, 50)}]
#create a list to store each dictionary with best lambdas for each stock
best_lam_all_lasso = []
#conduct lasso using regression
for name in name_list:
    #get cross-sectional returns and trading flows from the dictionary
    cross1 = dic1.get(name)
    cross2 = dic2.get(name)
    #create a dictionary to sotre best lambda values for each window
    best_window = {}
    for i in range(0,955):
        x_train = training_data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3",
                     cross1+"_lag1", cross1+"_lag2", cross1+"_lag3",
                     cross1+"_flows_lag1", cross1+"_flows_lag2", cross1+"_flows_lag3",
                     cross2+"_lag1", cross2+"_lag2", cross2+"_lag3",
                     cross2+"_flows_lag1", cross2+"_flows_lag2", cross2+"_flows_lag3"]].iloc[i:1000+i]
        y_train = training_data[name][i:1000+i]
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter = 100000))
        ])
        grid_cv = GridSearchCV(estimator=pipe, cv=tscv, param_grid=param_grid, n_jobs = -1)
        grid_cv.fit(x_train, y_train)
        best_window.update({i: grid_cv.best_params_['lasso__alpha']})
        i = i + 1
    #add dictionary into list
    best_lam_all_lasso.append(best_window)
#trial one stock
    """
for i in range(0,955):
        x_train = training_data[["50286"+"_lag1","50286"+"_lag2", "50286"+"_lag3", 
                     "50286"+"_flows_lag1", "50286"+"_flows_lag2", "50286"+"_flows_lag3"]].iloc[i:1000+i]
        y_train = training_data[name][i:1000+i]
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter = 100000))
        ])
        grid_cv = GridSearchCV(estimator=pipe, cv=tscv, param_grid=param_grid, n_jobs = -1)
        grid_cv.fit(x_train, y_train)
    
x_train = training_data[["50286"+"_lag1","50286"+"_lag2", "50286"+"_lag3", 
                     "50286"+"_flows_lag1", "50286"+"_flows_lag2", "50286"+"_flows_lag3"]]
y_train = training_data["50286"]
pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter = 100000))])
grid_cv = GridSearchCV(estimator=pipe, cv=tscv, param_grid=param_grid, n_jobs = -1)
grid_cv.fit(x_train, y_train)
    """

#random forest
#set parameters
param_grid_RF = [{'RF__n_estimators':[50, 75, 100], 
                'RF__max_features':['auto', 'sqrt'], 
                'RF__max_depth':[6, 7]
              }]
#creat a list for randomForest
best_params_RF = []
for name in name_list:
    #get cross-sectional returns and trading flows from the dictionary
    cross1 = dic1.get(name)
    cross2 = dic2.get(name)
    #create a dictionary to sotre best lambda values for each window
    best_window = []
    for i in range(0,955):
        x_train = training_data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3",
                     cross1+"_lag1", cross1+"_lag2", cross1+"_lag3",
                     cross1+"_flows_lag1", cross1+"_flows_lag2", cross1+"_flows_lag3",
                     cross2+"_lag1", cross2+"_lag2", cross2+"_lag3",
                     cross2+"_flows_lag1", cross2+"_flows_lag2", cross2+"_flows_lag3"]].iloc[i:1000+i]
        y_train = training_data[name][i:1000+i]
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('RF', RandomForestRegressor(random_state = 111, n_jobs = -1))
        ])
        grid_RF = GridSearchCV(estimator=pipe, cv=tscv, param_grid=param_grid_RF, n_jobs = -1, verbose =1)
        grid_RF.fit(x_train, y_train)
        best_window.append(grid_RF.best_params_)
#put results for all stocks into the list
best_params_RF.append(best_window)     


#random forest trial for the first stock
best_para_1 = []
for i in range(0,955):
        x_train = training_data[["50286"+"_lag1","50286"+"_lag2", "50286"+"_lag3", 
                     "50286"+"_flows_lag1", "50286"+"_flows_lag2", "50286"+"_flows_lag3",
                     cross1+"_lag1", cross1+"_lag2", cross1+"_lag3",
                     cross1+"_flows_lag1", cross1+"_flows_lag2", cross1+"_flows_lag3",
                     cross2+"_lag1", cross2+"_lag2", cross2+"_lag3",
                     cross2+"_flows_lag1", cross2+"_flows_lag2", cross2+"_flows_lag3"]].iloc[i:1000+i]
        y_train = training_data["50286"][i:1000+i]
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('RF', RandomForestRegressor(random_state = 111, n_jobs = -1))
        ])
        grid_RF = GridSearchCV(estimator=pipe, cv=tscv, param_grid=param_grid_RF, n_jobs = -1, verbose =1)
        grid_RF.fit(x_train, y_train)
        best_para_1.append(grid_RF.best_params_)


#cLassification
#calculate percentiles of each period in original dataset
percentile = returns.quantile([.33, .66], axis = 1)  
#delete first 3 columns of pecentile 
percentile = percentile.drop(percentile.columns[0:3], axis=1) 
#transfer returns into percentiles
for i in range(0,1955):
    for j in range(1,101):
        if data.iloc[i,j] <= percentile.iloc[0,i]:
            data.iloc[i,j] = 0
        elif data.iloc[i,j] > percentile.iloc[1,i]:
            data.iloc[i,j] = 2
        else:
            data.iloc[i,j] = 1
#use machine learning(random forest) method to predict
rf = RandomForestClassifier(n_estimators=100, random_state=0)
#creae a empty dataframe to store predicted values
pred_class = pd.DataFrame()
#use window size of 200 for now
for name in name_list:
    #create a list to store all prediction for a particular stock
    pred = []
    #select cross-sectional independent variables
    cross1 = dic1.get(name)
    cross2 = dic2.get(name)
    for i in range(200,1955):
        x_class = data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3",
                     cross1+"_lag1", cross1+"_lag2", cross1+"_lag3",
                     cross1+"_flows_lag1", cross1+"_flows_lag2", cross1+"_flows_lag3",
                     cross2+"_lag1", cross2+"_lag2", cross2+"_lag3",
                     cross2+"_flows_lag1", cross2+"_flows_lag2", cross2+"_flows_lag3"]][:i]
        y_class = data[name][:i]
        x_t = data[[name+"_lag1",name+"_lag2", name+"_lag3", 
                     name+"_flows_lag1", name+"_flows_lag2", name+"_flows_lag3",
                     cross1+"_lag1", cross1+"_lag2", cross1+"_lag3",
                     cross1+"_flows_lag1", cross1+"_flows_lag2", cross1+"_flows_lag3",
                     cross2+"_lag1", cross2+"_lag2", cross2+"_lag3",
                     cross2+"_flows_lag1", cross2+"_flows_lag2", cross2+"_flows_lag3"]].iloc[i]
        #reshape xtest in order to predict one preiod return type
        x_t = np.array(x_t).reshape(1,-1)
        rf.fit(x_class, y_class)
        #here, prediction is in form of array
        pr = rf.predict(x_t)
        #transform each predicion into int
        pred.append(int(pr))
    pred_class[name] = pred 

#compare out-of-sample prediction with actual classification
accurate = 0
for i in range(0, 1755):
    for j in range(1,101):
        if pred_class.iloc[i,j] == data.iloc[i+200, j]:
            accurate += 1
accurate_per = accurate / (1755*100)
        
'''        
#classification trial
pred1 = []
for i in range(200,1955):
        x_class = data[["50286"+"_lag1","50286"+"_lag2", "50286"+"_lag3", 
                     "50286"+"_flows_lag1", "50286"+"_flows_lag2", "50286"+"_flows_lag3"]][:i]
        y_class = data[name][:i]
        x_t = data[["50286"+"_lag1","50286"+"_lag2", "50286"+"_lag3", 
                     "50286"+"_flows_lag1", "50286"+"_flows_lag2", "50286"+"_flows_lag3"]].iloc[i]
        #y_t = y_class = data[name][i]
        #reshape xtest in order to predict one preiod return type
        x_t = np.array(x_t).reshape(1,-1)
        rf.fit(x_class, y_class)
        pr = rf.predict(x_t)
        pred1.append(int(pr))
'''

#save variables
import numpy as np
import dill
filename= 'cw2result.pkl'
dill.dump_session(filename)


