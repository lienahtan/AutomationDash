import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from numpy import asarray
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def MLPred(perioddata, dateDict, component_selected, rootcause_selected, endDate, period):
    period = 7
    filteredDf = perioddata.loc[perioddata['DT Reason Detail'] == component_selected]
    # filteredDf = filteredDf.loc[perioddata['Diagonstics'] == rootcause_selected]
    
    format = go.Layout(
        margin=go.layout.Margin(
                l=0, #left margin
                r=0, #right margin(
                b=0, #bottom margin
                t=0, #top margin
            ),
        # plot_bgcolor="#FFFFFF"
        )
    
    timeDict = {}
    
    timeDict.update({component_selected: dateDict.copy()})
    
    for row in filteredDf.values.tolist():
        timeDict[row[5]][row[0].strftime("%#d/%#m/%Y")] += float(row[4])
        

    date = list(list(timeDict.values())[0].keys())
    count = list(list(timeDict.values())[0].values())
    d = {'Count':count, 'ds':date}
    preddf = pd.DataFrame(d)
    preddf[['Day', 'Month', 'Year']] = preddf['ds'].str.split('/', expand=True)
    preddf['ds'] = pd.DatetimeIndex(preddf['Year']+ '/' + preddf['Month'] + '/' + preddf['Day'])
    preddf.drop(['Day', 'Month', 'Year'], axis= 1, inplace=True)
    
    # preddf['y'] = np.log(preddf['Y'])
    # st.write(preddf)
    
    preddf = preddf.set_index('ds')
    
    preddf['Count_LastMonth'] = preddf['Count'].shift(+1)
    preddf['Count_2MonthsBack'] = preddf['Count'].shift(+2)
    preddf['Count_3MonthsBack'] = preddf['Count'].shift(+3)
    preddf = preddf.dropna()
    
    # finding the 80% of the dataset
    lengthOfData = len(preddf)
    lengthOfLast20percent = 0.2 * lengthOfData
    ntest = int(lengthOfLast20percent)
   
    x1, x2, x3, y = preddf['Count_LastMonth'], preddf['Count_2MonthsBack'], preddf['Count_3MonthsBack'] , preddf['Count']
    x1, x2, x3, y = np.array(x1), np.array(x2), np.array(x3), np.array(y) 
    x1, x2, x3, y = x1.reshape(-1,1), x1.reshape(-1,1), x3.reshape(-1,1), y.reshape(-1,1)
    final_x = np.concatenate((x1, x2, x3),axis=1)
    # st.write(x1)
    # st.write(final_x) 
    st.subheader("Dataset timeseries")
    st.write(preddf)
   
    # testing ValiditY 
    # using 80% 20% to train data
    Xtrain, Xtest, Ytrain, Ytest = final_x[:-ntest], final_x[-ntest:], y[:-ntest], y[-ntest:]
    
    # models from the internet
    st.subheader('Algorithm Comparison')
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
    models.append(('KNN', KNeighborsRegressor())) 
    models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
    models.append(('SVR', SVR(gamma='auto'))) # kernel = linear
    
    results = []
    names = []
    for name, model in models:
        # TimeSeries Cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_results = cross_val_score(model, Xtrain, Ytrain, cv=tscv, scoring='neg_mean_squared_error')
        results.append(cv_results)
        names.append(name)
        st.write('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        
    fig = plt.figure(figsize=(10, 4))
    plt.boxplot(results, labels=names)
    st.pyplot(fig)
    
    # Own models
    lin_model = LinearRegression()
    model = RandomForestRegressor(n_estimators=1000, max_features=3, random_state=1)
    

    # validity model is built on Xtrain and Ytrain
    model = model.fit(Xtrain, Ytrain)
    lin_model = lin_model.fit(Xtrain, Ytrain)
   
    pred = model.predict(Xtest)
        
    ForestDist = go.Figure(layout = format)
    
    ForestDist.add_trace(
            go.Scatter(
                y = pred.tolist(),
                name = 'Random Forest',
                visible = True,
            )
        )
        
    ForestDist.add_trace(
            go.Scatter(
                y = [i[0] for i in Ytest.tolist()],
                name = 'DT',
                visible = True,
            )
        )
    st.subheader("RMSE for random forest")
    rmse_rf = sqrt(mean_squared_error(pred, Ytest)) 
    st.write(rmse_rf)
    st.plotly_chart(ForestDist, use_container_width= True)
    
    # validity model is built on full_dataset to forecast future values
    forecastmodel = model.fit(final_x, y)
    
    forecastpred = forecastmodel.predict(final_x)
    
    ForecastForestDist = go.Figure(layout = format)
    
    ForecastForestDist.add_trace(
            go.Scatter(
                x = [i for i in range(len(forecastpred), len(forecastpred) + len(y))],
                y = forecastpred.tolist(),
                name = 'Out of Dataste Random Forest',
                visible = True,
            )
        )
        
    ForecastForestDist.add_trace(
            go.Scatter(
                x = [i for i in range(len(forecastpred))],
                y = [i[0] for i in y.tolist()],
                name = 'DT',
                visible = True,
            )
        )
    
    ForecastForestDist.add_trace(
            go.Scatter(
                x = [i for i in range(len(forecastpred))],
                y = forecastpred.tolist(),
                name = 'Out of Dataste Random Forest on Actual DT',
                visible = True,
            )
        )
      
    st.subheader("FORECASTED random forest")
    st.plotly_chart(ForecastForestDist, use_container_width= True)
    
    
    lin_pred = lin_model.predict(Xtest)
    
    linearDist = go.Figure(layout = format)
    
    linearDist.add_trace(
            go.Scatter(
                y = [i[0] for i in lin_pred.tolist()],
                name = 'Linear Reg',
                visible = True,
            )
        )
        
    linearDist.add_trace(
            go.Scatter(
                y = [i[0] for i in Ytest.tolist()],
                name = 'Actual sales',
                visible = True,
            )
        )
   
    st.subheader("RMSE for linear")
    rmse_lr = sqrt(mean_squared_error(lin_pred, Ytest))
    st.write(rmse_lr)
    st.plotly_chart(linearDist, use_container_width= True)


    # tsfresh
    # df_pass, y_air = make_forecasting_frame(preddf["Count"], kind="Count", max_timeshift=12, rolling_direction=1)





    # return [ForestDist, rmse_rf, ForecastForestDist]
    # return ForestDist
    
