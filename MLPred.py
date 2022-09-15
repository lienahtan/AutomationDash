from re import X
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from numpy import asarray, insert
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
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
import datetime
import hydralit_components as hc
import time


def MLPred(perioddata, dateDict, alarmquery, rootcause_selected, endDate,):
    downtimeHolder = []
    period = st.slider('How many days do you want to predict ?', 0, 100, 7)
    if period > 30:
        st.warning("Your forecasting accuracy will be affected due to extended forecasting period.")

    for component_selected in alarmquery[10:11]:
        st.subheader("Forecast for " + component_selected)
        
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
        
        preddf['Count_LastDay'] = preddf['Count'].shift(+1)
        preddf['Count_2DayBack'] = preddf['Count'].shift(+2)
        preddf['Count_3DayBack'] = preddf['Count'].shift(+3)
        preddf['Count_4DayBack'] = preddf['Count'].shift(+4)
        preddf['Count_5DayBack'] = preddf['Count'].shift(+5)
        preddf['Count_6DayBack'] = preddf['Count'].shift(+6)
        preddf['Count_7DayBack'] = preddf['Count'].shift(+7)
        preddf['prev7day_rolling_avg'] = preddf['Count'].rolling(7).mean()
        preddf['prev7day_rolling_avg'] = preddf['prev7day_rolling_avg'].shift(+1)
        # to keep a copy of dates
        originaldf = preddf.copy()
        
        preddf = preddf.dropna()
        
        # finding the 80% of the dataset
        lengthOfData = len(preddf)
        lengthOfLast20percent = int(0.2 * lengthOfData)
        ntest = lengthOfLast20percent
    
        x1, x2, x3, x4, x5, x6, x7, x8, y = preddf['Count_LastDay'], preddf['Count_2DayBack'], preddf['Count_3DayBack'], preddf['Count_4DayBack'], preddf['Count_5DayBack'], preddf['Count_6DayBack'], preddf['Count_7DayBack'], preddf['prev7day_rolling_avg'] , preddf['Count']
        x1, x2, x3, x4, x5, x6, x7, x8, y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(x5), np.array(x6), np.array(x7),np.array(x8), np.array(y) 
        x1, x2, x3, x4, x5, x6, x7, x8, y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), x5.reshape(-1,1), x6.reshape(-1,1), x7.reshape(-1,1), x8.reshape(-1,1), y.reshape(-1,1)
        final_x = np.concatenate((x1, x2, x3, x4, x5, x6, x7, x8),axis=1)
        # st.write(x1)
        # st.write(final_x) 
        st.subheader("Dataset timeseries")
        st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:15px;">{"Original dataset with engineered features."}</h1>', unsafe_allow_html=True)
        
    
        # testing ValiditY 
        # using 80% 20% to train data
        Xtrain, Xtest, Ytrain, Ytest = final_x[:-ntest], final_x[-ntest:], y[:-ntest], y[-ntest:]
        # Comparing all algos
        st.subheader('Algorithm Comparison')
        st.markdown(f'<h2 style="text-align: left; color:#5F9EA0; font-size:15px;">{"Which machine learning model perfoms the best?"}</h1>', unsafe_allow_html=True)
        models = []
        models.append(('LR', LinearRegression()))
        models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
        models.append(('KNN', KNeighborsRegressor())) 
        models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
        models.append(('SVR', SVR(gamma='auto'))) # kernel = linearF
        
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
        
        
        # Grid Search, after realising that Random forest is the best one
        st.subheader("Conducting Gridsearch to find hyperparameters")
        selectedModel = st.radio(
        "Which model do you want to use for alarm: " + component_selected + "? (Application only supports Random Forest now)",
        ('Linear Regression', 'Neural Network', 'K Nearest Neighbours', 'Random Forest', 'Support Vector Regression'))
        
        if selectedModel == 'Linear Regression':
            model = LinearRegression()
        elif selectedModel == 'Neural Network':
            model = MLPRegressor(solver = 'lbfgs')
        elif selectedModel == 'K Nearest Neighbours':
            model = KNeighborsRegressor()
        elif selectedModel == 'Random Forest':
            model = RandomForestRegressor(n_estimators = 10)
        elif selectedModel == 'Support Vector Regression':
            model = SVR(gamma='auto')
        
        
        # KEY MODEL ACCURACY MEASURES
        st.subheader("Performance of Selected Model")
        with hc.HyLoader('Loading',hc.Loaders.pulse_bars,):
                        time.sleep(20)
        model = RandomForestRegressor()
        param_search = { 
            'n_estimators': [20, 50, 100, 1000],
            # 'max_features': ['auto', 'sqrt', 'log2'],
            # 'max_depth' : [i for i in range(5,15)]
            'random_state' : [1]
        }
        
        def regression_results(y_true, y_pred):
            # Regression metrics
            explained_variance=metrics.explained_variance_score(y_true, y_pred)
            mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
            mse=metrics.mean_squared_error(y_true, y_pred) 
            mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
            median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
            r2=metrics.r2_score(y_true, y_pred)
            st.write('explained_variance: ', round(explained_variance,4))    
            st.write('mean_squared_log_error: ', round(mean_squared_log_error,4))
            st.write('r2: ', round(r2,4))
            st.write('MAE: ', round(mean_absolute_error,4))
            st.write('MSE: ', round(mse,4))
            st.write('RMSE: ', round(np.sqrt(mse),4))
            
        def rmse(actual, predict):
            predict = np.array(predict)
            actual = np.array(actual)
            distance = predict - actual
            square_distance = distance ** 2
            mean_square_distance = square_distance.mean()
            score = np.sqrt(mean_square_distance)
            return score
        rmse_score = make_scorer(rmse, greater_is_better = False)
        
        tscv = TimeSeriesSplit(n_splits=5)
        gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
        
        # validity model is built on Xtrain and Ytrain
        gsearch.fit(Xtrain, Ytrain)
        best_score = gsearch.best_score_
        best_model = gsearch.best_estimator_
        st.write("Model used will be :")
        st.write(best_model)
        y_true = Ytest.tolist()
        y_pred = best_model.predict(Xtest)
        regression_results(y_true, y_pred)
        
        # visualising the model
        # model = RandomForestRegressor(n_estimators=1000, max_features=3, random_state=1)

        ForestDist = go.Figure(layout = format)
        
        ForestDist.add_trace(
                go.Scatter(
                    y = y_pred.tolist(),
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
        rmse_rf = sqrt(mean_squared_error(y_pred, Ytest)) 
        st.write(rmse_rf)
        st.plotly_chart(ForestDist, use_container_width= True)
        
        
#         # Ploting features importance
#         imp = best_model.feature_importances_
#         features = preddf.columns[1:].tolist()
#         indices = np.argsort(imp)
        
#         plotfeaturesImp = go.Figure(layout = format)
#         plotfeaturesImp.add_trace(
#                 go.Bar(
#                     name = "Feature Importance",
#                     x = indices,
#                     text = [features[i] for i in indices]
#                 )
#             )
#         st.plotly_chart(plotfeaturesImp, use_container_width= True)
        
        
        
        # forecasting is built on full_dataset to forecast future values
        howManySteps = period
        
        for step in range(howManySteps):
            # Getting the ONE STEP for prediction
            counttodayto7daysago = preddf['Count'][-7:].iloc[::-1]
            # Calculating 7 day average (this might need revision!!!)
            prev7day = counttodayto7daysago.sum()/7
            # Adding the 7 day average to series
            counttodayto7daysago = counttodayto7daysago.append(pd.Series(prev7day))
            # Transposing the Series
            onestep = np.array(counttodayto7daysago).reshape(1,-1)
            # FIT USING THE ENTIRE DATASET, TO FORECAST ONE STEP
            forecastmodel = model.fit(final_x, y)
            # prediciton (ouput: one value)
            forecastpred = forecastmodel.predict(onestep)
            # append the output to the holder to plot graph
            downtimeHolder.append(forecastpred)
            # output to insert back into preddf
            insertBack = onestep.copy()
            # UPDATES the 7 day average
            insertBack[0, 7] = np.sum(insertBack[: , 0:7])/7
            # inserts back into final_x for training model
            final_x = np.append(final_x , insertBack, axis=0)
            final_x = final_x[1:]
            # inserts back into y for training model
            y = np.append(y, forecastpred,)
            y = y[1:]
            # insert new results in the first index to insert into preddf
            copyonestep = onestep.copy()
            copyonestep = np.insert(copyonestep, 0 , forecastpred, axis= 1)
            # adding preddf date
            startDate = max(preddf.index)
            datenext = startDate + datetime.timedelta(days = 1)
            # UPDATES 7 day average and PREDDF
            copyonestep[0, 8] = np.sum(copyonestep[: , 0:8])/7
            frame = pd.DataFrame(data = copyonestep[:,:], index = [datenext], columns = preddf.columns)
            preddf = preddf.append(frame)
            originaldf = originaldf.append(frame)
        
        ForecastForestDist = go.Figure(layout = format)
        
        ForecastForestDist.add_trace(
            go.Scatter(
                x = originaldf.index[:-howManySteps].tolist(),
                y = originaldf['Count'][:-howManySteps].tolist(),
                name = 'Actual',
                visible = True,
            )
        )

        ForecastForestDist.add_trace(
            go.Scatter(
                x = originaldf.index[-howManySteps:].tolist(),
                y = originaldf['Count'][-howManySteps:].tolist(),
                name = 'Forecasted',
                visible = True,
            )
        )
            
        # ForecastForestDist.add_trace(
        #         go.Scatter(
        #             x = [i for i in range(len(forecastpred))],
        #             y = [i[0] for i in y.tolist()],
        #             name = 'DT',
        #             visible = True,
        #         )
        #     )
        
        # ForecastForestDist.add_trace(
        #         go.Scatter(
        #             x = [i for i in range(len(forecastpred))],
        #             y = forecastpred.tolist(),
        #             name = 'Out of Dataste Random Forest on Actual DT',
        #             visible = True,
        #         )
        #     )
        
        st.subheader("FORECASTED random forest")
        st.plotly_chart(ForecastForestDist, use_container_width= True)
        
    st.write(downtimeHolder)
        
        
        # counttodayto7daysago = preddf['Count'][-7:].iloc[::-1]
        # prev7day = counttodayto7daysago.sum()/7
        # counttodayto7daysago = counttodayto7daysago.append(pd.Series(prev7day))
        # onestep = np.array(counttodayto7daysago).reshape(1,-1)
        # st.write(onestep)

        # forecastmodel = model.fit(final_x, y)

        # forecastpred = forecastmodel.predict(onestep)
        # forecasteHolder.append(forecastpred)
        # st.write(forecastpred)
    
    #   insert new results in the first index
        # copyonestep = onestep.copy()
        # copyonestep = np.insert(copyonestep, 0 , forecastpred, axis= 1)
        # st.write(copyonestep)
        
        # delete column 7
        # copyonestep = np.delete(copyonestep, 7, 1)
        # st.write(copyonestep)
        
        # startDate = max(preddf.index)
        # st.write(startDate)
        # datenext = startDate + datetime.timedelta(days = 1)
        # st.write(datenext)
        
        
        # # UPDATES the 7 day average
        # copyonestep[0, 8] = np.sum(copyonestep[: , 0:8])/7
        # st.write(copyonestep)
        # frame = pd.DataFrame(data = copyonestep[:,:], index = [datenext], columns = preddf.columns)
        # st.write(frame)
        # preddf = preddf.append(frame)
        # st.write(preddf)
        # ForecastForestDist = go.Figure(layout = format)
        
        # # st.write(final_x)
        # final_x = np.append(final_x ,onestep, axis=0)
        # final_x = final_x[1:]
        # # st.write(final_x)
        
        # # st.write(type(forecastpred))
        # # st.write(type(y))
        # y = np.append(y, forecastpred,)
        # y = y[1:]
        # # st.write(y)
    
        
        
        # ####################### TESTING FOR LINEAR MODEL ###################
        # lin_model = lin_model.fit(Xtrain, Ytrain)
        # lin_pred = lin_model.predict(Xtest)
        
        # linearDist = go.Figure(layout = format)
        
        # linearDist.add_trace(
        #         go.Scatter(
        #             y = [i[0] for i in lin_pred.tolist()],
        #             name = 'Linear Reg',
        #             visible = True,
        #         )
        #     )
            
        # linearDist.add_trace(
        #         go.Scatter(
        #             y = [i[0] for i in Ytest.tolist()],
        #             name = 'Actual sales',
        #             visible = True,
        #         )
        #     )
    
        # st.subheader("RMSE for linear")
        # rmse_lr = sqrt(mean_squared_error(lin_pred, Ytest))
        # st.write(rmse_lr)
        # st.plotly_chart(linearDist, use_container_width= True)


        # tsfresh
        # df_pass, y_air = make_forecasting_frame(preddf["Count"], kind="Count", max_timeshift=12, rolling_direction=1)





        # return [ForestDist, rmse_rf, ForecastForestDist]
        # return ForestDist
        
