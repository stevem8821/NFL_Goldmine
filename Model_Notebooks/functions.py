import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import xgboost
from sklearn.model_selection import RandomizedSearchCV
import math
import sklearn

def get_two_week_average(df_data):
    df = pd.DataFrame()

    for team in df_data['Tm'].sort_values().unique():
        for year in df_data['Year'].sort_values().unique():
            df_ex = df_data[(df_data['Tm'] == team) & (df_data['Year'] == year)].sort_values('Week').reset_index().drop(columns = 'index')
            for row in range(0,len(df_ex.index)-2):
                avg_rows = df_ex[(df_ex.index == row) | (df_ex.index == row + 1)] 
                keep_columns = ['Year','Week','Tm','Opp','Home/Away','Score','QB','is_home']
                df_keep = df_ex[df_ex.index == row + 2][keep_columns].reset_index().drop(columns = 'index')
                df_avg = avg_rows.drop(columns = ['Year','Week','Tm','Opp','Home/Away','QB','is_home'])
                df_averages = pd.DataFrame(df_avg.mean()).transpose().reset_index().drop(columns = 'index')
                df_averages = round(df_averages,2)
                df_averages = df_averages.add_prefix('Avg_')
                df_final = pd.concat([df_keep,df_averages],axis = 1)
                df = df.append(df_final)
                
    df_opponent = df.copy()
    df_opponent = df_opponent.drop(columns = ['Opp','Home/Away','Score','is_home'])
    df_opponent = df_opponent.add_prefix('Opp_')
    df_opponent = df_opponent.rename(columns={"Opp_Year":"Year","Opp_Week":"Week","Opp_Tm": "Tm"})
    df = df.merge(df_opponent,left_on = ['Year','Week','Opp'],right_on = ['Year','Week','Tm'])
    df = df.drop(columns = 'Tm_y')
    df = df.rename(columns = {'Tm_x':'Tm'})
    
    return df

def get_three_week_average(df_data):

    df = pd.DataFrame()

    for team in df_data['Tm'].sort_values().unique():
        for year in df_data['Year'].sort_values().unique():
            df_ex = df_data[(df_data['Tm'] == team) & (df_data['Year'] == year)].sort_values('Week').reset_index().drop(columns = 'index')
            for row in range(0,len(df_ex.index)-3):
                avg_rows = df_ex[(df_ex.index == row) | (df_ex.index == row + 1) | (df_ex.index == row + 2)] 
                keep_columns = ['Year','Week','Tm','Opp','Home/Away','Score','QB','is_home']
                df_keep = df_ex[df_ex.index == row + 3][keep_columns].reset_index().drop(columns = 'index')
                df_avg = avg_rows.drop(columns = ['Year','Week','Tm','Opp','Home/Away','QB','is_home'])
                df_averages = pd.DataFrame(df_avg.mean()).transpose().reset_index().drop(columns = 'index')
                df_averages = round(df_averages,2)
                df_averages = df_averages.add_prefix('Avg_')
                df_final = pd.concat([df_keep,df_averages],axis = 1)
                df = df.append(df_final)

    df_opponent = df.copy()
    df_opponent = df_opponent.drop(columns = ['Opp','Home/Away','Score','is_home'])
    df_opponent = df_opponent.add_prefix('Opp_')
    df_opponent = df_opponent.rename(columns={"Opp_Year":"Year","Opp_Week":"Week","Opp_Tm": "Tm"})
    df = df.merge(df_opponent,left_on = ['Year','Week','Opp'],right_on = ['Year','Week','Tm'])
    df = df.drop(columns = 'Tm_y')
    df = df.rename(columns = {'Tm_x':'Tm'})
    
    return df

def get_four_week_average(df_data):

    df = pd.DataFrame()

    for team in df_data['Tm'].sort_values().unique():
        for year in df_data['Year'].sort_values().unique():
            df_ex = df_data[(df_data['Tm'] == team) & (df_data['Year'] == year)].sort_values('Week').reset_index().drop(columns = 'index')
            for row in range(0,len(df_ex.index)-4):
                avg_rows = df_ex[(df_ex.index == row) | (df_ex.index == row + 1) | 
                                 (df_ex.index == row + 2) | (df_ex.index == row + 3)] 
                keep_columns = ['Year','Week','Tm','Opp','Home/Away','Score','QB','is_home']
                df_keep = df_ex[df_ex.index == row + 4][keep_columns].reset_index().drop(columns = 'index')
                df_avg = avg_rows.drop(columns = ['Year','Week','Tm','Opp','Home/Away','QB','is_home'])
                df_averages = pd.DataFrame(df_avg.mean()).transpose().reset_index().drop(columns = 'index')
                df_averages = round(df_averages,2)
                df_averages = df_averages.add_prefix('Avg_')
                df_final = pd.concat([df_keep,df_averages],axis = 1)
                df = df.append(df_final)
                
    df_opponent = df.copy()
    df_opponent = df_opponent.drop(columns = ['Opp','Home/Away','Score','is_home'])
    df_opponent = df_opponent.add_prefix('Opp_')
    df_opponent = df_opponent.rename(columns={"Opp_Year":"Year","Opp_Week":"Week","Opp_Tm": "Tm"})
    df = df.merge(df_opponent,left_on = ['Year','Week','Opp'],right_on = ['Year','Week','Tm'])
    df = df.drop(columns = 'Tm_y')
    df = df.rename(columns = {'Tm_x':'Tm'})
    
    return df

def get_corr_and_plots(var1,var2,df_2_weeks,df_3_weeks,df_4_weeks):
    plt.figure(figsize = (15,15))
    plt.subplot(311)
    print('2 Weeks','Correlation:', np.corrcoef(df_2_weeks[var1],df_2_weeks[var2])[0,1])
    sns.regplot(df_2_weeks[var1],df_2_weeks[var2])
    plt.subplot(312)
    print('3 Weeks','Correlation:', np.corrcoef(df_3_weeks[var1],df_3_weeks[var2])[0,1])
    sns.regplot(df_3_weeks[var1],df_3_weeks[var2])
    plt.subplot(313)
    print('4 Weeks','Correlation:', np.corrcoef(df_4_weeks[var1],df_4_weeks[var2])[0,1])
    sns.regplot(df_4_weeks[var1],df_4_weeks[var2])
    return
   
def get_differentials(df):
    df['Passer_Rating_Dif'] = df['Avg_Passer_Rating'] - df['Opp_Def_Avg_Passer_Rating']
    df['Total_Yds_Dif'] = df['Avg_Total_Yds'] - df['Opp_Def_Avg_Total_Yds']
    df['Score_Dif'] = df['Avg_Score'] - df['Opp_Def_Avg_Score']
    df['Pass_Yds_Dif'] = df['Avg_Pass_Yds'] - df['Opp_Def_Avg_Pass_Yds']
    df['Rush_Yds_Dif'] = df['Avg_Rush_Yds'] - df['Opp_Def_Avg_Rush_Yds']
    df['Punts_Dif'] = df['Avg_Punts'] - df['Opp_Def_Avg_Punts']
    return df

def get_two_week_defense_average(df_data):
    df = pd.DataFrame()

    for opp in df_data['Opp'].sort_values().unique():
        for year in df_data['Year'].sort_values().unique():
            df_ex = df_data[(df_data['Opp'] == opp) & (df_data['Year'] == year)].sort_values('Week').reset_index().drop(columns = 'index')
            for row in range(0,len(df_ex.index)-2):
                avg_rows = df_ex[(df_ex.index == row) | (df_ex.index == row + 1)] 
                keep_columns = ['Year','Week','Opp']
                df_keep = df_ex[df_ex.index == row + 2][keep_columns].reset_index().drop(columns = 'index')
                df_avg = avg_rows.drop(columns = keep_columns)
                df_averages = pd.DataFrame(df_avg.mean()).transpose().reset_index().drop(columns = 'index')
                df_averages = round(df_averages,2)
                df_averages = df_averages.add_prefix('Opp_Def_Avg_')
                df_final = pd.concat([df_keep,df_averages],axis = 1)
                df = df.append(df_final)
    return df
    
def get_three_week_defense_average(df_data):
    df = pd.DataFrame()

    for opp in df_data['Opp'].sort_values().unique():
        for year in df_data['Year'].sort_values().unique():
            df_ex = df_data[(df_data['Opp'] == opp) & (df_data['Year'] == year)].sort_values('Week').reset_index().drop(columns = 'index')
            for row in range(0,len(df_ex.index)-3):
                avg_rows = df_ex[(df_ex.index == row) | (df_ex.index == row + 1) | (df_ex.index == row + 2)]
                keep_columns = ['Year','Week','Opp']
                df_keep = df_ex[df_ex.index == row + 3][keep_columns].reset_index().drop(columns = 'index')
                df_avg = avg_rows.drop(columns = keep_columns)
                df_averages = pd.DataFrame(df_avg.mean()).transpose().reset_index().drop(columns = 'index')
                df_averages = round(df_averages,2)
                df_averages = df_averages.add_prefix('Opp_Def_Avg_')
                df_final = pd.concat([df_keep,df_averages],axis = 1)
                df = df.append(df_final)
    return df

def get_four_week_defense_average(df_data):
    df = pd.DataFrame()

    for opp in df_data['Opp'].sort_values().unique():
        for year in df_data['Year'].sort_values().unique():
            df_ex = df_data[(df_data['Opp'] == opp) & (df_data['Year'] == year)].sort_values('Week').reset_index().drop(columns = 'index')
            for row in range(0,len(df_ex.index)-4):
                avg_rows = df_ex[(df_ex.index == row) | (df_ex.index == row + 1) | (df_ex.index == row + 2) | (df_ex.index == row + 3)]
                keep_columns = ['Year','Week','Opp']
                df_keep = df_ex[df_ex.index == row + 4][keep_columns].reset_index().drop(columns = 'index')
                df_avg = avg_rows.drop(columns = keep_columns)
                df_averages = pd.DataFrame(df_avg.mean()).transpose().reset_index().drop(columns = 'index')
                df_averages = round(df_averages,2)
                df_averages = df_averages.add_prefix('Opp_Def_Avg_')
                df_final = pd.concat([df_keep,df_averages],axis = 1)
                df = df.append(df_final)
    return df

def xgbrf_hypertuned_regression(Features,Target,param_grid,random_state,cv,n_iter,scoring,test_size):
    
    # First create the base model to tune
    model = xgboost.XGBRFRegressor(random_state = random_state)
    # search across 100 different combinations, and use all available cores
    xgbrf_regression_random = RandomizedSearchCV(model, param_distributions = param_grid, 
                                   n_iter = n_iter, cv = cv, scoring = scoring, verbose=2, 
                                   random_state=random_state, n_jobs = -1)
    # Fit the random search model
    xgbrf_regression_random.fit(Features, Target)
    
    #best params to feed to a final model
    best_params = xgbrf_regression_random.best_params_
    
    final_model = xgboost.XGBRFRegressor(random_state = random_state,
                                        subsample = best_params['subsample'],
                                        n_estimators = best_params['n_estimators'],
                                        max_depth = best_params['max_depth'],
                                        gamma = best_params['gamma'],
                                        eta = best_params['eta'],
                                        colsample_bytree = best_params['colsample_bytree'],
                                        colsample_bynode = best_params['colsample_bynode'],
                                        booster = best_params['booster'])
    
    X_train, X_test, y_train, y_test = train_test_split(Features,Target,test_size = test_size, random_state = random_state)
    
    final_model.fit(X_train,y_train)
    preds = final_model.predict(X_test)
    MSE = sklearn.metrics.mean_squared_error(y_test,preds)
    RMSE = math.sqrt(MSE)
    return preds, MSE, RMSE