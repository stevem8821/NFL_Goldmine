import pandas as pd

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