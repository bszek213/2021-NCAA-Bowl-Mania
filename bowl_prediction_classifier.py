"""
Predicting bowl game outcomes

@author: Brian
"""
from sportsipy.ncaaf.teams import Teams
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

class bowl_games_class:
    def __init__(self):
        team_data_2010 = pd.Series(dtype = float)
    
    def input_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-t1", "--team1", help = "team 1 input")
        parser.add_argument("-t2", "--team2", help = "team 2 input")
        self.args = parser.parse_args()
        
    def get_teams(self, year):
        all_teams = Teams(year)
        save_team = {}
        counter = 0
        for team in all_teams:
            if team.games > 0 or team.games:
                # schedule = team.schedule  # Returns a Schedule instance for each team
                df = team.schedule.dataframe_extended
                team_features = {}
                team_features['name'] = df.winning_abbr
                team_features['wins'] = team.wins
                team_features['away_first_downs'] = df.away_first_downs
                team_features['away_fumbles'] = df.away_fumbles
                team_features['away_fumbles_lost'] = df.away_fumbles_lost
                team_features['away_interceptions'] = df.away_interceptions
                team_features['away_pass_attempts'] = df.away_pass_attempts
                team_features['pass_att'] = df.away_penalties
                team_features['away_points'] = df.away_points
                team_features['away_total_yards'] = df.away_total_yards
                team_features['away_turnovers'] = df.away_turnovers
                team_features['away_yards_from_penalties'] = df.away_yards_from_penalties
                team_features['home_first_downs'] = df.home_first_downs #This will be my label
                team_features['home_fumbles'] = df.home_fumbles
                team_features['home_fumbles_lost'] = df.home_fumbles_lost
                team_features['home_interceptions'] = df.home_interceptions
                team_features['home_pass_attempts'] = df.home_pass_attempts
                team_features['home_pass_completions'] = df.home_pass_completions
                team_features['home_pass_touchdowns'] = df.home_pass_touchdowns
                team_features['home_pass_yards'] = df.home_pass_yards
                team_features['home_penalties'] = df.home_penalties
                team_features['home_points'] = df.home_points
                team_features['home_rush_attempts'] = df.home_rush_attempts
                team_features['home_rush_touchdowns'] = df.home_rush_touchdowns
                team_features['home_rush_yards'] = df.home_rush_yards
                team_features['home_total_yards'] = df.home_total_yards
                team_features['home_total_yards'] = df.home_total_yards
                team_features['home_total_yards'] = df.home_total_yards
                team_features['points_per_game'] = team.points_per_game
                # team_features['outcome'] = df.winner.replace('Home', 1)
                # team_features['outcome'] = team_features['outcome'].replace('Away', -1)
                save_team[counter] = team_features
                counter = counter + 1

        data_transformed_df = pd.DataFrame.from_dict(save_team).T
        return data_transformed_df 

    def split(self, pandas_df):
        temp_df = pandas_df
        y = temp_df['wins']
        temp_df = temp_df.drop(columns=['wins','name'])
        # scaler = MinMaxScaler()
        # scaled_data = scaler.fit_transform(temp_df)
        # cols = temp_df.columns
        # x = pd.DataFrame(scaled_data, columns = cols)
        x = temp_df
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, train_size=0.8)
        
    def machine(self):
        #Cross validate all to find one that works the best
        GB = cross_validate(GradientBoostingRegressor(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        RF = cross_validate(RandomForestRegressor(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        LR = cross_validate(LinearRegression(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        DT = cross_validate(DecisionTreeRegressor(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        SV_R = cross_validate(SVR(kernel='linear'),self.x_train, self.y_train, cv=10,
                              scoring=['neg_root_mean_squared_error'],return_train_score=True)
        KNN = cross_validate(KNeighborsRegressor(),self.x_train, self.y_train, cv=10,
                              scoring=['neg_root_mean_squared_error'],return_train_score=True)
        GPR = cross_validate(GaussianProcessRegressor(),self.x_train, self.y_train, cv=10,
                              scoring=['neg_root_mean_squared_error'],return_train_score=True)
        MLP = cross_validate(MLPRegressor(),self.x_train, self.y_train, cv=10,
                              scoring=['neg_root_mean_squared_error'],return_train_score=True)
        
        DT_err = np.abs(np.mean(DT['test_neg_root_mean_squared_error']))
        GB_err = np.abs(np.mean(GB['test_neg_root_mean_squared_error']))
        SV_R_err = np.abs(np.mean(SV_R['test_neg_root_mean_squared_error']))
        RF_err = np.abs(np.mean(RF['test_neg_root_mean_squared_error']))
        LR_err = np.abs(np.mean(LR['test_neg_root_mean_squared_error']))
        KNN_err = np.abs(np.mean(KNN['test_neg_root_mean_squared_error']))
        GPR_err = np.abs(np.mean(GPR['test_neg_root_mean_squared_error']))
        MLP_err = np.abs(np.mean(MLP['test_neg_root_mean_squared_error']))
        
        
        print('results of DT: ',DT_err)
        print('results of GB: ',GB_err)
        print('results of SV_R: ',SV_R_err)
        print('results of RF: ',RF_err)
        print('results of LR: ',LR_err)
        print('results of KNN: ',KNN_err)
        print('results of GPR: ',GPR_err)
        print('results of MLP: ',MLP_err)
        
        err_list = [DT_err, GB_err, SV_R_err, RF_err, LR_err]
        lowest_error = err_list.index(min(err_list))
        
        #Return the best performing alg
        if lowest_error == 0:
            final_model = DecisionTreeRegressor().fit(self.x_test,self.y_test)
            print(r2_score(self.y_test, final_model.predict(self.x_test)))
            return final_model, 'DecisionTreeRegressor'
        elif lowest_error == 1:
            final_model = GradientBoostingRegressor().fit(self.x_test,self.y_test)
            print(r2_score(self.y_test, final_model.predict(self.x_test)))
            return final_model, 'GradientBoostingRegressor'
        if lowest_error == 2:
            final_model = SVR().fit(self.x_test,self.y_test)
            print(r2_score(self.y_test, final_model.predict(self.x_test)))
            return final_model, 'SVR'
        elif lowest_error == 3: 
            final_model = RandomForestRegressor().fit(self.x_test,self.y_test)
            print(r2_score(self.y_test, final_model.predict(self.x_test)))
            return final_model, 'RandomForestRegressor'
        if lowest_error == 4:
            final_model = LinearRegression().fit(self.x_test,self.y_test)
            print(r2_score(self.y_test, final_model.predict(self.x_test)))
            return final_model, 'LinearRegression'
    
    def parameter_tuning(self, final_model, name):
        
        if name == 'DecisionTreeRegressor':
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
            max_features = ['auto', 'sqrt', 'log2']
            max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            min_weight_fraction_leaf = [float(x) for x in np.linspace(start = 0, stop = 10, num = 5)]
            random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_weight_fraction_leaf':min_weight_fraction_leaf,
                   'min_samples_leaf': min_samples_leaf}
            mod = DecisionTreeRegressor()
            test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
            test_model.fit(self.x_train,self.y_train) #Do I use self.x_test,self.y_test or self.x_train,self.y_train?
            print('print best params: ',test_model.best_params_)
            return test_model
        elif name == 'GradientBoostingRegressor':
            learning_rate = [float(x) for x in np.linspace(start = 0.001, stop = 1, num = 10)]
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            random_grid = {'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
            mod = GradientBoostingRegressor()
            test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
            test_model.fit(self.x_train,self.y_train)
            print('print best params: ',test_model.best_params_)
            return test_model
        elif name == 'RandomForestRegressor':
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
            max_depth.append(None)
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
            mod = RandomForestRegressor()
            test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
            test_model.fit(self.x_train,self.y_train)
            print('print best params: ',test_model.best_params_)
            return test_model
        elif name == 'SVR':
            tol = [float(x) for x in np.linspace(start = 0.001, stop = 1, num = 10)]
            kernel = ['linear???', 'poly', 'rbf', 'sigmoid', 'precomputed']
            gamma = ['scale', 'auto']
            C = [float(x) for x in np.linspace(1, 20, num = 15)]
            bootstrap = [True, False]
            random_grid = {'tol': tol,
                    'kernel': kernel,
                    'C': C,
                    'gamma': gamma
            }
            mod = SVR()
            test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
            test_model.fit(self.x_train,self.y_train)
            print('print best params: ',test_model.best_params_)
            return test_model
        elif name == 'LinearRegression':
            print('Linear Regressor has no hyperparameters')
            mod = LinearRegression().fit(self.x_train,self.y_train)
            return mod               

    def compare_two_teams(self, pandas_df, model):
        
        team1_loc = np.where(self.args.team1 == pandas_df['name'])
        team2_loc = np.where(self.args.team2 == pandas_df['name'])
        
        team_1_compare = pandas_df.iloc[team1_loc]
        team_2_compare = pandas_df.iloc[team2_loc]
        
        team_1_compare = team_1_compare.drop(columns=['wins','name'])
        team_2_compare = team_2_compare.drop(columns=['wins','name'])
        
        team1_np =team_1_compare.to_numpy()
        team2_np =team_2_compare.to_numpy()
    
        diff = [a - b for a, b in zip(team1_np, team2_np)]
        arr = np.array(diff)
        nx, ny = arr.shape
        final_vector = arr.reshape((1,nx*ny))
    
        diff = [b - a for a, b in zip(team1_np, team2_np)]
        arr = np.array(diff)
        nx, ny = arr.shape
        final_vector2 = arr.reshape((1,nx*ny))
        
        print(f"Probability that {self.args.team1} will win is {model.predict(final_vector)}")
        print(f"Probability that {self.args.team2} will win is {model.predict(final_vector2)}")

if __name__ == '__main__':
    start_time = time.time()
    
    team_csv = Teams(2021)
    pd.DataFrame(team_csv).to_csv('allteams.csv')
    
    bowl = bowl_games_class()
    bowl.input_arg()
    combine_list = {}
    data_2008 = bowl.get_teams(2008)
    data_2009 = bowl.get_teams(2009)
    data_2010 = bowl.get_teams(2010)
    data_2011 = bowl.get_teams(2011)
    data_2012 = bowl.get_teams(2012)
    data_2013 = bowl.get_teams(2013)
    data_2014 = bowl.get_teams(2014)
    data_2015 = bowl.get_teams(2015)
    data_2016 = bowl.get_teams(2016)
    data_2017 = bowl.get_teams(2017)
    data_2018 = bowl.get_teams(2018)
    data_2019 = bowl.get_teams(2019)
    data_2021 = bowl.get_teams(2021)
    all_data = pd.concat([data_2008, data_2009, data_2010, data_2011, data_2012, data_2013, data_2014,data_2015,data_2016,
                     data_2017, data_2018, data_2019,
                     data_2021],ignore_index=True)
    print(f"number of samples: {len(all_data)}")
    
    bowl.split(all_data)
    final_model, name = bowl.machine()
    mod = bowl.parameter_tuning(final_model, name)
    bowl.compare_two_teams(data_2021, mod)
    # bowl.plot_feature_importance_classify(mod, name)
    
    print("--- %s seconds ---" % (time.time() - start_time))