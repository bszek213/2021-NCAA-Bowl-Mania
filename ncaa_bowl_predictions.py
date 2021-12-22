# -*- coding: utf-8 -*-
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import r2_score
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.feature_selection import SelectKBest, chi2

class bowl_games:
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
                team_features = {}
                team_features['name'] = team.abbreviation
                team_features['wins'] = team.wins
                # team_features['loss'] = team.losses
                # team_features['SOS'] = team.strength_of_schedule
                team_features['fum_lost'] = team.fumbles_lost
                team_features['int'] = team.interceptions
                # team_features['opp_first'] = team.opponents_first_downs
                team_features['opp_lost_fum'] = team.opponents_fumbles_lost
                team_features['opp_pass_com_per'] = team.opponents_pass_completion_percentage
                team_features['pass_att'] = team.pass_attempts
                # team_features['pass_com_per'] = team.pass_completion_percentage
                # team_features['pass_com'] = team.pass_completions
                # team_features['penalties'] = team.penalties
                team_features['plays'] = team.plays
                team_features['points_per_game'] = team.points_per_game #This will be my label
                # team_features['opp_pts_game'] = team.points_against_per_game
                # team_features['pass_yards'] = team.pass_yards
                team_features['rush_touch'] = team.rush_touchdowns
                team_features['pass_touch'] = team.pass_touchdowns
                team_features['turnovers'] = team.turnovers
                team_features['rush_first_downs'] = team.rush_first_downs
                team_features['rush_attempts'] = team.rush_attempts
                # team_features['rush_yards'] = team.rush_yards
                team_features['pass_first_downs'] = team.pass_first_downs
                save_team[counter] = team_features
                counter = counter + 1
        data_transformed_df = pd.DataFrame.from_dict(save_team).T
        return data_transformed_df
    
    def split(self, pandas_df):
        temp_df = pandas_df
        y = temp_df['points_per_game']
        temp_df = temp_df.drop(columns=['points_per_game','name'])
        # scaler = MinMaxScaler()
        # scaled_data = scaler.fit_transform(temp_df)
        # cols = temp_df.columns
        # x = pd.DataFrame(scaled_data, columns = cols)
        x = temp_df
        #Select features that matter
        # yy = y.astype('int')
        # select = SelectKBest(chi2, k=15)
        # X_new = select.fit_transform(x, yy)
        # filter = select.get_support()
        # features = x.columns
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, train_size=0.75)
        
        # return features[filter]

    # def split_2(self, pandas_df):
    #     temp_df = pandas_df
    #     y = temp_df['wins']
    #     temp_df = temp_df.drop(columns=['wins','name'])
    #     # scaler = MinMaxScaler()
    #     # scaled_data = scaler.fit_transform(temp_df)
    #     # cols = temp_df.columns
    #     # x = pd.DataFrame(scaled_data, columns = cols)
    #     x = temp_df
    #     self.x_train_2, self.x_test_2, self.y_train_2, self.y_test_2 = train_test_split(x,y, train_size=0.8)
        
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
        
        
        print('results of DT (score as labels): ',DT_err)
        print('results of GB (score as labels): ',GB_err)
        print('results of SV_R (score as labels): ',SV_R_err)
        print('results of RF (score as labels): ',RF_err)
        print('results of LR (score as labels): ',LR_err)
        print('results of KNN (score as labels): ',KNN_err)
        print('results of GPR (score as labels): ',GPR_err)
        print('results of MLP (score as labels): ',MLP_err)
        
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
            kernel = ['linear’', 'poly', 'rbf', 'sigmoid', 'precomputed']
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
        
    # def machine_2(self):
    #     #Cross validate all to find one that works the best
    #     GB = cross_validate(GradientBoostingRegressor(),
    #                                     self.x_train_2, self.y_train_2, cv=10,
    #                                     scoring=['neg_root_mean_squared_error'],return_train_score=True)
    #     RF = cross_validate(RandomForestRegressor(),
    #                                     self.x_train_2, self.y_train_2, cv=10,
    #                                     scoring=['neg_root_mean_squared_error'],return_train_score=True)
    #     LR = cross_validate(LinearRegression(),
    #                                     self.x_train_2, self.y_train_2, cv=10,
    #                                     scoring=['neg_root_mean_squared_error'],return_train_score=True)
    #     DT = cross_validate(DecisionTreeRegressor(),
    #                                     self.x_train_2, self.y_train_2, cv=10,
    #                                     scoring=['neg_root_mean_squared_error'],return_train_score=True)
    #     SV_R = cross_validate(SVR(kernel='linear'),self.x_train_2, self.y_train_2, cv=10,
    #                           scoring=['neg_root_mean_squared_error'],return_train_score=True)
    #     KNN = cross_validate(KNeighborsRegressor(),self.x_train_2, self.y_train_2, cv=10,
    #                           scoring=['neg_root_mean_squared_error'],return_train_score=True)
    #     GPR = cross_validate(GaussianProcessRegressor(),self.x_train_2, self.y_train_2, cv=10,
    #                           scoring=['neg_root_mean_squared_error'],return_train_score=True)
    #     MLP = cross_validate(MLPRegressor(),self.x_train_2, self.y_train_2, cv=10,
    #                           scoring=['neg_root_mean_squared_error'],return_train_score=True)
        
    #     DT_err = np.abs(np.mean(DT['test_neg_root_mean_squared_error']))
    #     GB_err = np.abs(np.mean(GB['test_neg_root_mean_squared_error']))
    #     SV_R_err = np.abs(np.mean(SV_R['test_neg_root_mean_squared_error']))
    #     RF_err = np.abs(np.mean(RF['test_neg_root_mean_squared_error']))
    #     LR_err = np.abs(np.mean(LR['test_neg_root_mean_squared_error']))
    #     KNN_err = np.abs(np.mean(KNN['test_neg_root_mean_squared_error']))
    #     GPR_err = np.abs(np.mean(GPR['test_neg_root_mean_squared_error']))
    #     MLP_err = np.abs(np.mean(MLP['test_neg_root_mean_squared_error']))
        
        
    #     print('results of DT (wins as labels): ',DT_err)
    #     print('results of GB (wins as labels): ',GB_err)
    #     print('results of SV_R (wins as labels): ',SV_R_err)
    #     print('results of RF (wins as labels): ',RF_err)
    #     print('results of LR (wins as labels): ',LR_err)
    #     print('results of KNN (wins as labels): ',KNN_err)
    #     print('results of GPR (wins as labels): ',GPR_err)
    #     print('results of MLP (wins as labels): ',MLP_err)
        
    #     err_list = [DT_err, GB_err, SV_R_err, RF_err, LR_err]
    #     lowest_error = err_list.index(min(err_list))
        
    #     #Return the best performing alg
    #     if lowest_error == 0:
    #         final_model = DecisionTreeRegressor().fit(self.x_test_2,self.y_test_2)
    #         print(r2_score(self.y_test_2, final_model.predict(self.x_test_2)))
    #         return final_model, 'DecisionTreeRegressor'
    #     elif lowest_error == 1:
    #         final_model = GradientBoostingRegressor().fit(self.x_test_2,self.y_test_2)
    #         print(r2_score(self.y_test_2, final_model.predict(self.x_test_2)))
    #         return final_model, 'GradientBoostingRegressor'
    #     if lowest_error == 2:
    #         final_model = SVR().fit(self.x_test,self.y_test_2)
    #         print(r2_score(self.y_test_2, final_model.predict(self.x_test_2)))
    #         return final_model, 'SVR'
    #     elif lowest_error == 3: 
    #         final_model = RandomForestRegressor().fit(self.x_test_2,self.y_test_2)
    #         print(r2_score(self.y_test_2, final_model.predict(self.x_test_2)))
    #         return final_model, 'RandomForestRegressor'
    #     if lowest_error == 4:
    #         final_model = LinearRegression().fit(self.x_test_2,self.y_test_2)
    #         print(r2_score(self.y_test_2, final_model.predict(self.x_test_2)))
    #         return final_model, 'LinearRegression'
    
    # def parameter_tuning_2(self, final_model, name):
        
    #     if name == 'DecisionTreeRegressor':
    #         n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    #         max_features = ['auto', 'sqrt', 'log2']
    #         max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
    #         max_depth.append(None)
    #         min_samples_split = [2, 5, 10]
    #         min_samples_leaf = [1, 2, 4]
    #         min_weight_fraction_leaf = [float(x) for x in np.linspace(start = 0, stop = 10, num = 5)]
    #         random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_weight_fraction_leaf':min_weight_fraction_leaf,
    #                'min_samples_leaf': min_samples_leaf}
    #         mod = DecisionTreeRegressor()
    #         test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
    #         test_model.fit(self.x_train_2,self.y_train_2) #Do I use self.x_test,self.y_test or self.x_train,self.y_train?
    #         print('print best params: ',test_model.best_params_)
    #         return test_model
    #     elif name == 'GradientBoostingRegressor':
    #         learning_rate = [float(x) for x in np.linspace(start = 0.001, stop = 1, num = 10)]
    #         n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    #         max_features = ['auto', 'sqrt']
    #         max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
    #         max_depth.append(None)
    #         min_samples_split = [2, 5, 10]
    #         min_samples_leaf = [1, 2, 4]
    #         random_grid = {'learning_rate': learning_rate,
    #                 'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf}
    #         mod = GradientBoostingRegressor()
    #         test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
    #         test_model.fit(self.x_train_2,self.y_train_2)
    #         print('print best params: ',test_model.best_params_)
    #         return test_model
    #     elif name == 'RandomForestRegressor':
    #         n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
    #         max_features = ['auto', 'sqrt']
    #         max_depth = [int(x) for x in np.linspace(1, 110, num = 10)]
    #         max_depth.append(None)
    #         min_samples_split = [2, 5, 10]
    #         min_samples_leaf = [1, 2, 4]
    #         bootstrap = [True, False]
    #         random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #         mod = RandomForestRegressor()
    #         test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
    #         test_model.fit(self.x_train_2,self.y_train_2)
    #         print('print best params: ',test_model.best_params_)
    #         return test_model
    #     elif name == 'SVR':
    #         tol = [float(x) for x in np.linspace(start = 0.001, stop = 1, num = 10)]
    #         kernel = ['linear’', 'poly', 'rbf', 'sigmoid', 'precomputed']
    #         gamma = ['scale', 'auto']
    #         C = [float(x) for x in np.linspace(1, 20, num = 15)]
    #         bootstrap = [True, False]
    #         random_grid = {'tol': tol,
    #                 'kernel': kernel,
    #                 'C': C,
    #                 'gamma': gamma
    #         }
    #         mod = SVR()
    #         test_model = RandomizedSearchCV(estimator = mod, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, n_jobs = -1)
    #         test_model.fit(self.x_train_2,self.y_train_2)
    #         print('print best params: ',test_model.best_params_)
    #         return test_model
    #     elif name == 'LinearRegression':
    #         print('Linear Regressor has no hyperparameters')
    #         mod = LinearRegression().fit(self.x_train_2,self.y_train_2)
    #         return mod
        
    def predict_points(self, pandas_df, final_model):
        
        # TO DO: add parameter tuning here for the final alg
        
        team1_loc = np.where(self.args.team1 == pandas_df['name'])
        team2_loc = np.where(self.args.team2 == pandas_df['name'])
        
        team_1_compare = pandas_df.iloc[team1_loc]
        team_2_compare = pandas_df.iloc[team2_loc]
        
        # print(team_1_compare)
        # print(team_2_compare)
        team_1_compare = team_1_compare.drop(columns=['points_per_game','name'])
        team_2_compare = team_2_compare.drop(columns=['points_per_game','name'])
        
        print(f"{self.args.team1} has a predicted score of {final_model.predict(team_1_compare)}")
        print(f"{self.args.team2} has a predicted score of {final_model.predict(team_2_compare)}")
        
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
        
        print(f"{self.args.team1} will score {model.predict(final_vector)} more points than {self.args.team2}")
        print(f"{self.args.team2} will score {model.predict(final_vector2)} more points than {self.args.team1}")
    
    def plot_feature_importance_classify(self, test_model, name):
        if name != "LinearRegression":
            feature_imp = pd.Series(test_model.best_estimator_.feature_importances_,index=self.x_test_class.columns).sort_values(ascending=False)
        else:
            importance = final_model.coef_
            feature_imp = pd.Series(np.abs(importance),index=self.x_test.columns).sort_values(ascending=False)
        sns.barplot(x=feature_imp,y=feature_imp.index)
        plt.show()
        
        
if __name__ == '__main__':
    start_time = time.time()
    
    tem_csv = Teams(2021)
    pd.DataFrame(tem_csv).to_csv('allteams.csv')
    
    bowl = bowl_games()
    bowl.input_arg()
    combine_list = {}
    # data_2008 = bowl.get_teams(2008)
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
    all_data = pd.concat([data_2009, data_2010, data_2011, data_2012, data_2013, data_2014,data_2015,data_2016,
                     data_2017, data_2018, data_2019,
                     data_2021],ignore_index=True)
    print(f"number of samples: {len(all_data)}")
    
    bowl.split(all_data)
    # bowl.split_2(all_data)
    
    final_model, name = bowl.machine()
    # final_model_2, name_2 = bowl.machine_2()
    
    mod = bowl.parameter_tuning(final_model, name)
    # mod_2 = bowl.parameter_tuning(final_model_2, name_2)
    
    bowl.predict_points(data_2021, mod)
    bowl.compare_two_teams(data_2021, mod)
    
    # bowl.plot_feature_importance_classify(mod, name)
    
    print("--- %s seconds ---" % (time.time() - start_time))
        