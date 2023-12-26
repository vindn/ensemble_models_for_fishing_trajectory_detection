#!/usr/bin/env python
# coding: utf-8
# %%
import dataframe_image as dfi
from bokeh.resources import INLINE
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import numpy as np

import geopandas as gpd
import movingpandas as mpd
import shapely as shp
import hvplot.pandas

import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
import time
from datetime import datetime, timedelta
import sys

# %%
random_state_trajs_fishing_info = 0
random_state_trajs_fishing = 0.1
min_duration_trajectory = 10  # in minutes

# train_acc = np.array([])

def load_dataset():
    # Dataset processing and filter
    input_csv = "dataset/dataset_fishing_train.csv"
    n = 0
    chunksize = 1000000
    first = True
    mmsi_dict = {}
    with pd.read_csv(input_csv, chunksize=chunksize, header=0) as reader:
        for pd_chunk in reader:

            try:
                # remove without mmsi
                pd_chunk.dropna(subset=['mmsi'], inplace=True)
                pd_chunk["mmsi"] = pd_chunk["mmsi"].astype(int)

                # set column type
                pd_chunk['timestamp'] = pd.to_datetime(pd_chunk['timestamp'])

                try:
                    pd_chunk['lat'] = pd.to_numeric(
                        pd_chunk['lat'], downcast='float', errors='coerce')
                except Exception as A:
                    print("Error chunk ", n,  " ", ": ", A)
                    pd_chunk.dropna()

                try:
                    pd_chunk['lon'] = pd.to_numeric(
                        pd_chunk['lon'], downcast='float', errors='coerce')
                except Exception as A:
                    print("Error chunk ", n,  " ", ": ", A)
                    pd_chunk.dropna()

                pd_chunk['shipcourse'] = pd_chunk['shipcourse'].astype(float)
                pd_chunk['shipspeed'] = pd_chunk['shipspeed'].astype(float)

                # delete lines nan
                pd_chunk.dropna()

                # append results
                if first:
                    df_gfw = pd_chunk
                    first = False
                else:
                    df_gfw = df_gfw.append(pd_chunk)

                # append results
                # append data frame to CSV file
                print("chunk: " + str(n))
                n += 1
            except Exception as A:
                print("Error chunk ", n,  " ", ": ", A)
                pass

    df_gfw['timestamp'] = pd.to_datetime(df_gfw['timestamp'], utc=True)
    return df_gfw


# GDF
def load_gdf(df_gfw):
    import geopandas as gpd
    import movingpandas as mpd
    import shapely as shp
    import hvplot.pandas

    from geopandas import GeoDataFrame, read_file
    from datetime import datetime, timedelta
    from holoviews import opts

    import warnings
    warnings.filterwarnings('ignore')

    opts.defaults(opts.Overlay(
        active_tools=['wheel_zoom'], frame_width=500, frame_height=400))

    gdf = gpd.GeoDataFrame(
        df_gfw.set_index('timestamp'), geometry=gpd.points_from_xy(df_gfw.lon, df_gfw.lat))

    gdf.set_crs('epsg:4326')

    return gdf

# Filter GDF to equal data


def filter_gdf(gdf, len_gdf_only_fishing, len_gdf_no_fishing):

    gdf_only_fishing = gdf[gdf['vesseltype'] ==
                           'Fishing'][:len_gdf_only_fishing]  # 263K
    gdf_no_fishing = gdf[gdf['vesseltype'] !=
                         'Fishing'][:len_gdf_no_fishing]  # 16M

    # gdf_only_fishing = gdf[ gdf['vesseltype'] == 'Fishing'][:2600000] #263K
    # gdf_no_fishing   = gdf[ gdf['vesseltype'] != 'Fishing'][:3000000] #16M

    gdf_filtered = pd.concat([gdf_only_fishing, gdf_no_fishing])

    return gdf_only_fishing, gdf_no_fishing, gdf_filtered


# Trajectories
def create_trajectory(gdf):
    import movingpandas as mpd
    import shapely as shp
    import hvplot.pandas
    import time

    # reset index
    gdf = gdf.reset_index()
    gdf['timestamp'] = pd.to_datetime(gdf['timestamp'], utc=True)

    # limit to avoid slow
#     gdf = gdf[:10000]

    # create trajectories

    start_time = time.time()

    # Specify minimum length for a trajectory (in meters)
    minimum_length = 0
    # collection = mpd.TrajectoryCollection(gdf, "imo",  t='timestamp', min_length=0.001)
    collection = mpd.TrajectoryCollection(
        gdf, "mmsi",  t='timestamp', min_length=0.001, crs='epsg:4326')
    collection.add_direction(gdf.shipcourse)
    collection.add_speed(gdf.shipspeed)

    # set time gap between trajectories for split
    collection = mpd.ObservationGapSplitter(
        collection).split(gap=timedelta(minutes=90))

    collection.add_speed(overwrite=True)
    collection.add_direction(overwrite=True)

    end_time = time.time()
    print("Time creation trajectories: ", (end_time-start_time)/60,  " min")

    return collection


# Trajectories Filter Situation - Only use for experiments
def mean_duration_fishing(trajs):
    mean = 0.0
    for traj in trajs.trajectories:
        mean += traj.get_duration().seconds

    return mean / len(trajs.trajectories)


# filters for trajectories
def filter_trajs(trajs_fishing, trajs_no_fishing):
    # criteria
    # 1 < speed < 50
    # min duration 10 min
    # number of points > 2

    new_traj_fishing = []
    new_traj_no_fishing = []
    mean_traj_duration_fishing = mean_duration_fishing(trajs_fishing)
    percent_mean = 0.1

    # fishing
    for traj in trajs_fishing.trajectories[:]:
        if traj.get_duration().seconds > 60*min_duration_trajectory and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50 and len(traj.df) > 2:
            new_traj_fishing.append(traj)

    # non fishing
    for traj in trajs_no_fishing.trajectories[:]:
        if traj.get_duration().seconds > 60*min_duration_trajectory and traj.df.speed.mean()*(100000*1.94384) > 1 and traj.df.speed.mean()*(100000*1.94384) < 50 and len(traj.df) > 2:
            new_traj_no_fishing.append(traj)

    print("fishing trajs: ", len(new_traj_fishing))
    print("non fishing trajs: ", len(new_traj_no_fishing))

    return mpd.TrajectoryCollection(new_traj_fishing),  mpd.TrajectoryCollection(new_traj_no_fishing)



# Select if load from file or build all trajectories from gdf
def load_trajectories( ):
    import pickle

    ####
    # select if load from file or build all trajectories from gdf
    ###
    object_data_dir = 'objects/'

    try:
        # Load trajectories collection
        with open(object_data_dir + 'trajs_fishing.mvpandas.2600000' , 'rb') as trajs_fishing_file:
            trajs_fishing = pickle.load(trajs_fishing_file)
        with open(object_data_dir + 'trajs_no_fishing.mvpandas.3000000', 'rb') as trajs_no_fishing_file:
            trajs_no_fishing = pickle.load(trajs_no_fishing_file)

        # with open(object_data_dir + 'trajs_fishing_gfw.mvpandas.13161', 'rb') as trajs_fishing_file:
        #     trajs_fishing = pickle.load(trajs_fishing_file)
        # with open(object_data_dir + 'trajs_no_fishing_gfw.mvpandas.8376', 'rb') as trajs_no_fishing_file:
        #     trajs_no_fishing = pickle.load(trajs_no_fishing_file)
        # with open(object_data_dir + 'trajs_unknow_gfw.mvpandas.21886', 'rb') as trajs_no_fishing_file:
        #     trajs_false_postives = pickle.load(trajs_no_fishing_file)
        # trajs_no_fishing = mpd.TrajectoryCollection( trajs_false_postives.trajectories + trajs_no_fishing.trajectories )

    except Exception as e:
        print(e, "Trajectories Collection File not Found!")

    # trajs_fishing, trajs_no_fishing = filter_trajs(trajs_fishing, trajs_no_fishing)
    print("Loaded ", len(trajs_fishing), " trajs fishing and ",
          len(trajs_no_fishing), " trajs non fishing.")

    return trajs_fishing, trajs_no_fishing

def circular_variance(directions):
    """
    Calcula a variância circular para uma série de dados angulares.

    Parâmetros:
    direcoes (pandas.Series): Série contendo os dados angulares (em graus).

    Retorna:
    float: Variância circular dos dados na série.
    """
    # Converter de graus para radianos
    direction_rad = np.radians(directions)

    # Calcular seno e cosseno
    sins = np.sin(direction_rad)
    cossins = np.cos(direction_rad)

    # Calcular médias de seno e cosseno
    mean_sin = np.mean(sins)
    mean_cossin = np.mean(cossins)

    # Calcular a variância circular
    r = np.sqrt(mean_sin**2 + mean_cossin**2)
    circular_variance = 1 - r

    return circular_variance

def circular_mean(directions):
    # Converter de graus para radianos
    direction_rad = np.radians(directions)

    # Calcular seno e cosseno médios
    mean_sin = np.mean(np.sin(direction_rad))
    mean_cos = np.mean(np.cos(direction_rad))

    # Calcular a média angular
    angular_mean = np.arctan2(mean_sin, mean_cos)

    # Converter de radianos para graus
    angular_mean_degrees = np.degrees(angular_mean)

    # Ajustar para que o resultado esteja entre 0 e 360 graus
    angular_mean_degrees = angular_mean_degrees % 360

    return angular_mean_degrees



# Traj Info
def init_trajectory_data(collection):
    import movingpandas as mpd
    import shapely as shp
    import hvplot.pandas
    from datetime import datetime, timedelta
    import time

    collection.add_speed(overwrite=True)
    collection.add_direction(overwrite=True)

    # format trajectories to clustering
    lines_traj_id = np.array([])
    mmsi = np.array([])
    area = np.array([])
    varCourse = np.array([])
    varSpeed = np.array([])
    duration = np.array([])
    medshipcourse = np.array([])
    # shipname = np.array([])
    meanShipSpeed = np.array([])
    meanSpeedKnot = np.array([])
    endTrajCoastDist = np.array([])
    # vesseltype = np.array([])
    traj_len = np.array([])
    n_points = np.array([])
    for traj in collection.trajectories:
        traj_id = str(traj.to_traj_gdf()["traj_id"]).split()[1]
#         mmsi        =  np.append( mmsi, traj.df['mmsi'][0].astype(int) )
        mmsi = np.append(mmsi, traj.df['mmsi'][0])
        lines_traj_id = np.append(lines_traj_id, traj_id)
        area = np.append(area, traj.get_mcp().area)
        circular_var = circular_variance( traj.df.shipcourse )
        varCourse = np.append(varCourse, circular_var)
        # varCourse = np.append(varCourse, traj.df.direction.var())
        medshipcourse = np.append(medshipcourse, traj.df.shipcourse.mean())
        varSpeed = np.append(varSpeed, traj.df.speed.var())
        duration = np.append(duration, traj.get_duration().seconds)
        # shipname = np.append(shipname, traj.df["shipname"][0])
        meanShipSpeed = np.append(meanShipSpeed, traj.df.speed.mean())
        meanSpeedKnot = np.append(
            meanSpeedKnot, traj.df.speed.mean()*(100000*1.94384))
        traj.df["speed_knot"] = traj.df.speed*(100000*1.94384)
#         endTrajCoastDist =    np.append( endTrajCoastDist, traj.get_end_location().distance(line_coast)*100 )
        # vesseltype = np.append(vesseltype, traj.df["vesseltype"][0])
        traj_len = np.append(traj_len, traj.get_length())
        n_points = np.append(n_points, len(traj.df))
        traj.df['ang_diff'] = angular_diff_for_rnn( traj.df['shipcourse'], traj.df['shipcourse'].shift(1))
        time_difference_seconds = (pd.Series( traj.df.index ).diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype(int))
        traj.df["time_diff"] = time_difference_seconds.values


    clus_df = pd.DataFrame()
    clus_df["traj_id"] = lines_traj_id
    clus_df["mmsi"] = mmsi
    clus_df["area"] = area
    clus_df["varCourse"] = varCourse
    clus_df["medshipcourse"] = medshipcourse
    clus_df["varSpeed"] = varSpeed
    clus_df["duration"] = duration
    # clus_df["shipname"] = shipname
    clus_df["meanShipSpeed"] = meanShipSpeed
    clus_df["meanSpeedKnot"] = meanSpeedKnot
#     clus_df["endTrajCoastDist"]  = endTrajCoastDist
    # clus_df["vesseltype"] = vesseltype
    clus_df["traj_len"] = traj_len
    clus_df["n_points"] = n_points

    return clus_df

import numpy as np

def angular_diff_for_rnn(direction1, direction2):
    """
    Calcula a menor diferença angular entre duas séries de direções.

    Parâmetros:
    direcao1 (pandas.Series ou array-like): Primeira série de direções (em graus).
    direcao2 (pandas.Series ou array-like): Segunda série de direções (em graus).

    Retorna:
    array-like: A menor diferença angular entre as direções.
    """
    # Converter de graus para radianos
    direction1_rad = np.radians(direction1)
    direction2_rad = np.radians(direction2)

    # Calcular a diferença angular em radianos
    difference = np.arctan2(np.sin(direction1_rad - direction2_rad), 
                            np.cos(direction1_rad - direction2_rad))

    # Converter de radianos para graus
    degrees_diff = np.degrees(difference)

    # Ajustar para que o resultado esteja entre -180 e 180 graus
    degrees_diff = (degrees_diff + 180) % 360 - 180

    return degrees_diff



# %% split dataset in 80% train and 20% test keeping proporcional number of classes;
# split dataset in 80% train and 20% test keeping proporcional number of classes;
def get_random_dataset( trajs_fishing, trajs_no_fishing, traj_info_fishing, traj_info_no_fishing ):
    import random
    import time
    from sklearn.preprocessing import StandardScaler

    random.seed(time.time())

    # number of fishing trajectories
    if len(trajs_fishing) < len(trajs_no_fishing):
        nft = len(trajs_fishing)
    else:
        nft = len(trajs_no_fishing)

    n_train_dataset = int(nft * 0.80)
    n_test_dataset = int(nft - n_train_dataset)

    list_index_fishing = list( range(nft) ) 
    list_index_nofishing = list( range(nft) )

    train_index_fishing   = random.sample(list_index_fishing, n_train_dataset)
    train_index_nofishing = random.sample(list_index_nofishing, n_train_dataset)

    test_index_fishing   = list(set(list_index_fishing) - set(train_index_fishing))
    test_index_nofishing = list(set(list_index_nofishing) - set(train_index_nofishing))

    x_train_trajs = []
    x_train_trajs_info = []
    y_train_trajs = []
    y_train_trajs_info = []
    x_test_trajs = []
    x_test_trajs_info = []
    y_test_trajs = []
    y_test_trajs_info = []
   
    scaler = StandardScaler()

    # X, Y train fishing
    for i in train_index_fishing[:n_train_dataset]:
        rnn_df = trajs_fishing.trajectories[i].df
        # rnn_df['ang_diff'] = angular_diff_for_rnn( rnn_df['shipcourse'], rnn_df['shipcourse'].shift(1))
        # rnn_df['ang_diff'].fillna(0, inplace=True)  # Preencher o primeiro valor NaN com 0   
        # rnn_df[['shipspeed', 'shipcourse']] = scaler.fit_transform(rnn_df[['shipspeed', 'shipcourse']])
        # x_train_trajs.append( rnn_df[['shipspeed', 'shipcourse']].to_numpy() )
        x_train_trajs.append( trajs_fishing.trajectories[i].df[['shipspeed', 'shipcourse', 'ang_diff', 'time_diff']].to_numpy() )
        
        x_train_trajs_info.append( traj_info_fishing.iloc[i][
            ['duration', 'varCourse', 'varSpeed', 'traj_len', 'n_points']
            ].to_numpy())
        y_train_trajs.append('fishing')
        y_train_trajs_info.append('fishing')

    # X, Y test fishing
    for i in test_index_fishing[:n_test_dataset]:
        rnn_df = trajs_fishing.trajectories[i].df
        # rnn_df['ang_diff'] = angular_diff_for_rnn( rnn_df['shipcourse'], rnn_df['shipcourse'].shift(1))
        # rnn_df['ang_diff'].fillna(0, inplace=True)  # Preencher o primeiro valor NaN com 0
        # rnn_df[['shipspeed', 'shipcourse']] = scaler.fit_transform(rnn_df[['shipspeed', 'shipcourse']])        
        # x_test_trajs.append( rnn_df[['shipspeed', 'shipcourse']].to_numpy() )
        x_test_trajs.append( trajs_fishing.trajectories[i].df[['shipspeed', 'shipcourse', 'ang_diff', 'time_diff']].to_numpy() )

        x_test_trajs_info.append( traj_info_fishing.iloc[i][
            ['duration', 'varCourse', 'varSpeed', 'traj_len', 'n_points']
            ].to_numpy())
        y_test_trajs.append('fishing')
        y_test_trajs_info.append('fishing')

    # X, Y train no fishing
    for i in train_index_nofishing[:n_train_dataset]:
        rnn_df = trajs_no_fishing.trajectories[i].df
        # rnn_df['ang_diff'] = angular_diff_for_rnn( rnn_df['shipcourse'], rnn_df['shipcourse'].shift(1))
        # rnn_df['ang_diff'].fillna(0, inplace=True)  # Preencher o primeiro valor NaN com 0
        # rnn_df[['shipspeed', 'shipcourse']] = scaler.fit_transform(rnn_df[['shipspeed', 'shipcourse']])
        # x_train_trajs.append( rnn_df[['shipspeed', 'shipcourse']].to_numpy() )
        x_train_trajs.append( trajs_no_fishing.trajectories[i].df[['shipspeed', 'shipcourse', 'ang_diff', 'time_diff']].to_numpy() )

        x_train_trajs_info.append( traj_info_no_fishing.iloc[i][
            ['duration', 'varCourse', 'varSpeed', 'traj_len', 'n_points']
            ].to_numpy())
        y_train_trajs.append('sailing')
        y_train_trajs_info.append('sailing')
   
    # X, Y test no fishing
    for i in test_index_nofishing[:n_test_dataset]:
        rnn_df = trajs_no_fishing.trajectories[i].df
        # rnn_df['ang_diff'] = angular_diff_for_rnn( rnn_df['shipcourse'], rnn_df['shipcourse'].shift(1))
        # rnn_df['ang_diff'].fillna(0, inplace=True)  # Preencher o primeiro valor NaN com 0
        # rnn_df[['shipspeed', 'shipcourse']] = scaler.fit_transform(rnn_df[['shipspeed', 'shipcourse']])
        # x_test_trajs.append( rnn_df[['shipspeed', 'shipcourse']].to_numpy() )
        x_test_trajs.append( trajs_no_fishing.trajectories[i].df[['shipspeed', 'shipcourse', 'ang_diff', 'time_diff']].to_numpy() )

        x_test_trajs_info.append( traj_info_no_fishing.iloc[i][
            ['duration', 'varCourse', 'varSpeed', 'traj_len', 'n_points']
            ].to_numpy())
        y_test_trajs.append('sailing')
        y_test_trajs_info.append('sailing')

    return np.array(x_train_trajs), np.array(x_train_trajs_info), np.array(y_train_trajs), np.array(y_train_trajs_info), np.array(x_test_trajs), np.array(x_test_trajs_info), np.array(y_test_trajs), np.array(y_test_trajs_info)



# %%

# MODELS

def logistic_regression_model(x, y):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    print("*** Logistic Regression")

    # Set test parameters
    valores_C = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
    regularizacao = ['l1', 'l2']
    valores_grid = {'C': valores_C, 'penalty': regularizacao}

    # Model
    modelLR = LogisticRegression(solver='liblinear', max_iter=1000)

    # Building GRIDS
    model = GridSearchCV(estimator=modelLR, param_grid=valores_grid, cv=5)
    model.fit(x, y)

    # Best accuracy and best parameters
    print("LR Best accuracy: ", model.best_score_)
    print("C parameter: ",     model.best_estimator_.C)
    print("Regularization: ",   model.best_estimator_.penalty)
    global train_acc
    train_acc = np.append(train_acc, ["LR", model.best_score_ ])

    return model

def decision_tree_model(x, y):
    from sklearn import decomposition, datasets
    from sklearn import tree
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("*** Decision Tree")

    std_slc = StandardScaler()
    pca = decomposition.PCA()
    dec_tree = tree.DecisionTreeClassifier()

    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])

    n_components = list(range(1, x.shape[1]+1, 1))
    criterion = ['gini', 'entropy']
    max_depth = [2, 4, 6, 8, 10, 12]

    parameters = dict(pca__n_components=n_components,
                      dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

    model = GridSearchCV(pipe, parameters, cv=5)
    r = model.fit(x, y)

    print('Best Criterion:', model.best_estimator_.get_params()
          ['dec_tree__criterion'])
    print('Best max_depth:', model.best_estimator_.get_params()
          ['dec_tree__max_depth'])
    print('Best Number Of Components:',
          model.best_estimator_.get_params()['pca__n_components'])
    print()
    print(model.best_estimator_.get_params()['dec_tree'])
    print("DT Best accuracy: ", model.best_score_)
    global train_acc
    train_acc = np.append(train_acc, ["DT", model.best_score_ ])

    return model

# Random Forest in Trajectory-base data
def random_forest_model(x, y):
    from sklearn import decomposition, datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("*** Random Forest")

    rf = RandomForestClassifier(max_depth=2, random_state=0)

    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]
    }

    model = GridSearchCV(rf, param_grid, cv=5, n_jobs=14)

    model.fit(x, y)

    print("RF Best accuracy: ", model.best_score_)
    print("RF Best parameters: ", model.best_params_)
    global train_acc
    train_acc = np.append(train_acc, ["RF", model.best_score_ ])

    return model

# NN in data Trajectory-base data
def nn_model(x, y, x_test, y_test, epochs):
    import gc
    from keras import backend as K
    # checkpoint
    checkpoint_nn_path = 'nn_val_checkpoint'

    print("*** Neural Network")

    # Dependencies
    import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import to_categorical
    from tensorflow.keras.layers import Dropout
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.preprocessing import LabelEncoder
    import seaborn as sn
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold, StratifiedKFold

    sc = StandardScaler()
    X = sc.fit_transform(x)

    lb = LabelEncoder()
    lb_trainy = lb.fit_transform(y)
    Y = to_categorical(lb_trainy)

    input_x = len(x[0])

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_nn_path,
                                  monitor='acc',
                                  mode='max',
                                  save_best_only=True,
                                  save_weights_only=True,
                                  verbose=1
                                  )

    # normalize x_test and y_test
    X_test = sc.fit_transform(x_test)
    Y_test = lb.fit_transform(y_test)
    Y_test = to_categorical(Y_test)

    best_accuracy = 0.0
    mean_time = np.array([])
    mean_accuracy = np.array([])
    n_rounds = 0.0

    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

    for train_index, val_index in skf.split(X, y):
        train_x = X[train_index]
        train_y = Y[train_index]
        val_x = X[val_index]
        val_y = Y[val_index]

        model = Sequential()
        model.add(Dense(32, input_dim=input_x, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop', metrics=['acc'])
#         print(model.summary())

        r = model.fit(train_x, train_y, epochs=epochs, batch_size=2048, validation_data=(
            val_x, val_y), callbacks=[cp_callback], verbose=2)

        # Loads the weights
        model.load_weights(checkpoint_nn_path)

        # prediction y_test
        start_time = time.time()
        p = model.predict(val_x)
        end_time = time.time()
        print(p)

        # for nn, for RL comment
        y_pred = np.array([np.argmax(i) for i in p])
        y_true = np.array([np.argmax(i) for i in val_y])

        accuracy = accuracy_score(y_true, y_pred)
        print("NN prediction accuracy: ", accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_weights('nn_best_model.vgg.h5')

        mean_accuracy = np.append(mean_accuracy, accuracy)
        mean_time = np.append(mean_time, end_time-start_time)
        n_rounds += 1

    print("NN BEST prediction accuracy: ", best_accuracy)
    print("NN MEAN prediction accuracy: ", mean_accuracy.mean())
    print("NN STD prediction accuracy: ", mean_accuracy.std())
    print("NN VAR prediction accuracy: ", mean_accuracy.var())
    print("NN MEAN execution time     : ", mean_time.mean())
    print("NN STD execution time     : ", mean_time.std())
    print("NN VAR execution time     : ", mean_time.var())
    print("NN prediction accuracies: ", mean_accuracy)
    global train_acc
    train_acc = np.append(train_acc, ["NN", best_accuracy ])

    # Loads the best weights
    model.load_weights('nn_best_model.vgg.h5')

    return model, X, Y, X_test, Y_test

# RNN in raw data
def rnn(x, y, test_x, test_y, epochs):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Embedding
    from tensorflow.keras.preprocessing import sequence
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.model_selection import KFold, StratifiedKFold
    import time
    import gc
    from keras import backend as K

    print("*** Recurent Neural Network")
    # truncate and pad input sequences
    max_trajectory_length = 500

    # x, y, test_x, test_y = prepare_data_rnn(trajs_fishing, trajs_no_fishing)

    test_X = sequence.pad_sequences(test_x, maxlen=max_trajectory_length)
    lb = LabelEncoder()
    lb_valy = lb.fit_transform(test_y)
    test_Y = to_categorical(lb_valy)

    best_accuracy = 0.0
    X_train = sequence.pad_sequences(x, maxlen=max_trajectory_length)
    # X_val = sequence.pad_sequences(val_x, maxlen=max_trajectory_length)

    # Set label in columns format
    # lb = LabelEncoder()
    lb_trainy = lb.fit_transform(y)
    Y_train = to_categorical(lb_trainy)
    # lb = LabelEncoder()
    # lb_testy = lb.fit_transform(val_y)
    # Y_val = to_categorical(lb_testy)

    ##
    # RNN Architecture
    ##
    model = Sequential()
    # model.add(Embedding(5000, embedding_vecor_length, input_length=(max_trajectory_length*4) ))
    # model.add(Embedding(5000, embedding_vecor_length, input_shape=(max_trajectory_length, 4) ))
    # model.add(Dropout(0.2))
    # model.add(LSTM(100, input_shape=(max_trajectory_length, 4)))
    model.add(LSTM(100, input_shape=(max_trajectory_length, 4)))
    # model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    # model.compile(loss='macro_crossentropy', optimizer='adam', metrics=['acc'])
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    ##
    # Fit the model
    ##
    checkpoint_path = '.'
    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                    monitor='val_acc',
                                    mode='max',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    verbose=1
                                    )

    r = model.fit(X_train, Y_train, epochs=epochs, batch_size=2048, callbacks=[cp_callback])


    # Avaliar o Modelo
    loss, accuracy = model.evaluate(X_train, Y_train)
    print(f'RNN Model Accuracy: {accuracy * 100:.2f}%')
    global train_acc
    train_acc = np.append(train_acc, ["RNN", accuracy ])

    return model, X_train, Y_train, test_X, test_Y

#TODO implement SVM
def svm(x, y):
    from sklearn import decomposition, datasets
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    print("*** SVM")

    # std_slc = StandardScaler()
    # pca = decomposition.PCA()
    ## Substituir para nao demorar nos testes!
    model = RandomForestClassifier(max_depth=2, random_state=0)
    # model = SVC(C=0.1, kernel='linear', probability=True)

#     param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    # A GridSearchCV round with search parameters was made, and this parameter is optimal.
    # SVM is too slow, then I will use the optimal parameters, if you wish, use the commented line above.
    # param_grid = {'C': [1], 'kernel': ['linear'], 'gamma': [1]}

    # model = GridSearchCV(svm, param_grid, cv=5, n_jobs=14)
    model.fit(x, y)

    print("SVM Best accuracy: ", model.score(x,y))
    global train_acc
    train_acc = np.append(train_acc, ["SVM", model.best_score_ ])

    return model

def gradientBoosting( x, y ):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score

    print("*** GB")

    # Definir o modelo
    gb = GradientBoostingClassifier(random_state=42)

    # GB best parameters:  {
    # 'learning_rate': 0.1, 
    # 'max_depth': 3, 
    # 'min_samples_leaf': 1, 
    # 'min_samples_split': 2, 
    # 'n_estimators': 300, 
    # 'subsample': 0.9}

    # Definir os parâmetros para a pesquisa em grade
    # param_grid = {
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 4, 5],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'min_samples_split': [2, 4, 6],
    #     'min_samples_leaf': [1, 2, 3]
    # }
    # # Definir a pesquisa em grade com validação cruzada
    # grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # # Executar a pesquisa em grade
    # grid_search.fit(x, y)
    # # Melhor conjunto de parâmetros
    # best_parameters = grid_search.best_params_
    # # Melhor modelo a partir da pesquisa em grade
    # best_model = grid_search.best_estimator_

    best_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,max_depth=3, random_state=42, min_samples_leaf=1, min_samples_split=2, subsample=0.9)
    best_model.fit(x, y)

    acc = best_model.score(x,y)
    print("GB Accuracy: ", acc)
    # print("GB best parameters: ", best_parameters)
    global train_acc
    train_acc = np.append(train_acc, ["GB", acc ])

    return best_model




# %%
# def ensemble_voting_predict( X, X_nn, X_rnn, lr_model, dt_model, rf_model, nn_simple_model, rnn_model , svm_model, gb_model):
#     # Obter as predições de classe de cada modelo
#     lr_probs = lr_model.predict_proba(X)[:, 1]
#     dt_probs = dt_model.predict_proba(X)[:, 1]
#     rf_probs = rf_model.predict_proba(X)[:, 1]
#     nn_probs = nn_simple_model.predict(X_nn)[:, 1]
#     rnn_probs = rnn_model.predict(X_rnn)[:, 1]
#     # svm_probs = svm_model.predict_proba(X)[:, 1]
#     gb_probs = gb_model.predict_proba(X)[:, 1]

#     # Calcular as predições do ensemble por votação majoritária
#     # ensemble_preds = (lr_probs + dt_probs + rf_probs + nen_probs) >= 2.0
#     ensemble_preds = (lr_probs + dt_probs + rf_probs + nn_probs + rnn_probs + gb_probs) >= 3.0
    
#     return ensemble_preds.astype(int)  # Converte para inteiros 0 ou 1

# def ensemble_mean_predict(X, X_nn, X_rnn, lr_model, dt_model, rf_model, nn_simple_model, rnn_model, svm_model, gb_model):
#     # Obter as probabilidades previstas por cada modelo
#     lr_probs = lr_model.predict_proba(X)[:, 1]
#     dt_probs = dt_model.predict_proba(X)[:, 1]
#     rf_probs = rf_model.predict_proba(X)[:, 1]
#     nn_probs = nn_simple_model.predict(X_nn)[:, 1]
#     rnn_probs = rnn_model.predict(X_rnn)[:, 1]
#     # svm_probs = svm_model.predict_proba(X)[:, 1]
#     gb_probs = gb_model.predict_proba(X)[:, 1]

#     # Calcular a média das probabilidades
#     avg_probs = (lr_probs + dt_probs + rf_probs + nn_probs + rnn_probs + gb_probs) / 6
    
#     # Converter as médias de probabilidades em predições de classe
#     predictions = np.round(avg_probs)
    
#     return predictions

# # def ensemble_weighted_voting(X, X_nn, X_rnn, weights, lr_model, dt_model, rf_model, nn_simple_model, rnn_model, svm_model, gb_model ):
# #     lr_preds = lr_model.predict_proba(X)[:, 1] * weights[0]
# #     dt_preds = dt_model.predict_proba(X)[:, 1] * weights[1]
# #     rf_probs = rf_model.predict_proba(X)[:, 1] * weights[2]
# #     nn_preds = nn_simple_model.predict(X_nn)[:, 1] * weights[3]
# #     rnn_preds = rnn_model.predict(X_rnn)[:, 1] * weights[4]
# #     svm_probs = svm_model.predict_proba(X)[:, 1] * weights[5]
# #     gb_probs = gb_model.predict_proba(X)[:, 1] * weights[6]
# #     # nen_preds = nen_model.predict(nn_x_test)[:, 1] * weights[4]
    
# #     # ensemble_preds = (lr_preds + dt_preds + rf_probs + nen_preds + svm_probs ) >= (sum(weights) / 2.5 )
# #     # ensemble_preds = (lr_preds + dt_preds + rf_probs + nn_preds + rnn_preds) >= (sum(weights) / 2.5 )
# #     ensemble_preds = (lr_preds + dt_preds + rf_probs + nn_preds + rnn_preds + svm_probs + gb_probs) >= (sum(weights) / 3.5 )
    
# #     return ensemble_preds.astype(int)
# def ensemble_weighted_voting(X, X_nn, X_rnn, weights, lr_model, dt_model, rf_model, nn_simple_model, rnn_model, svm_model, gb_model ):
#     lr_preds = lr_model.predict_proba(X)[:, 1] * 1
#     dt_preds = dt_model.predict_proba(X)[:, 1] * 1
#     rf_probs = rf_model.predict_proba(X)[:, 1] * 2
#     nn_preds = nn_simple_model.predict(X_nn)[:, 1] * 1
#     rnn_preds = rnn_model.predict(X_rnn)[:, 1] * 4
#     gb_probs = gb_model.predict_proba(X)[:, 1] * 1


#     ensemble_preds = (lr_preds + dt_preds + rf_probs + nn_preds + rnn_preds + gb_probs) >= 5
    
#     return ensemble_preds.astype(int)



def my_precision_recall_fscore_support( y_true, y_pred ):

    print("y_true len: ", len(y_true))

    prfs = np.array([])

    # for class fishing
    tn, fp, fn, tp = 0,0,0,0
    support=0
    for yt, yp in zip(y_true, y_pred):
        if yt == "fishing":
            support +=1
        if yt == "fishing" and yt == yp:
            tp += 1
        elif yt == "sailing" and yt == yp:
            tn += 1
        elif yt == "fishing" and yt != yp:
            fp += 1
        elif yt == "sailing" and yt != yp:
            fn += 1

    precision = round( tp / (tp+fp), 2 )
    recall    = round( tp / (tp+fn), 2 )
    f1        = round( 2 * ((precision*recall)/(precision+recall)), 2 )
    
    prfs = np.append( prfs, np.array([
        "{:.2f}".format( precision ), 
        "{:.2f}".format( recall ), 
        "{:.2f}".format( f1 ), 
        "{:.0f}".format( support )
        ]) )

    # for class sailing
    tn, fp, fn, tp = 0,0,0,0
    support=0
    for yt, yp in zip(y_true, y_pred):
        if yt == "sailing":
            support +=1
        if yt == "sailing" and yt == yp:
            tp += 1
        elif yt == "fishing" and yt == yp:
            tn += 1
        elif yt == "sailing" and yt != yp:
            fp += 1
        elif yt == "fishing" and yt != yp:
            fn += 1
    
    precision = round( tp / (tp+fp), 2 )
    recall    = round( tp / (tp+fn), 2 )
    f1        = round( 2 * ((precision*recall)/(precision+recall)), 2 )
    
    prfs = np.append( prfs, np.array([
        "{:.2f}".format( precision ), 
        "{:.2f}".format( recall ), 
        "{:.2f}".format( f1 ), 
        "{:.0f}".format( support )
        ]) )

    return prfs.reshape(2, 4)

####################
# Call Models
####################

def disable_gpu( ):
    import tensorflow as tf
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("Invalid device or cannot modify virtual devices once initialized.")
        pass

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#######################################
# MAIN
# Dataset
# || train (80%) || test (20%) ||
#######################################

# disable_gpu( )

# load dataset from file
# df_gfw = load_dataset()
# transform data frame in geo data frame
# gdf = load_gdf(df_gfw)
# %%
# transform gdf in moving pandas trajectories. Or load from file the prior built.
trajs_fishing, trajs_no_fishing = load_trajectories( )

trajs_fishing, trajs_no_fishing = filter_trajs( trajs_fishing, trajs_no_fishing )

# %%

# we have 12K fishing trajectories and 108K non fishing trajectories
# limit trajs non fishing to avoid waste unnecessary processing; <-------------------
trajs_no_fishing = mpd.TrajectoryCollection(
    trajs_no_fishing.trajectories[:30000])

# trajs info fishing (trajectory-based data)
traj_info_fishing = init_trajectory_data(trajs_fishing)
n_traj_info_fishing = len(traj_info_fishing)
print("n_traj_info_fishing: ", n_traj_info_fishing)
# trajs info no fishing (trajectory-based data)
traj_info_no_fishing = init_trajectory_data(trajs_no_fishing)
n_traj_info_nofishing = len(traj_info_no_fishing)
print("n_traj_info_nofishing", n_traj_info_nofishing)

# %%


# %%
# CALL ENSEMBLE METHODS IN DATASET
# instantiate models
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

epochs = 200
rounds = 50 
acc_all = np.array([])
scores_all = np.array([])
acc_stacking_best = np.array([])
scores_stacking_best = np.array([])
acc_stacking_tree = np.array([])
scores_stacking_tree = np.array([])
probs_all = np.array([])
train_acc = np.array([])

for r in range(rounds):
    # get random in dataset
    x_train_trajs, x_train_trajs_info, y_train_trajs, y_train_trajs_info, x_test_trajs, x_test_trajs_info, y_test_trajs, y_test_trajs_info = get_random_dataset( trajs_fishing, trajs_no_fishing, traj_info_fishing, traj_info_no_fishing )
    n_rows_test = len(x_test_trajs_info)

    # Train models   
    lr_model = logistic_regression_model(x_train_trajs_info, y_train_trajs_info)
    dt_model = decision_tree_model(x_train_trajs_info, y_train_trajs_info)
    rf_model = random_forest_model(x_train_trajs_info, y_train_trajs_info)
    nn_simple_model, nn_x_train, nn_y_train, nn_x_test, nn_y_test = nn_model(x_train_trajs_info, y_train_trajs_info, x_test_trajs_info, y_test_trajs_info, epochs)
    rnn_model, rnn_train_x, rnn_train_y, rnn_test_x, rnn_test_y = rnn( x_train_trajs, y_train_trajs, x_test_trajs, y_test_trajs, epochs )
    # svm_model = svm(x_train_trajs_info, y_train_trajs_info)
    gb_model = gradientBoosting(x_train_trajs_info, y_train_trajs_info)

    # lr
    lr_probs = lr_model.predict(x_test_trajs_info)
    lr_proba = lr_model.predict_proba(x_test_trajs_info)
    accuracy_lr = accuracy_score(y_test_trajs_info, lr_probs)
    lr_score = my_precision_recall_fscore_support( y_test_trajs_info, lr_probs )
    acc_all    = np.append( acc_all, [r, 'LR', accuracy_lr])
    scores_all = np.append( scores_all, [r, 'LR', lr_score])
    probs = np.column_stack((lr_proba, y_test_trajs_info))
    # probs = np.column_stack((probs, np.array(['RNN']*n_rows_test)))
    probs = np.column_stack((probs, np.array(['LR']*n_rows_test)))
    probs_all = probs
    
    # dt
    dt_probs = dt_model.predict(x_test_trajs_info)
    dt_proba = dt_model.predict_proba(x_test_trajs_info)
    accuracy_dt = accuracy_score(y_test_trajs_info, dt_probs)
    dt_score = my_precision_recall_fscore_support( y_test_trajs_info, dt_probs )
    acc_all    = np.append( acc_all, [r, 'DT', accuracy_dt])
    scores_all = np.append( scores_all, [r, 'DT', dt_score])
    probs = np.column_stack((dt_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['DT']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )


    # rf
    rf_probs = rf_model.predict(x_test_trajs_info)
    rf_proba = rf_model.predict_proba(x_test_trajs_info)
    accuracy_rf = accuracy_score(y_test_trajs_info, rf_probs)
    rf_score = my_precision_recall_fscore_support( y_test_trajs_info, rf_probs )
    acc_all    = np.append( acc_all, [r, 'RF', accuracy_rf])
    scores_all = np.append( scores_all, [r, 'RF', rf_score])
    probs = np.column_stack((rf_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['RF']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )


    # NN
    nn_proba = nn_simple_model.predict(nn_x_test)
    nn_probs = nn_proba[:, 1]
    nn_probs = np.round(nn_probs)
    nn_probs = np.array( [  'fishing' if i == 0 else 'sailing' for i in nn_probs  ] )
    accuracy_nn = accuracy_score(y_test_trajs_info, nn_probs)
    nn_score = my_precision_recall_fscore_support( y_test_trajs_info, nn_probs )    
    acc_all    = np.append( acc_all, [r, 'NN', accuracy_nn])
    scores_all = np.append( scores_all, [r, 'NN', nn_score])
    probs = np.column_stack((nn_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['NN']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # RNN
    rnn_proba = rnn_model.predict(rnn_test_x)
    rnn_p = rnn_proba[:, 1]
    rnn_p = np.round( rnn_p )
    rnn_p = np.array( [  'fishing' if i == 0 else 'sailing' for i in rnn_p  ] )
    accuracy_rnn = accuracy_score(y_test_trajs, rnn_p)
    rnn_score = my_precision_recall_fscore_support( y_test_trajs, rnn_p )    
    acc_all    = np.append( acc_all, [r, 'RNN', accuracy_rnn])
    scores_all = np.append( scores_all, [r, 'RNN', rnn_score])
    probs = np.column_stack((rnn_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['RNN']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )


    # svm
    # svm_probs = svm_model.predict_proba(x_test_trajs_info)
    # svm_probs = np.round(svm_probs[:, 1])
    # svm_probs = np.array( [  'fishing' if i == 0 else 'sailing' for i in svm_probs  ] )
    # accuracy_svm = accuracy_score(y_test_trajs_info, svm_probs)
    # svm_score = my_precision_recall_fscore_support( y_test_trajs_info, svm_probs )
    # print(accuracy_svm)

    # GB
    gb_proba = gb_model.predict_proba(x_test_trajs_info)
    gb_probs = np.round(gb_proba[:, 1])
    gb_probs = np.array( [  'fishing' if i == 0 else 'sailing' for i in gb_probs  ] )
    accuracy_gb = accuracy_score(y_test_trajs_info, gb_probs)
    gb_score = my_precision_recall_fscore_support( y_test_trajs_info, gb_probs )
    acc_all    = np.append( acc_all, [r, 'GB', accuracy_gb])
    scores_all = np.append( scores_all, [r, 'GB', gb_score])
    probs = np.column_stack((gb_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['GB']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE MEAN (RF, RNN, GB)
    # lr_1 = lr_proba[:, 1] 
    # dt_1 = dt_proba[:, 1] 
    rf_1 = rf_proba[:, 1] 
    # nn_1 = nn_proba[:, 1] 
    rnn_1 = rnn_proba[:, 1] 
    gb_1 = gb_proba[:, 1] 

    ensemble_proba = (rf_proba + rnn_proba + gb_proba ) / 3
    ensemble_preds = (rf_1 + rnn_1 + gb_1) >= 1.5
    ensemble_preds = ensemble_preds.astype(int)
    ensemble_mean_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    ensemble_all_mean_accuracy = accuracy_score(y_test_trajs_info, ensemble_mean_predictions_labels)
    ensemble_all_mean_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_mean_predictions_labels )
    acc_all    = np.append( acc_all, [r, 'Ensemble Mean (RF, RNN, GB)', ensemble_all_mean_accuracy])
    scores_all = np.append( scores_all, [r, 'Ensemble Mean All (RF, RNN, GB)', ensemble_all_mean_score])
    probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Ensemble Mean All (RF, RNN, GB)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE MEAN (RNN, GB)
    ensemble_proba = (rnn_proba + gb_proba ) / 2
    ensemble_preds = (rnn_1 + gb_1) >= 1
    ensemble_preds = ensemble_preds.astype(int)
    ensemble_mean_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    ensemble_all_mean_accuracy = accuracy_score(y_test_trajs_info, ensemble_mean_predictions_labels)
    ensemble_all_mean_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_mean_predictions_labels )
    acc_all    = np.append( acc_all, [r, 'Ensemble Mean (RNN, GB)', ensemble_all_mean_accuracy])
    scores_all = np.append( scores_all, [r, 'Ensemble Mean (RNN, GB)', ensemble_all_mean_score])
    probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Ensemble Mean (RNN, GB)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE MEAN (RF, RNN)
    ensemble_proba = (rf_proba + rnn_proba ) / 2
    ensemble_preds = (rf_1 + rnn_1 ) >= 1
    ensemble_preds = ensemble_preds.astype(int)
    ensemble_mean_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    ensemble_all_mean_accuracy = accuracy_score(y_test_trajs_info, ensemble_mean_predictions_labels)
    ensemble_all_mean_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_mean_predictions_labels )
    acc_all    = np.append( acc_all, [r, 'Ensemble Mean (RF, RNN)', ensemble_all_mean_accuracy])
    scores_all = np.append( scores_all, [r, 'Ensemble Mean (RF, RNN)', ensemble_all_mean_score])
    probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Ensemble Mean (RF, RNN)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE WEIGHTED VOTING ALL
    # original (1,1,2,1,4,1) - 0.95
    # (1,1,3,1,2,1) - 0.9425
    # (1,1,3,2,3,2) - 93.65
    # (1,1,3,1,3,2) - 95.90
    # (1,1,3,1,2,2) - 
    # ensemble_proba = (lr_proba*1 + dt_proba*1 + rf_proba*3 + nn_proba*1 + rnn_proba*2 + gb_proba*2 ) / 10
    # ensemble_preds = (lr_1*1 + dt_1*1 + rf_1*3 + nn_1*1 + rnn_1*2 + gb_1*2) >= 5
    # ensemble_preds = ensemble_preds.astype(int)
    # ensemble_weighted_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    # ensemble_all_weighted_accuracy = accuracy_score(y_test_trajs_info, ensemble_weighted_predictions_labels)
    # ensemble_all_weighted_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_weighted_predictions_labels )
    # acc_all    = np.append( acc_all, [r, 'Ensemble Weighted All', ensemble_all_weighted_accuracy])
    # scores_all = np.append( scores_all, [r, 'Ensemble Weighted All', ensemble_all_weighted_score])
    # probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    # probs = np.column_stack((probs, np.array(['Ensemble Weighted All']*n_rows_test)))   
    # probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE WEIGHTED (2RF, 1RNN)
    ensemble_proba = (rf_proba*2 + rnn_proba*1 ) / 3
    ensemble_preds = (rf_1*2 + rnn_1*1 ) >= 1.5
    ensemble_preds = ensemble_preds.astype(int)
    ensemble_weighted_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    ensemble_all_weighted_accuracy = accuracy_score(y_test_trajs_info, ensemble_weighted_predictions_labels)
    ensemble_all_weighted_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_weighted_predictions_labels )
    acc_all    = np.append( acc_all, [r, 'Ensemble Weighted (2RF, 1RNN)', ensemble_all_weighted_accuracy])
    scores_all = np.append( scores_all, [r, 'Ensemble Weighted (2RF, 1RNN)', ensemble_all_weighted_score])
    probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Ensemble Weighted (2RF, 1RNN)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE WEIGHTED (2.RNN, 1.GB)
    ensemble_proba = (rnn_proba*2 + gb_proba*1 ) / 3
    ensemble_preds = (rnn_1*2 + gb_1*1) >= 1.5
    ensemble_preds = ensemble_preds.astype(int)
    ensemble_weighted_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    ensemble_all_weighted_accuracy = accuracy_score(y_test_trajs_info, ensemble_weighted_predictions_labels)
    ensemble_all_weighted_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_weighted_predictions_labels )
    acc_all    = np.append( acc_all, [r, 'Ensemble Weighted (2RNN, 1GB)', ensemble_all_weighted_accuracy])
    scores_all = np.append( scores_all, [r, 'Ensemble Weighted (2RNN, 1GB)', ensemble_all_weighted_score])
    probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Ensemble Weighted (2RNN, 1GB)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE WEIGHTED (1.RNN, 2.GB)
    ensemble_proba = (rnn_proba*1 + gb_proba*2 ) / 3
    ensemble_preds = (rnn_1*1 + gb_1*2) >= 1.5
    ensemble_preds = ensemble_preds.astype(int)
    ensemble_weighted_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    ensemble_all_weighted_accuracy = accuracy_score(y_test_trajs_info, ensemble_weighted_predictions_labels)
    ensemble_all_weighted_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_weighted_predictions_labels )
    acc_all    = np.append( acc_all, [r, 'Ensemble Weighted (1RNN, 2GB)', ensemble_all_weighted_accuracy])
    scores_all = np.append( scores_all, [r, 'Ensemble Weighted (1RNN, 2GB)', ensemble_all_weighted_score])
    probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Ensemble Weighted (1RNN, 2GB)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # ENSEMBLE WEIGHTED (2.RF, 1.RNN, 1.GB)
    # (3,2,1) - 0.9460
    # (2,2,1) - 0.9400
    # (2,1,1) - 
    ensemble_proba = (rf_proba*2 + rnn_proba*1 + gb_proba*1 ) / 4
    ensemble_preds = (rf_1*2 + rnn_1*1 + gb_1*1) >= 2
    ensemble_preds = ensemble_preds.astype(int)
    ensemble_weighted_predictions_labels = ['fishing' if pred == 0 else 'sailing' for pred in ensemble_preds]
    ensemble_all_weighted_accuracy = accuracy_score(y_test_trajs_info, ensemble_weighted_predictions_labels)
    ensemble_all_weighted_score = my_precision_recall_fscore_support( y_test_trajs_info, ensemble_weighted_predictions_labels )
    acc_all    = np.append( acc_all, [r, 'Ensemble Weighted (2RF, 1RNN, 1GB)', ensemble_all_weighted_accuracy])
    scores_all = np.append( scores_all, [r, 'Ensemble Weighted (2RF, 1RNN, 1GB)', ensemble_all_weighted_score])
    probs = np.column_stack((ensemble_proba, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Ensemble Weighted (2RF, 1RNN, 1GB)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )


    # STACKING (RF, RNN, GB)
    lr_train_probs  = lr_model.predict_proba(x_train_trajs_info)
    dt_train_probs  = dt_model.predict_proba(x_train_trajs_info)
    rf_train_probs  = rf_model.predict_proba(x_train_trajs_info)
    nn_train_probs  = nn_simple_model.predict(nn_x_train)
    rnn_train_probs = rnn_model.predict(rnn_train_x)
    gb_train_probs  = gb_model.predict_proba(x_train_trajs_info)

    # Train DF
    df_stacking_train = pd.DataFrame({
        # 'lr_0': lr_train_probs[:, 0],
        # 'lr_1': lr_train_probs[:, 1],
        # 'dt_0': dt_train_probs[:, 0],
        # 'dt_1': dt_train_probs[:, 1],
        'rf_0': rf_train_probs[:, 0],
        'rf_1': rf_train_probs[:, 1],
        # 'nn_0': nn_train_probs[:, 0],
        # 'nn_1': nn_train_probs[:, 1],
        'rnn_0': rnn_train_probs[:, 0],
        'rnn_1': rnn_train_probs[:, 1],
        'gb_0': gb_train_probs[:, 0],
        'gb_1': gb_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        # 'lr_0': lr_proba[:, 0],
        # 'lr_1': lr_proba[:, 1],
        # 'dt_0': dt_proba[:, 0],
        # 'dt_1': dt_proba[:, 1],
        'rf_0': rf_proba[:, 0],
        'rf_1': rf_proba[:, 1],
        # 'nn_0': nn_proba[:, 0],
        # 'nn_1': nn_proba[:, 1],
        'rnn_0': rnn_proba[:, 0],
        'rnn_1': rnn_proba[:, 1],
        'gb_0': gb_proba[:, 0],
        'gb_1': gb_proba[:, 1],
    })

    meta_model = random_forest_model( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_probra = meta_model.predict_proba( df_stacking_test.to_numpy() )
    meta_predictions = meta_probra[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking (RF, RNN, GB)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking (RF, RNN, GB)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking (RF, RNN, GB)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # STACKING (GB, RNN)
    # Train DF
    df_stacking_train = pd.DataFrame({
        'rnn_0': rnn_train_probs[:, 0],
        'rnn_1': rnn_train_probs[:, 1],
        'gb_0': gb_train_probs[:, 0],
        'gb_1': gb_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        'rnn_0': rnn_proba[:, 0],
        'rnn_1': rnn_proba[:, 1],
        'gb_0': gb_proba[:, 0],
        'gb_1': gb_proba[:, 1],
    })

    meta_model = random_forest_model( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_probra = meta_model.predict_proba( df_stacking_test.to_numpy() )
    meta_predictions = meta_probra[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking (GB, RNN)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking (GB, RNN)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking (GB, RNN)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # STACKING (GB, RF)
    # Train DF
    df_stacking_train = pd.DataFrame({
        'rf_0': rf_train_probs[:, 0],
        'rf_1': rf_train_probs[:, 1],
        'gb_0': gb_train_probs[:, 0],
        'gb_1': gb_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        'rf_0': rf_proba[:, 0],
        'rf_1': rf_proba[:, 1],
        'gb_0': gb_proba[:, 0],
        'gb_1': gb_proba[:, 1],
    })

    meta_model = random_forest_model( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_probra = meta_model.predict_proba( df_stacking_test.to_numpy() )
    meta_predictions = meta_probra[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking (GB, RF)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking (GB, RF)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking (GB, RF)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # STACKING (RNN, RF)
    # Train DF
    df_stacking_train = pd.DataFrame({
        'rf_0': rf_train_probs[:, 0],
        'rf_1': rf_train_probs[:, 1],
        'rnn_0': rnn_train_probs[:, 0],
        'rnn_1': rnn_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        'rf_0': rf_proba[:, 0],
        'rf_1': rf_proba[:, 1],
        'rnn_0': rnn_proba[:, 0],
        'rnn_1': rnn_proba[:, 1],
    })

    meta_model = random_forest_model( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_probra = meta_model.predict_proba( df_stacking_test.to_numpy() )
    meta_predictions = meta_probra[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking (RNN, RF)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking (RNN, RF)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking (RNN, RF)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    # STACKING (STACKING (GB, NN) RNN)
    
    # STACKING (GB, NN)
    # Train DF
    df_stacking_train = pd.DataFrame({
        'nn_0': nn_train_probs[:, 0],
        'nn_1': nn_train_probs[:, 1],
        'gb_0': gb_train_probs[:, 0],
        'gb_1': gb_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        'nn_0': nn_proba[:, 0],
        'nn_1':nn_proba[:, 1],
        'gb_0': gb_proba[:, 0],
        'gb_1': gb_proba[:, 1],
    })

 
    meta_model1 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model1.fit( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_train_probs = meta_model1.predict_proba( df_stacking_train.to_numpy() )
    meta_test_probs = meta_model1.predict_proba( df_stacking_test.to_numpy() )
    
    # STACKING (STACKING (GB, NN) RNN)
    # Train DF
    df_stacking_meta_train = pd.DataFrame({
        'rnn_0': rnn_train_probs[:, 0],
        'rnn_1': rnn_train_probs[:, 1],
        'meta_0': meta_train_probs[:, 0],
        'meta_1': meta_train_probs[:, 1],
    })

    # Test DF
    df_stacking_meta_test = pd.DataFrame({
        'rnn_0': rnn_proba[:, 0],
        'rnn_1': rnn_proba[:, 1],
        'meta_0': meta_test_probs[:, 0],
        'meta_1': meta_test_probs[:, 1],
    })

    meta_model2 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model2.fit( df_stacking_meta_train.to_numpy(), y_train_trajs_info )
    meta_probra2 = meta_model2.predict_proba( df_stacking_meta_test.to_numpy() )
   
    meta_predictions = meta_probra2[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking( Stacking (GB, NN), RNN)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking( Stacking (GB, NN), RNN)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking( Stacking (GB, NN), RNN)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    ##
    # STACKING (STACKING (RF, NN) RNN)
    
    # STACKING (RF, NN)
    # Train DF
    df_stacking_train = pd.DataFrame({
        'nn_0': nn_train_probs[:, 0],
        'nn_1': nn_train_probs[:, 1],
        'rf_0': rf_train_probs[:, 0],
        'rf_1': rf_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        'nn_0': nn_proba[:, 0],
        'nn_1':nn_proba[:, 1],
        'rf_0': rf_proba[:, 0],
        'rf_1': rf_proba[:, 1],
    })

 
    meta_model1 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model1.fit( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_train_probs = meta_model1.predict_proba( df_stacking_train.to_numpy() )
    meta_test_probs = meta_model1.predict_proba( df_stacking_test.to_numpy() )
    
    # STACKING (STACKING (GB, NN) RNN)
    # Train DF
    df_stacking_meta_train = pd.DataFrame({
        'rnn_0': rnn_train_probs[:, 0],
        'rnn_1': rnn_train_probs[:, 1],
        'meta_0': meta_train_probs[:, 0],
        'meta_1': meta_train_probs[:, 1],
    })

    # Test DF
    df_stacking_meta_test = pd.DataFrame({
        'rnn_0': rnn_proba[:, 0],
        'rnn_1': rnn_proba[:, 1],
        'meta_0': meta_test_probs[:, 0],
        'meta_1': meta_test_probs[:, 1],
    })

    meta_model2 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model2.fit( df_stacking_meta_train.to_numpy(), y_train_trajs_info )
    meta_probra2 = meta_model2.predict_proba( df_stacking_meta_test.to_numpy() )
   
    meta_predictions = meta_probra2[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking( Stacking (RF, NN), RNN)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking( Stacking (RF, NN), RNN)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking( Stacking (RF, NN), RNN)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    ##
    # STACKING (STACKING (RF, RNN) NN)
    
    # STACKING (RF, RNN)
    # Train DF
    df_stacking_train = pd.DataFrame({
        'rnn_0': rnn_train_probs[:, 0],
        'rnn_1': rnn_train_probs[:, 1],
        'rf_0': rf_train_probs[:, 0],
        'rf_1': rf_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        'rnn_0': rnn_proba[:, 0],
        'rnn_1': rnn_proba[:, 1],
        'rf_0': rf_proba[:, 0],
        'rf_1': rf_proba[:, 1],
    })

 
    meta_model1 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model1.fit( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_train_probs = meta_model1.predict_proba( df_stacking_train.to_numpy() )
    meta_test_probs = meta_model1.predict_proba( df_stacking_test.to_numpy() )
    
    # STACKING (STACKING (RF, DT) NN)
    # Train DF
    df_stacking_meta_train = pd.DataFrame({
        'nn_0': nn_train_probs[:, 0],
        'nn_1': nn_train_probs[:, 1],
        'meta_0': meta_train_probs[:, 0],
        'meta_1': meta_train_probs[:, 1],
    })

    # Test DF
    df_stacking_meta_test = pd.DataFrame({
        'nn_0': nn_proba[:, 0],
        'nn_1': nn_proba[:, 1],
        'meta_0': meta_test_probs[:, 0],
        'meta_1': meta_test_probs[:, 1],
    })

    meta_model2 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model2.fit( df_stacking_meta_train.to_numpy(), y_train_trajs_info )
    meta_probra2 = meta_model2.predict_proba( df_stacking_meta_test.to_numpy() )
   
    meta_predictions = meta_probra2[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking( Stacking (RF, RNN), NN)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking( Stacking (RF, RNN), NN)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking( Stacking (RF, RNN), NN)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

    ##
    # STACKING (STACKING (GB, RNN), NN)
    
    # STACKING (GB, RNN)
    # Train DF
    df_stacking_train = pd.DataFrame({
        'gb_0': gb_train_probs[:, 0],
        'gb_1': gb_train_probs[:, 1],
        'rnn_0': rnn_train_probs[:, 0],
        'rnn_1': rnn_train_probs[:, 1],
    })

    # Test DF
    df_stacking_test = pd.DataFrame({
        'gb_0': dt_proba[:, 0],
        'gb_1': dt_proba[:, 1],
        'rnn_0': rf_proba[:, 0],
        'rnn_1': rf_proba[:, 1],
    })

 
    meta_model1 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model1.fit( df_stacking_train.to_numpy(), y_train_trajs_info )
    meta_train_probs = meta_model1.predict_proba( df_stacking_train.to_numpy() )
    meta_test_probs = meta_model1.predict_proba( df_stacking_test.to_numpy() )
    
    # STACKING (STACKING (GB, RNN), NN)
    # Train DF
    df_stacking_meta_train = pd.DataFrame({
        'nn_0': nn_train_probs[:, 0],
        'nn_1': nn_train_probs[:, 1],
        'meta_0': meta_train_probs[:, 0],
        'meta_1': meta_train_probs[:, 1],
    })

    # Test DF
    df_stacking_meta_test = pd.DataFrame({
        'rf_0': rf_proba[:, 0],
        'rf_1': rf_proba[:, 1],
        'meta_0': meta_test_probs[:, 0],
        'meta_1': meta_test_probs[:, 1],
    })

    meta_model2 = RandomForestClassifier(max_depth=2, random_state=0)
    meta_model2.fit( df_stacking_meta_train.to_numpy(), y_train_trajs_info )
    meta_probra2 = meta_model2.predict_proba( df_stacking_meta_test.to_numpy() )
   
    meta_predictions = meta_probra2[:, 1]
    meta_predictions_lb = np.round(meta_predictions)
    meta_predictions_lb = np.array( [  'fishing' if i == 0 else 'sailing' for i in meta_predictions_lb  ] )
    accuracy_meta = accuracy_score(y_test_trajs_info, meta_predictions_lb)
    
    meta_score = my_precision_recall_fscore_support( y_test_trajs_info, meta_predictions_lb )
    acc_all    = np.append( acc_all, [r, 'Stacking( Stacking (GB, RNN), NN)', accuracy_meta])
    scores_all = np.append( scores_all, [r, 'Stacking( Stacking (GB, RNN), NN)', meta_score])
    probs = np.column_stack((meta_probra, y_test_trajs_info))
    probs = np.column_stack((probs, np.array(['Stacking( Stacking (GB, RNN), NN)']*n_rows_test)))   
    probs_all  = np.vstack( (probs_all, probs) )

# %% Save or load variables / for further necessity new graphics or modifications in results plots
import pickle
import datetime

# Get date e hour
dt_hr_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Filename with date and hour
file_name = f'my_results_{dt_hr_now}.pkl'

# Salvando as variáveis em um arquivo
with open(file_name, 'wb') as f:
    pickle.dump((rounds, acc_all, scores_all, probs, probs_all, train_acc), f)

# Load variables from file results / OBS: comment the write file code before uncomment this!
# with open('my_results_2023-12-01_10-17-11.pkl', 'rb') as f:
#     rounds, acc_all, scores_all, probs, probs_all = pickle.load(f)

# %%

# Formatar dados
# Extraindo os nomes das classes únicas (da segunda coluna)
n_classes = 21
acc_all = acc_all.reshape(rounds, n_classes, 3)
# classes = np.unique(acc_all[:, :, 1])
classes = acc_all[0,:,1]

# Inicializando dicionário para armazenar média e desvio padrão para cada classe
stats_by_class = {}

# Calculando média e desvio padrão para cada classe
for class_name in classes:
    class_values = acc_all[acc_all[:, :, 1] == class_name][:, 2].astype(float)
    class_mean = np.mean(class_values)
    class_std_dev = np.std(class_values)
    stats_by_class[class_name] = (class_mean, class_std_dev)

acuracies_label_all = classes

len_acc = len(acc_all)
len_labels = len(acuracies_label_all)

# Inicializando arrays para armazenar média e desvio padrão para cada classe
acc_mean = np.zeros(len(acuracies_label_all))
acc_std = np.zeros(len(acuracies_label_all))

# Calculando média e desvio padrão para cada classe e armazenando nos arrays
for idx, class_name in enumerate(classes):
    class_values = acc_all[acc_all[:, :, 1] == class_name][:, 2].astype(float)
    acc_mean[idx] = np.mean(class_values)
    acc_std[idx] = np.std(class_values)    

scores_all = scores_all.reshape(rounds, len_labels, 3)


# %%


#######
# PLOT GRAPHICS
#######

# Simple models

import matplotlib.pyplot as plt
from cycler import cycler

# Obtendo as 16 primeiras cores do mapa de cores 'tab20'
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

# Aplicando as cores ao ciclo de propriedades
plt.rc('axes', prop_cycle=(cycler('color', colors)))

results = {}
for r in range(rounds):
    for l in range(6):
        results[ acc_all[r][l][1] ] = acc_all[:,l,2].astype(float)                    

# Criar uma figura e um conjunto de subplots
plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.set_xticks(range(1, rounds+1))

# Definindo diferentes marcadores
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
m=0
# Iterar sobre os resultados e plotar uma linha para cada modelo
for model, accuracies in results.items():
    plt.plot(range(1, rounds+1), accuracies, label=model, marker=markers[m])
    m += 1

# Adicionar título, rótulos aos eixos, e legenda
plt.title('Model Accuracy over ' + str(rounds) +  ' Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

# Adicionar gridlines horizontais
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar o gráfico
plt.tight_layout()
plt.show()


# %%

# All models
import matplotlib.pyplot as plt
from cycler import cycler

# Obtendo as 16 primeiras cores do mapa de cores 'tab20'
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

# Aplicando as cores ao ciclo de propriedades
plt.rc('axes', prop_cycle=(cycler('color', colors)))

results = {}
for r in range(rounds):
    for l in range(len(acuracies_label_all)):
        results[ acc_all[r][l][1] ] = acc_all[:,l,2].astype(float)                    

# Criar uma figura e um conjunto de subplots
plt.figure(figsize=(10,6))
ax = plt.subplot(111)
ax.set_xticks(range(1, rounds+1))

# Definindo diferentes marcadores
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_', '1', '2', '3', '4', '8', 'P', 'r', 'm', 'y', 'k', 'b', 'g', 'c', 'w', 'l', 'u', 'n', 'a']
m=0
# Iterar sobre os resultados e plotar uma linha para cada modelo
for model, accuracies in results.items():
    plt.plot(range(1, rounds+1), accuracies, label=model, marker=markers[m])
    m += 1

# Adicionar título, rótulos aos eixos, e legenda
plt.title('Model Accuracy over ' + str(rounds) +  ' Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)

# Adicionar gridlines horizontais
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar o gráfico
plt.tight_layout()
plt.show()

# %%
# plot graphics (Bar graphic - only LR, DT, RF, NN, RNN and GB)
# Mean acc of models

import matplotlib.pyplot as plt

# Preparar os dados
models = []
for i in range(6):
    models.append( {"label": acuracies_label_all[i], "mean_acc": round(acc_mean[i],2), "std": round(acc_std[i],2)} )

# Extrair labels, médias e desvios padrão
labels = [model["label"] for model in models]
means = [model["mean_acc"] for model in models]
stds = [model["std"] for model in models]

# Configurar a figura e os eixos
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

# Criar o gráfico de barras e adicionar linhas de erro para o desvio padrão
ax.bar(labels, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color=colors)

# Adicionar título e rótulos aos eixos
ax.set_title('Model Accuracy Comparison (mean and standard deviation over ' +  str(rounds) + ' rounds)')
ax.set_ylabel('Mean Accuracy')
ax.set_xlabel('Model')

# Definir os limites do eixo Y
ax.set_ylim(0.8, 1.0)

# Adicionar gridlines horizontais
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Rotacionar os rótulos do eixo x para melhor visualização
plt.xticks(rotation=45, ha='right')

# Ajustar layout
plt.tight_layout()

# Mostrar o gráfico
plt.show()


# %%

# plot graphics
# Mean acc of models

import matplotlib.pyplot as plt

# Preparar os dados
models = []
for i in range(len(acuracies_label_all)):
    models.append( {"label": acuracies_label_all[i], "mean_acc": round(acc_mean[i],2), "std": round(acc_std[i],2)} )

# Extrair labels, médias e desvios padrão
labels = [model["label"] for model in models]
means = [model["mean_acc"] for model in models]
stds = [model["std"] for model in models]

# Configurar a figura e os eixos
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

# Criar o gráfico de barras e adicionar linhas de erro para o desvio padrão
ax.bar(labels, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color=colors)

# Adicionar título e rótulos aos eixos
ax.set_title('Model Accuracy Comparison (mean and standard deviation over ' +  str(rounds) + ' rounds)')
ax.set_ylabel('Mean Accuracy')
ax.set_xlabel('Model')

# Definir os limites do eixo Y
ax.set_ylim(0.8, 1.0)

# Adicionar gridlines horizontais
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Rotacionar os rótulos do eixo x para melhor visualização
plt.xticks(rotation=45, ha='right')

# Ajustar layout
plt.tight_layout()

# Mostrar o gráfico
plt.show()

# %%
# %%

## Just simple models

import matplotlib.pyplot as plt
import numpy as np

data = {}
n_models=6
# for l in range(len(acuracies_label_all)):
for l in range(n_models):
    a = np.array([])
    for i in range(rounds):
        a = np.append(a,scores_all[:,l,2][i][0])
    mean_score = a.reshape(rounds,4).astype(float).mean(axis=0)
    # mean_score = scores_all[:,l,2][0].astype(float).mean(axis=0)
    # std_score = scores_all[:,l,2][0].astype(float).std(axis=0)
    std_score = a.reshape(rounds,4).astype(float).std(axis=0)
    data[scores_all[r,l,1]] = {'Precision': {'mean':mean_score[0], 'std':std_score[0]}, 'Recall': {'mean':mean_score[1], 'std':std_score[1]}, 'F1': {'mean':mean_score[2], 'std':std_score[2]}}

# Configurações do gráfico
num_models = len(data)
bar_width = 0.1
group_width = num_models * bar_width
index = np.arange(3)  # Três grupos de barras para Precision, Recall, e F1

# Criar uma figura e um conjunto de subplots
plt.figure(figsize=(12, 8))

# Iterar sobre os modelos e adicionar barras ao gráfico
for i, (model, metrics) in enumerate(data.items()):
    means = [metrics[metric]['mean'] for metric in ['Precision', 'Recall', 'F1']]
    stds = [metrics[metric]['std'] for metric in ['Precision', 'Recall', 'F1']]
    positions = index + i * bar_width
    plt.bar(positions, means, yerr=stds, width=bar_width, label=model, alpha=0.8, capsize=10)

# Definir os limites do eixo Y
ax = plt.subplot(111)
ax.set_ylim(0.8, 1.0)

# Adicionar título, rótulos aos eixos, e legenda
plt.title('Model Evaluation Metrics for Fishing Class (Mean and Std Dev over ' + str(rounds) + ' Rounds)', fontsize=16)
plt.xlabel('Metric', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.xticks(index + (bar_width * num_models / 2), ['Precision', 'Recall', 'F1'], fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

# Adicionar gridlines horizontais
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.yticks(fontsize=15)

# Ajustar layout e mostrar o gráfico
plt.tight_layout()
plt.show()

# %%

# All models

import matplotlib.pyplot as plt
import numpy as np

data = {}
for l in range(len(acuracies_label_all)):
    a = np.array([])
    for i in range(rounds):
        a = np.append(a,scores_all[:,l,2][i][0])
    mean_score = a.reshape(rounds,4).astype(float).mean(axis=0)
    # mean_score = scores_all[:,l,2][0].astype(float).mean(axis=0)
    # std_score = scores_all[:,l,2][0].astype(float).std(axis=0)
    std_score = a.reshape(rounds,4).astype(float).std(axis=0)
    data[scores_all[r,l,1]] = {'Precision': {'mean':mean_score[0], 'std':std_score[0]}, 'Recall': {'mean':mean_score[1], 'std':std_score[1]}, 'F1': {'mean':mean_score[2], 'std':std_score[2]}}

# Configurações do gráfico
num_models = len(data)
bar_width = 0.04
group_width = num_models * bar_width
index = np.arange(3)  # Três grupos de barras para Precision, Recall, e F1

# Criar uma figura e um conjunto de subplots
plt.figure(figsize=(20, 9))

# Iterar sobre os modelos e adicionar barras ao gráfico
for i, (model, metrics) in enumerate(data.items()):
    means = [metrics[metric]['mean'] for metric in ['Precision', 'Recall', 'F1']]
    stds = [metrics[metric]['std'] for metric in ['Precision', 'Recall', 'F1']]
    positions = index + i * bar_width
    plt.bar(positions, means, yerr=stds, width=bar_width, label=model, alpha=0.8, capsize=10)

# Definir os limites do eixo Y
ax = plt.subplot(111)
ax.set_ylim(0.8, 1.0)

# Adicionar título, rótulos aos eixos, e legenda
plt.title('Model Evaluation Metrics for Fishing Class (Mean and Std Dev over ' + str(rounds) + ' Rounds)', fontsize=16)
plt.xlabel('Metric', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.xticks(index + (bar_width * num_models / 2), ['Precision', 'Recall', 'F1'], fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

# Adicionar gridlines horizontais
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.yticks(fontsize=15)

# Ajustar layout e mostrar o gráfico
plt.tight_layout()
plt.show()

# %%

###############
# TABLES
################
f = open('./logs/execution30/table_all_models.txt', 'w')
acuracies_label_all = acc_all[0,:,1]
# execution table
for l in acuracies_label_all:
    print( l + "; ",end="")
    f.write(l + "; ")
print("")
f.write("\n")

for r in range(rounds):
    for l in range(len(acuracies_label_all)):
        print(f"{acc_all[r,l,2].astype(float):.2f}; ", end="")
        f.write(f"{acc_all[r,l,2].astype(float):.2f}; ")
    print("")
    f.write("\n")

# score table
rnd = np.array([])
mdl = np.array([])
cl = np.array([])
pr = np.array([])
rc = np.array([])
f1 = np.array([])
sp = np.array([])
print("Round; Model; Class; Precision; Recall; F1; Support")
f.write("Round; Model; Class; Precision; Recall; F1; Support\n")
for r in range(rounds):
    for l in range(len(acuracies_label_all)):
        for i, c in enumerate(['fishing', 'sailing']):
            print(f"{r}; {acuracies_label_all[l]}; {c}; {scores_all[r,l,2][i][0].astype(float):.2f}; {scores_all[r,l,2][i][1].astype(float):.2f}; {scores_all[r,l,2][i][2].astype(float):.2f}; {scores_all[r,l,2][i][3].astype(float):.0f}")
            f.write( f"{r}; {acuracies_label_all[l]}; {c}; {scores_all[r,l,2][i][0].astype(float):.2f}; {scores_all[r,l,2][i][1].astype(float):.2f}; {scores_all[r,l,2][i][2].astype(float):.2f}; {scores_all[r,l,2][i][3].astype(float):.0f}\n" )
            rnd = np.append( rnd, r )
            mdl = np.append( mdl, acuracies_label_all[l] )
            cl = np.append( cl, c )
            pr = np.append( pr, scores_all[r,l,2][i][0].astype(float) )
            rc = np.append( rc, scores_all[r,l,2][i][1].astype(float) )
            f1 = np.append( f1, scores_all[r,l,2][i][2].astype(float) )
            sp = np.append( sp, scores_all[r,l,2][i][3].astype(float) )
f.close()

data_perf = {'Round':rnd, 'Model':mdl, 'Class':cl, 'Precision':pr, 'Recall':rc, 'F1':f1, 'Support':sp}
df_performance = pd.DataFrame(data=data_perf)

# %% 
#
# Print Table performance
#

# df_performance.drop(['Round'], axis='columns', inplace=True)
df_performance[df_performance['Class']=='fishing'].drop(['Class', 'Round', 'Support'], axis=1).groupby(['Model']).mean()*100

# %%
# Train acc analisys

column_names = ['model', 'acc']
df_train_acc = pd.DataFrame(train_acc.reshape( int(len(train_acc)/2), 2 ), columns=column_names)
# df_train_acc = pd.DataFrame(train_acc, columns=column_names)
df_train_acc['acc'] = df_train_acc['acc'].astype(float)
df_train_analisys = df_train_acc.groupby('model').mean()

results = {}
for r in range(rounds):
    for l in range(6):
        results[ acc_all[r][l][1] ] = acc_all[:,l,2].astype(float) 

acc_test = {}
for model, accuracies in results.items():
    acc_test[ model ] = accuracies.mean()

acc_test = dict(sorted(acc_test.items()))

df_train_analisys['acc_test'] = np.array( acc_test.values() )
df_train_analisys * 100



# %%
## Box plot Precision
import matplotlib.pyplot as plt
import seaborn as sns

df_fishing = df_performance[ df_performance['Class'] == 'fishing']

# Calcular a média de precisão para a classe 'fishing'
mean_precision = df_fishing['Precision'].mean()

# Precisão
plt.figure(figsize=(14, 6))
sns.boxplot(x='Model', y='Precision', data=df_fishing)
plt.axhline(mean_precision, color='r', linestyle='--')
plt.title('Boxplot of Precision for each Model over ' + str(rounds) + ' rounds')
# Rotacionar os rótulos do eixo x para melhor visualização
plt.xticks(rotation=45, ha='right')
# Adicionar gridlines horizontais
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %%
## Box plot Recall
import matplotlib.pyplot as plt
import seaborn as sns

df_fishing = df_performance[ df_performance['Class'] == 'fishing']

# Calcular a média de recall para a classe 'fishing'
mean_precision = df_fishing['Recall'].mean()

# Precisão
plt.figure(figsize=(14, 6))
sns.boxplot(x='Model', y='Recall', data=df_fishing)
plt.axhline(mean_precision, color='r', linestyle='--')
plt.title('Boxplot of Recall for each Model')
# Rotacionar os rótulos do eixo x para melhor visualização
plt.xticks(rotation=45, ha='right')
plt.show()


# %%
# Analise de correlação
# Stacking ALL - Analisar o quanto os modelos discordam entre si
lr_probs = lr_model.predict_proba(x_train_trajs_info)
dt_probs = dt_model.predict_proba(x_train_trajs_info)
rf_probs = rf_model.predict_proba(x_train_trajs_info)
nen_probs = nn_simple_model.predict(nn_x_train)
rnn_p = rnn_model.predict(rnn_train_x)
# svm_probs = svm_model.predict_proba(x_train_trajs_info)
gb_probs = gb_model.predict_proba(x_train_trajs_info)

# Criando o DataFrame
df_stacking_train = pd.DataFrame({
    'lr_0': lr_probs[:, 0],
    'lr_1': lr_probs[:, 1],
    'dt_0': dt_probs[:, 0],
    'dt_1': dt_probs[:, 1],
    'rf_0': rf_probs[:, 0],
    'rf_1': rf_probs[:, 1],
    'nn_0': nen_probs[:, 0],
    'nn_1': nen_probs[:, 1],
    'rnn_0': rnn_p[:, 0],
    'rnn_1': rnn_p[:, 1],
    # 'svm_0': svm_probs[:, 0],
    # 'svm_1': svm_probs[:, 1],
    'gb_0': gb_probs[:, 0],
    'gb_1': gb_probs[:, 1],
})


lr_probs_test = lr_model.predict_proba(x_test_trajs_info)
dt_probs_test = dt_model.predict_proba(x_test_trajs_info)
rf_probs_test = rf_model.predict_proba(x_test_trajs_info)
nen_probs_test = nn_simple_model.predict(nn_x_test)
rnn_p_test = rnn_model.predict(rnn_test_x)
# svm_probs_test = svm_model.predict_proba(x_test_trajs_info)
gb_probs_test = gb_model.predict_proba(x_test_trajs_info)

# Criando o DataFrame
df_stacking_test = pd.DataFrame({
    'lr_0': lr_probs_test[:, 0],
    'lr_1': lr_probs_test[:, 1],
    'dt_0': dt_probs_test[:, 0],
    'dt_1': dt_probs_test[:, 1],
    'rf_0': rf_probs_test[:, 0],
    'rf_1': rf_probs_test[:, 1],
    'nn_0': nen_probs_test[:, 0],
    'nn_1': nen_probs_test[:, 1],
    'rnn_0': rnn_p_test[:, 0],
    'rnn_1': rnn_p_test[:, 1],
    # 'svm_0': svm_probs_test[:, 0],
    # 'svm_1': svm_probs_test[:, 1],
    'gb_0': gb_probs_test[:, 0],
    'gb_1': gb_probs_test[:, 1],
})

# %%
#Correlacao Entre modelos:
df_stacking_test['Y'] = y_test_trajs_info
# dado que GB é o melhor modelo, quem prediz certo fishing quando GB erra?
# df = df_stacking_test[ (df_stacking_test['Y'] == 'fishing') & (df_stacking_test['gb_1'] > 0.5) ]
df = df_stacking_test[ (df_stacking_test['Y'] == 'fishing')  ]
# Separando as colunas de previsão para cada classe
fishing_cols = [col for col in df.columns if col.endswith('_0')]
sailing_cols = [col for col in df.columns if col.endswith('_1')]

# Calculando a matriz de correlação para cada classe
fishing_corr = df[fishing_cols].corr()
sailing_corr = df[sailing_cols].corr()

# Quem concordou menos com GB quando ele errou predizendo fishing?
fishing_corr.round(4)*100, sailing_corr.round(4)*100

# Analisando a matriz de correlação para "fishing":
# Os modelos lr e nn têm uma correlação muito alta de 0.8, o que indica que concordam significativamente em suas previsões.
# Os modelos rf e svm também têm uma correlação muito alta de 0.818, mostrando concordância entre suas previsões.
# rf e nn têm uma correlação de 0.773, indicando boa concordância.
# Equivalencias lr, nn, rf
# sem equivalencias dt, gb e rnn

# %% Save or load variables / for further necessity new graphics or modifications in results plots
import pickle
import datetime

# Get date e hour
dt_hr_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Filename with date and hour
file_name = f'my_correlation_{dt_hr_now}.pkl'

# Salvando as variáveis em um arquivo
with open(file_name, 'wb') as f:
    pickle.dump((fishing_corr, sailing_corr, df_stacking_test, df), f)

# Load variables from file results / OBS: comment the write file code before uncomment this!
# with open('my_correlatio.pkl', 'rb') as f:
#     fishing_corr, sailing_corr, df_stacking_test, df = pickle.load(f)


# %%
# TESTEEE - Utility code
 

# %%
column_names = ['prob_fishing', 'prob_sailing', 'class', 'model']
df_probs_all = pd.DataFrame(probs_all, columns=column_names)
df_probs_all['prob_fishing'] = df_probs_all['prob_fishing'].astype(float)
df_probs_all['prob_sailing'] = df_probs_all['prob_sailing'].astype(float)

# %% grafico de dispersao para detectar modelos com alta confiança que erram;

# modelos_selecionados = ['LR', 'DT', 'RF', 'NN', 'GB',  'RNN']
modelos_selecionados = ['RF', 'NN', 'GB', 'RNN']
# Definindo uma paleta de cores
cores_modelos = {
    'LR': 'pink',
    'RNN': 'blue',
    'DT': 'green',
    'RF': 'orange',
    'NN': 'purple',
    'GB': 'red'
}

df_filtrado = df_probs_all[df_probs_all['model'].isin(modelos_selecionados)]

# Supondo que a classe prevista é aquela com a maior probabilidade
df_filtrado['predicted_class'] = df_filtrado[['prob_fishing', 'prob_sailing']].idxmax(axis=1).str.replace('prob_', '')
df_filtrado['confidence'] = df_filtrado[['prob_fishing', 'prob_sailing']].max(axis=1)
df_filtrado['is_correct'] = df_filtrado['predicted_class'] == df_filtrado['class']


markers = ['o', 's', 'D', '^', 'v', 'h']
jitter = 0.45

plt.figure(figsize=(10, 8))
for i, model in enumerate(modelos_selecionados):
    subset = df_filtrado[df_filtrado['model'] == model]
    plt.scatter(subset['confidence'], subset['is_correct'] + np.random.uniform(-jitter, jitter, subset.shape[0]), label=model, alpha=0.5, marker=markers[i], color=cores_modelos[model])

plt.title('Confiança da Predição vs Precisão por Modelo')
plt.xlabel('Confiança da Predição')
plt.ylabel('Precisão (Correto/Incorreto)')
plt.yticks([0, 1], ['Incorreto', 'Correto'])
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Ajuste da posição da legenda
plt.show()


# %%

# grafico de densidade KDE

cores_modelos = {
    'LR': 'pink',
    'RNN': 'blue',
    'DT': 'green',
    'RF': 'orange',
    'NN': 'purple',
    'GB': 'red'
}

# densidade para os corretos
plt.figure(figsize=(8, 6))
for model in modelos_selecionados:
    subset = df_filtrado[ (df_filtrado['model'] == model) & (df_filtrado['is_correct'] == True)  ]
    sns.kdeplot(subset['confidence'], label=model, shade=True, color=cores_modelos[model])

plt.title('Correct Prediction Confidence Density by Model')
plt.xlabel('Prediction Confidence')
plt.ylabel('Density')
plt.xlim(0, 1)  # Limitando o eixo X de 0 a 1
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Ajuste da posição da legenda
plt.tight_layout()
plt.show()

# densidade para os errados
plt.figure(figsize=(8, 6))
for model in modelos_selecionados:
    subset = df_filtrado[ (df_filtrado['model'] == model) & (df_filtrado['is_correct'] == False)  ]
    sns.kdeplot(subset['confidence'], label=model, shade=True, color=cores_modelos[model])

plt.title('Wrong Prediction Confidence Density by Model')
plt.xlabel('Prediction Confidence')
plt.ylabel('Density')
plt.xlim(0, 1)  # Limitando o eixo X de 0 a 1
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Ajuste da posição da legenda
plt.tight_layout()
plt.show()


# %%
# MSG in telegram warning the execution end;
def msg_telegram(message):
    ip_address = "127.0.0.1"
    import socket
    try:
        # Criar um socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Conectar ao servidor
        s.connect((ip_address, 12345))
        
        # Enviar a mensagem
        s.sendall(message.encode())
        
        # Fechar o socket
        s.close()
        print(f"Message sent to {ip_address}:12345")
    except Exception as e:
        print(f"An error occurred: {e}")

msg_telegram("Execução terminada!")
tele_str = df_performance[df_performance['Class']=='fishing'].drop(['Class', 'Round', 'Support'], axis=1).groupby(['Model']).mean()*100
msg_telegram(tele_str.to_string())

# %%
df_probs_all[ df_probs_all['prob_fishing'] > 0.5 ].groupby(['model']).mean()

# %%

df_probs_all[ (df_probs_all['model'] =='GB') & (df_probs_all['prob_fishing'] > 0.5) ].mean()

# %%

# Analizys Train
column_names = ['model', 'acc']
df_train_acc = pd.DataFrame(train_acc.reshape(45,2), columns=column_names)
df_train_acc['acc'] = df_train_acc['acc'].astype(float)
df_train_acc.groupby('model').mean()


# %%
##
## Plot trajectory for use in the paper
##
# plot gdf points
def plot_gdf( gdf, vessel_description ):
    import folium

    latitude_initial = gdf.iloc[0]['lat']
    longitude_initial = gdf.iloc[0]['lon']

    m = folium.Map(location=[latitude_initial, longitude_initial], zoom_start=10)

    for _, row in gdf.iterrows():

        # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")
        # vessel_description = vessel_type_mapping.get( int( row['VesselType'] ), "Unknown")

        # Concatenar colunas para o popup
        popup_content = f"<b>Timestamp:</b> {row.name}<br><b>VesselName:</b> {row['shipname']}<br><b>MMSI</b>: {row['mmsi']}<br><b>LAT:</b> {row['lat']}<br><b>LON:</b> {row['lon']}<br><b>SOG:</b> {row['shipspeed']}<br><b>Type:</b> {vessel_description}<br><b>COG:</b> {row['shipcourse']}<br><b>Heading:</b> {row['direction']}"
        # color = mmsi_to_color( row['MMSI'] )
        
        folium.CircleMarker(
            location=[row['geometry'].y, row['geometry'].x],
            popup=popup_content,
            radius=1,  # Define o tamanho do ponto
            # color=color,  # Define a cor do ponto
            fill=True,
            fill_opacity=1
        ).add_to(m)            

    return m


def create_linestring(group):        
    # Ordenar por timestamp
    group = group.sort_values(by='timestamp')      
    # Se há mais de um ponto no grupo, crie uma LineString, caso contrário, retorne None
    return LineString(group.geometry.tolist()) if len(group) > 1 else None


# plot trajectories from points
def plot_trajectory( gdf, vessel_description ):
    import folium

    lines = gdf.groupby('mmsi').apply(create_linestring)

    # Remove possíveis None (se algum grupo tiver apenas um ponto)
    lines = lines.dropna()

    # Crie um novo GeoDataFrame com as LineStrings
    lines_gdf = gpd.GeoDataFrame(lines, columns=['geometry'], geometry='geometry')

    lines_gdf.reset_index(inplace=True)

    # start_point = Point(lines_gdf.iloc[0].geometry.coords[0])
    # m = folium.Map(location=[start_point.y, start_point.x], zoom_start=10)

    m = plot_gdf( gdf, vessel_description )

    for _, row in lines_gdf.iterrows():            
        if row['geometry'].geom_type == 'LineString':
            popup_content = f"{row['mmsi']}"
            coords = list(row['geometry'].coords)
                
            folium.PolyLine(locations=[(lat, lon) for lon, lat in coords], 
                        popup=popup_content,
                        weight=0.5
            ).add_to(m)

    return m



# %%

plot_gdf( trajs_fishing.trajectories[1].df, "Fishing" )
# %%
plot_trajectory( trajs_fishing.trajectories[2].df, "Fishing" )
# %%
plot_trajectory( trajs_no_fishing.trajectories[50].df, "Saling" )
# %%
