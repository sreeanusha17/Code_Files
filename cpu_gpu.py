from IPython import get_ipython
import pandas as pd
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, RepeatVector, TimeDistributed, BatchNormalization, Convolution2D, \
    MaxPooling2D, Flatten, Conv1D, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

pd.options.mode.chained_assignment = None
from datetime import datetime

# import geopy
from sklearn.cluster import AffinityPropagation, KMeans
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def split_series(series, n_past, n_future):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

x='ABMF'
df_nearest = pd.read_csv('E://Dashboard EQ pred/data/vtec/all_combined/2018_2021_VTEC_ABMF.csv', index_col=0)
df_nearest.index = pd.to_datetime(df_nearest.index, format='%y%m%d')
expected_data = df_nearest.tail(1).values.flatten()

df_moving_median = df_nearest.rolling(15).median()
df_moving_median = df_moving_median.dropna(axis=0)

LQ = df_nearest.rolling(15).quantile(.25)
LQ = LQ.dropna(axis=0)
UQ = df_nearest.rolling(15).quantile(.75)
UQ = UQ.dropna(axis=0)
df_UB = df_moving_median + 1.5 * (UQ - df_moving_median)
df_LB = df_moving_median - 1.5 * (df_moving_median - LQ)

flatten_data = df_moving_median.values.flatten()
flatten_data_1 = df_UB.values.flatten()
flatten_data_2 = df_LB.values.flatten()
flatten_data_3 = df_nearest.iloc[14:].values.flatten()
indexed_data = pd.date_range(df_moving_median.index[0], periods=len(flatten_data), freq='H')
data_op = {'MM_Data': flatten_data, 'Obs_Data': flatten_data_3, 'Lower_Bound': flatten_data_2,
            'Upper_Bound': flatten_data_1}
data_for_analysis = pd.DataFrame(data_op, index=indexed_data)
################################  Anomalies Count     ##########################################################
graph_plots = data_for_analysis
graph_plots['anomaly_lower'] = np.where(graph_plots['Lower_Bound'] >= graph_plots['Obs_Data'], 1, 0)
graph_plots['anomaly_upper'] = np.where(graph_plots['Obs_Data'] >= graph_plots['Upper_Bound'], 1, 0)

anomalies = graph_plots[['anomaly_lower', 'anomaly_upper']]
anomalies = anomalies.groupby(anomalies.index.date).sum()

anomalies['total'] = anomalies['anomaly_lower'] + anomalies['anomaly_upper']

count_email = anomalies.iloc[-15:]['total'][
    anomalies.iloc[-15:].apply(lambda x: x["total"] >= 8, axis=1)].count()

###################################################################################################################################
############################### ############K Means Clustering  ##################################################################
df = graph_plots

df_dst = pd.read_csv('E://Dashboard EQ pred/data/dst_data/all_combined/2018_2021_DST.csv', index_col=0)
edf_dst = df_dst.values.flatten()
df_dst = pd.DataFrame(edf_dst, columns=['DST'])
df_dst.index = pd.date_range('2018-01-01', periods=df_dst.shape[0], freq='H')

df = df.join(df_dst)

df_kp = pd.read_csv('E://Dashboard EQ pred/data/kp_ap_data/all_combined/2018_2021_KP.csv', index_col=0)
edf_kp = df_kp.values.flatten()
df_kp = pd.DataFrame(edf_kp, columns=['KP'])
df_kp.index = pd.date_range('2018-01-01', periods=df_kp.shape[0], freq='3H')
# print(df_kp.head())
df_kp = df_kp.resample('H').ffill()

df = df.join(df_kp)

df_ap = pd.read_csv('E://Dashboard EQ pred/data/kp_ap_data/all_combined/2018_2021_AP.csv', index_col=0)
edf_ap = df_ap.values.flatten()
df_ap = pd.DataFrame(edf_ap, columns=['AP'])
df_ap.index = pd.date_range('2018-01-01', periods=df_ap.shape[0], freq='3H')
# print(df_ap.head())
df_ap = df_ap.resample('H').ffill()

df = df.join(df_ap)

df_solar = pd.read_csv('E://Dashboard EQ pred/data/kp_ap_data/all_combined/2018_2021_SOLAR_FLUX.csv',
                        index_col=0)
df_solar = df_solar.drop('F10.7adj', axis=1)
df_solar.columns = ['SOLAR']
df_solar.index = pd.date_range('2018-01-01', periods=df_solar.shape[0], freq='D')
df_solar = df_solar.resample('H').ffill()

df = df.join(df_solar)
df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(df)

labels = kmeans.predict(df)

centroids = kmeans.cluster_centers_
df['labels'] = labels

df['Anomaly'] = np.abs((df['Obs_Data'] - df['MM_Data']) / df['Obs_Data'])

new_df = df[['Obs_Data', 'Anomaly', 'DST', 'KP', 'AP', 'SOLAR', 'labels']]
actual_df = df[['Obs_Data', 'Anomaly', 'DST', 'KP', 'AP', 'SOLAR', 'labels']]

n_features = 7
data=new_df
train_test_split_percentage = 0.8
validation_split_percentage = 0.2

split_training_test_starting_point = int(round(train_test_split_percentage * len(data)))
split_train_validation_starting_point = int(
    round(split_training_test_starting_point * (1 - validation_split_percentage)))

df_train = data[:split_train_validation_starting_point]
df_val = data[split_train_validation_starting_point:split_training_test_starting_point]
df_test = data[split_training_test_starting_point:]

# print(f'{df_train.shape}, {df_val.shape}, {df_test.shape}')

train = df_train
scalers = {}
for i in df_train.columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers['scaler_' + i] = scaler
    train[i] = s_s
test = df_test
for i in df_train.columns:
    scaler = scalers['scaler_' + i]
    s_s = scaler.transform(test[i].values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers['scaler_' + i] = scaler
    test[i] = s_s

val = df_val
for i in df_train.columns:
    scaler = scalers['scaler_' + i]
    s_s = scaler.transform(val[i].values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers['scaler_' + i] = scaler
    val[i] = s_s

n_future = 24*5
n_past = 24*15

X_train, y_train = split_series(train.values, n_future=n_future, n_past=n_past)
X_val, y_val = split_series(val.values, n_future=n_future, n_past=n_past)
X_test, y_test = split_series(test.values, n_future=n_future, n_past=n_past)

# y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
# y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))
# y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 7))
y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], 7))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 7))

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

# print('-----------------------------')
n_features = 7
n_future = 24*5
n_past = 24*15
encoder_inputs = Input(shape=(n_past, n_features))
encoder_l1 = LSTM(100, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)

encoder_states1 = encoder_outputs1[1:]

#
decoder_inputs = RepeatVector(n_future)(encoder_outputs1[0])

#
decoder_l1 = LSTM(100, return_sequences=True)(decoder_inputs, initial_state=encoder_states1)
decoder_outputs1 = TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l1)

#
model_e1d1 = Model(encoder_inputs, decoder_outputs1)

#
summary = model_e1d1.summary()

model_name = 'ABMF'

model_e1d1.compile(optimizer='adam', loss='mse')

import time
gpu_list=[]
batch_sizes_gpu = []
print("gpu_list : ", gpu_list)
with tf.device("/gpu:0"):
    for i in range(0,7):
        k=8*2**i
        print("batch size "+str(k))
        t1 = time.time()
        model_e1d1.fit(X_train, y_train,
                            epochs=1,
                            validation_data=(X_val, y_val),
                            batch_size=k,
                            verbose=1,
                            )
        t2 = time.time()
        gpu_list.append(int(t2-t1))
        batch_sizes_gpu.append(k)
print(gpu_list)
