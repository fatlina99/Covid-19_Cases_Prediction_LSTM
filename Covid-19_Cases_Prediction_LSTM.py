#%%
# 1. Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import callbacks, Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

import os, datetime

#%%
# 2. Data Loading
df = pd.read_csv('cases_malaysia_train.csv')
df_test = pd.read_csv('cases_malaysia_test.csv')

#%%
# 3. Data cleaning
# Remove '?' - non numeric
df.replace('?', np.nan, inplace=True)
df_test.replace('?', np.nan, inplace=True)

# Replace ' ' with NaN
df['cases_new'].replace(' ', np.nan, inplace=True)
df_test['cases_new'].replace(' ', np.nan, inplace=True)

# Drop rows with missing values (NaN)
df.dropna(inplace=True)
df_test.dropna(inplace=True)

#%%
df.info()
print(df['cases_new'].unique())

#%%
df.info()
print(df.columns)
#%%
df_test.info()
print(df_test.columns)
#%%
df = df.drop(columns = ['cases_import', 'cases_recovered', 'cases_active',
       'cases_cluster', 'cases_unvax', 'cases_pvax', 'cases_fvax',
       'cases_boost', 'cases_child', 'cases_adolescent', 'cases_adult',
       'cases_elderly', 'cases_0_4', 'cases_5_11', 'cases_12_17',
       'cases_18_29', 'cases_30_39', 'cases_40_49', 'cases_50_59',
       'cases_60_69', 'cases_70_79', 'cases_80'], axis=1)

df_test = df_test.drop(columns = ['cases_import', 'cases_recovered', 'cases_active',
       'cases_cluster', 'cases_unvax', 'cases_pvax', 'cases_fvax',
       'cases_boost', 'cases_child', 'cases_adolescent', 'cases_adult',
       'cases_elderly', 'cases_0_4', 'cases_5_11', 'cases_12_17',
       'cases_18_29', 'cases_30_39', 'cases_40_49', 'cases_50_59',
       'cases_60_69', 'cases_70_79', 'cases_80'], axis=1)

#%%
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')
df_test['cases_new'] = pd.to_numeric(df_test['cases_new'], errors='coerce')

# Interpolate missing values
df = df.interpolate(method='polynomial', order=2)
df_test = df_test.interpolate(method='polynomial', order=2)

# Apply np.floor to 'cases_new' column
df['cases_new'] = df['cases_new'].apply(np.floor)
df_test['cases_new'] = df_test['cases_new'].apply(np.floor) 

#%%
# 4. Features selection
# train dataset
X = df['cases_new'] # only 1 feature

mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))


window_size = 30
X_train = []
y_train = []

for i in range(window_size, len(X)):
    X_train.append(X[i-window_size:i])
    y_train.append(X[i])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
# %%
# Test dataset
dataset_cat = pd.concat((df['cases_new'],df_test['cases_new']))

length_days = len(dataset_cat) - len(df_test) - window_size
tot_input = dataset_cat[length_days:]

Xtest = mms.transform(np.expand_dims(tot_input,axis=-1))

X_test = []
y_test = []

for i in range(window_size, len(Xtest)):
    X_test.append(Xtest[i-window_size:i])
    y_test.append(Xtest[i])
    
X_test = np.array(X_test)
y_test = np.array(y_test)

# %%
# 5. Data preprocessing
input_shape = np.shape(X_train)[1:]

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Bidirectional(LSTM(64, return_sequences=(True))))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'linear'))
model.summary()

plot_model(model,show_shapes=True,show_layer_names= True)
#%%
# 6. Compile the model
model.compile(optimizer = 'adam', loss ='mean_squared_error', metrics=['mean_absolute_percentage_error','mse'])

# %%
# 7. TensorBoard callback
base_log_path =r"tensorboard_logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)

#%%
# 8. Model training
hist= model.fit(X_train,y_train,
                epochs= 30, 
                callbacks = [tb],
                validation_data=(X_test,y_test))

#%% model evaluation
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['mse'])
plt.plot(hist.history['val_mse'])
plt.legend(['Train MSE','Validation MSE'])
plt.show()

#%%
predicted_number_of_cases = model.predict(X_test)

plt.figure()
plt.plot(y_test, color ='red')
plt.plot(predicted_number_of_cases, color ='blue')
plt.xlabel('Time')
plt.ylabel('Number of cases')
plt.legend(['Actual','Predicted'])
plt.show()

#%%
# Plot the predicted and actual COVID cases
actual_cases = mms.inverse_transform(y_test)
predicted_cases = mms.inverse_transform(predicted_number_of_cases)

plt.figure()
plt.plot(actual_cases , color ='red')
plt.plot(predicted_cases, color ='blue')
plt.xlabel('Time')
plt.ylabel('Number of cases')
plt.legend(['Actual','Predicted'])
plt.show()

#%%
print(mean_absolute_error(actual_cases,predicted_cases))
print(mean_squared_error(actual_cases,predicted_cases))
mape_error = mean_absolute_percentage_error(actual_cases, predicted_cases)
print("Mean Absolute Percentage Error: {:.2f}%".format(mape_error * 100))

#%%
# Model's architecture 
keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
# %%
