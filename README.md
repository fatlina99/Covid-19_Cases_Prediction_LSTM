# Covid-19_Cases_Prediction_LSTM

The year 2020 was a catastrophic year for humanity. Pneumonia of unknown 
etiology was first reported in December 2019., since then, COVID-19 spread to 
the whole world and has become a global pandemic. More than 200 countries were 
affected due to the pandemic and many countries were trying to save the precious lives 
of their people by imposing travel restrictions, quarantines, social distances, event 
postponements, and lockdowns to prevent the spread of the virus. However, due 
to a lackadaisical attitude, efforts attempted by the governments were jeopardized, 
thus, predisposing to the widespread of the virus and loss of lives. 

The scientists believed that the absence of AI-assisted automated tracking and 
predicting system is the cause of the widespread COVID-19 pandemic. Hence, 
the scientist proposed the usage of a deep learning model to predict the daily 
COVID cases to determine if travel bans should be imposed or rescinded 

Thus, your task is to create a deep learning model using LSTM neural 
network to predict new cases (cases_new) in Malaysia using the past 30 days
of a number of cases.

# Architecture of the model
![model_architecture](https://github.com/fatlina99/Covid-19_Cases_Prediction_LSTM/assets/141213373/7429ae6e-b85d-402e-a6c6-12123565fe42)

# The predicted and actual COVID cases plot for MSE
![MSE plot](https://github.com/fatlina99/Covid-19_Cases_Prediction_LSTM/assets/141213373/bf0bcd9d-ed41-4d76-98ec-f22039d2db92)

The training loss (MSE) is decreasing, but the validation loss is fluctuating and sometimes even increasing. This could indicate overfitting, where the model is performing well on the training data but struggling to generalize to unseen validation data.

#  The predicted and actual COVID cases plot for MAPE
![Prediction vs actual](https://github.com/fatlina99/Covid-19_Cases_Prediction_LSTM/assets/141213373/feb2f274-3e13-4b38-b7fe-2c1ccbc9ba2a)

The Mean Absolute Percentage Error (MAPE) of 14.58% indicates that, on average, the predictions are around 14.58% off the actual values. While this is an improvement compared to the high MAPE values initially observed, it's still relatively high for accurate predictions.

# Credit
https://github.com/MoH-Malaysia/covid19-public




