# goog_pred
submission for Siraj Raval's weekly coding challenge

The LSTM in this network has optimal performance with 4 layers and 20 neurons per layer.
It works even better with around 5 inputs with other data such as financial macroecnomic time series, but for the purpose of the contest,
this code only includes transformations of the Google stock price history, some basic transformations, 
and a Fast Fourier transform implementation over 3 window sizes: 20, 40 and 80. 

It is possible to rig the network to create leading predictions within 60 time steps if we set the Y predition 20 steps ahead. 

Different parameters like inputs all have their uses: setting the prediction 20 time steps ahead can give us a leading prediction, 
the moving average gives a nice slope description (between 5-20 days). 

The main point of tihs network is to use the Fourier window to obtain frequency measurements of the time series data and be able to tell 
apart noise from large scale movements. For this reason we use 3 different window sizes and pick the maximum to be a feature of the 
sequences input into the LSTM. 

There is a lot of potential in putting a series of these LSTMs together with different parameters as they do a very good job of predicting 
the shape of the time series with just 80% training size. 
