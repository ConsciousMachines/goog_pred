# Original Code modified from Siraj Raval's video, and:
# https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction
import numpy as np 
import warnings
import matplotlib.pyplot as plt
import time
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
warnings.filterwarnings("ignore")


directory = 'googl.csv'


goog = np.genfromtxt(directory,delimiter=',',skip_header=1)
goog = goog[::-1,:] # O R I G I N A L   V A R I A B L E S
opn = goog[:,1] 
lo = goog[:,2] 
hi = goog[:,3] 
clo = goog[:,4] 
vol = goog[:,5]


y2 = clo # main variable to be predicted


moving_average = 30 # M O V I N G   A V E R A G E 
displaced_ys = []
for i in range(moving_average):
    displaced_ys.append( y2[i:-(moving_average - i)] )
avg15 = np.sum( np.array( displaced_ys), axis = 0) / moving_average


volatility = (hi/lo) / (clo/opn) # T R A N S F O R M A T I O N S 
delta_log = np.log10( clo[1:] / clo[:-1] ) # these are candidates for model inputs
delta_pc =  ((clo[1:]-clo[:-1]) / clo[:-1] )-1
vol_pc = (vol[1:] / vol[:-1]) / vol[1:]


seq_len = 101 # M O D E L   P A R A M E T E R S
train_percentage = 0.8
sequence_length = seq_len + 1


ffts1 = [] # F O U R I E R   T R A N S F O R M S 
ffts2 = [] # take the max frequencies from windows of time size 80, 40, 20,
fft_len = 80 # and are used as one of the two predictors of price. 
for i in range(seq_len,len(y2)):
    segment0 = 2/fft_len*np.abs(np.fft.fft(y2[i:i+fft_len])[:(fft_len//2)])
    scale = max(segment0)
    segment0 = segment0 / scale
    segment1 = 2/fft_len*np.abs(np.fft.fft(y2[i:i+fft_len//2])[:(fft_len//2)])
    scale = max(segment1)
    segment1 = segment1 / scale
    segment2 = 2/fft_len*np.abs(np.fft.fft(y2[i:i+fft_len//4])[:(fft_len//2)])
    scale = max(segment2)
    segment2 = segment2 / scale
    swag = np.array([np.mean(np.log10(segment0)) ,
            np.mean(np.log10(segment1)) ,
            np.mean(np.log10(segment2))])
    swagg = max(swag) * np.where(swag==max(swag))[0][0]*20 # introduce non-linearity by magnifying
    swag2 = np.array([np.max(np.log10(segment0)) , # the largest frequency by its respective time window.
            np.max(np.log10(segment1)) ,
            np.max(np.log10(segment2))])
    swagg2 = max(swag2) * np.where(swag2==max(swag2))[0][0]*20
    ffts1.append(swagg)
    ffts2.append(swagg2)


# After experimenting with many inputs, the 15 day average gives the best shape
# of the trend while the Fourier transform windows reveal underlying frequencies.


g = np.min([len(ffts1),len(avg15),len(clo)]) # make all vectors the same length

X = np.array([ffts1[-g:],ffts2[-g:],clo[-g:]]).T # main data array

result = np.zeros([1,sequence_length-1,len(X[0])]) # placeholder for input tensor
for index in range(len(X) - sequence_length-20):
    indices = [i for i in range(index,index - 1 + sequence_length)]
    + [index + sequence_length+20] # train against y_ variable 20 steps ahead
    #indices = [i for i in range(index,index + sequence_length)] # 1 step ahead 
    subset = X[indices,:]
    first_col = subset[0,:] # first row for normalisation
    first_col[first_col==0]= 1 # prevent division by 0
    norm_subset = np.array((subset / first_col) - 1)[1:,:]
    result = np.append(result,[norm_subset],axis=0)


train_size = int( train_percentage*result.shape[0] )
train_set = result[:train_size,:,:]

np.random.shuffle(train_set)
test_set = result[train_size:,:,:]

X_train = train_set[:,:-1,:] # 3D tensor of Length x Timesteps x Variables
y_train = train_set[:,-1,-1]
X_test = test_set[:,:-1,:]
y_test = test_set[:,-1,-1]


def predict_sequences_multiple(model, data, window_size, prediction_len=50):
    prediction_seqs = []
    for i in range(len(data)//prediction_len):
        predicted = []
        for j in range(prediction_len):
            curr_frame = data[i*prediction_len+j]
            prediction = model.predict(curr_frame[np.newaxis,:,:])                                                
            predicted.append(prediction[0,0])
        prediction_seqs.append(predicted)
    return prediction_seqs



def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction'+str(i))
        plt.legend()
    plt.show()


input_length = X_train.shape[1]
input_dim = X_train.shape[2]
output_dim = 1
print(input_length, input_dim, output_dim,'input_length, input_dim, output_dim')



# Siraj LSTM RNN model
model = Sequential()

model.add(LSTM(output_dim=20, input_dim=input_dim, \
    input_length = input_length, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    20,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    20,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    20,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(output_dim=output_dim))
model.add(Activation('linear')) 

model.compile(loss='mse', optimizer='rmsprop') # optimizers

model.fit(X_train,y_train,
    batch_size=500,
    nb_epoch=3,
    validation_split=0.05)

#Step 4 - Plot the predictions!
predictions = predict_sequences_multiple(model, X_test, seq_len, 100)
plot_results_multiple(predictions, y_test,100)





'''
# E X T R A S :   V I E W I N G   M O V I N G   F O U R I E R   W I N D O W
fft_len = 40
ffts = []
curvature = [1]
plt.figure(figsize=(10,8))
for i in range(len(y2)-fft_len):
    segment = 2/fft_len*np.abs(np.fft.fft(y2[i:i+fft_len])[:(fft_len//2)+1])
    scale = max(segment)
    segment = segment / scale
    swag = np.log10(segment)
    curvature.append(np.mean(np.abs(swag)))
    ffts.append(segment)
    if i >= fft_len+1:
        plt.subplot(121)
        plt.plot(swag)
        plt.subplot(122)
        plt.plot(y2[i-fft_len:i],color='green',label='True Data')
        plt.plot(curvature[i-fft_len:i],color = 'black',label = 'Prediction')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

'''

