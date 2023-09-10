import yfinance as yf
tesla = yf.download('TSLA',start='2010-01-01',end='2023-01-01')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(tesla['Close'].values.reshape(-1,1))
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
import numpy as np
def create_sequences(data, sequence_length):
    X,y =[], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)
sequence_length = 10
X_train, y_train =create_sequences(train_data, sequence_length)
X_test, y_test =create_sequence(test_data,sequence_length)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(tesla.index[train_size + sequence_length:],tesla['Close'][train_size + sequence_length:],lable='True Price')
plt.plot(tesla.index[train_size + sequence_length:], predictions, label='Predicted Price')
plt.legend()
plt.show()


    
