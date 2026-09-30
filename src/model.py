from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten

def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential([
        # CNN block for feature extraction
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM block for temporal sequence learning
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        
        # Fully connected layers for classification
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model