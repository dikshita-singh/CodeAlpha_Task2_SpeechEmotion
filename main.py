import os
from sklearn.model_selection import train_test_split
from src.features import load_ravdess_data
from src.model import build_cnn_lstm_model

# NEW: Import the evaluation function
from src.evaluate import evaluate_model 

def run_pipeline():
    print("1. Loading RAVDESS Data and Extracting Audio Features...")
    try:
        X, y = load_ravdess_data("data/")
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print(f"\nData Shape: {X.shape} - Loaded successfully!")
    
    print("\n2. Splitting Data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n3. Building CNN-LSTM Model...")
    model = build_cnn_lstm_model(input_shape=(150, 40), num_classes=8)
    
    print("\n4. Training Model...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

    # NEW: Run the evaluation script
    print("\n5. Evaluating Model Performance...")
    evaluate_model(model, X_test, y_test)

    print("\n6. Saving Trained Model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/emotion_model.h5')
    print("Success! Model saved to 'models/emotion_model.h5'.")

if __name__ == "__main__":
    run_pipeline()