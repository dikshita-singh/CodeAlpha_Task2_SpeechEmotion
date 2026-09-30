import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    print("\nGenerating Evaluation Metrics...")
    
    # 1. Get raw probability predictions from the neural network
    y_pred_probs = model.predict(X_test)
    
    # 2. Convert probabilities to actual class labels (0 to 7)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # RAVDESS Emotion Labels (0 to 7)
    emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    
    # 3. Print Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=emotions))
    
    # 4. Generate and Save Confusion Matrix
    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Emotion Recognition Confusion Matrix')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    
    # Save the plot instead of trying to open a window
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    print("Success! Confusion matrix image saved to 'models/confusion_matrix.png'.")