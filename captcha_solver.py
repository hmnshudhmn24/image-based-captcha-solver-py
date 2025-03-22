import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import string

# Define CAPTCHA character set (A-Z, 0-9)
characters = string.ascii_uppercase + string.digits
char_dict = {char: idx for idx, char in enumerate(characters)}

# Load and preprocess CAPTCHA images
def load_data(dataset_path, img_size=(50, 200)):
    images, labels = [], []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize

            label = filename.split("_")[0]  # Extract label from filename
            encoded_label = [char_dict[char] for char in label]
            
            images.append(img)
            labels.append(encoded_label)

    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels)
    return images, labels

# Define CNN model for CAPTCHA recognition
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")  # Multi-class classification
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Main training function
def train_model(dataset_path):
    images, labels = load_data(dataset_path)
    labels = to_categorical(labels, num_classes=len(characters))

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = create_model(input_shape=X_train.shape[1:], num_classes=len(characters))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save("captcha_solver.h5")
    print("Model trained and saved as captcha_solver.h5")

# Load trained model and predict CAPTCHA
def predict_captcha(image_path):
    model = tf.keras.models.load_model("captcha_solver.h5")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (50, 200)) / 255.0
    img = img.reshape(1, 50, 200, 1)

    prediction = model.predict(img)
    predicted_label = "".join([characters[np.argmax(pred)] for pred in prediction])
    return predicted_label

# Example Usage
if __name__ == "__main__":
    dataset_folder = "captcha_dataset"  # Provide dataset path
    train_model(dataset_folder)
    test_image = "test_captcha.png"  # Provide test CAPTCHA image path
    print(f"Predicted CAPTCHA: {predict_captcha(test_image)}")
