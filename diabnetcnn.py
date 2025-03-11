import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Define dataset paths
diabetic_folder = r"C:/Users/karth/OneDrive/Desktop/ecg_dataset/diabetic/"
non_diabetic_folder = r"C:/Users/karth/OneDrive/Desktop/ecg_dataset/non_diabetic/"

IMG_SIZE = 224
images = []
labels = []

# ✅ Load Diabetic Images (Label = 1)
for img_name in os.listdir(diabetic_folder):
    img_path = os.path.join(diabetic_folder, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize for CNN
    images.append(image)
    labels.append(1)  # Diabetic Label

# ✅ Load Non-Diabetic Images (Label = 0)
for img_name in os.listdir(non_diabetic_folder):
    img_path = os.path.join(non_diabetic_folder, img_name)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  # Resize for CNN
    images.append(image)
    labels.append(0)  # Non-Diabetic Label

# ✅ Convert to NumPy arrays and normalize
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
labels = np.array(labels)

# ✅ Split dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# ✅ One-hot encoding for categorical labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# ✅ Apply Data Augmentation to Improve Generalization
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
)
datagen.fit(X_train)

print("✅ Dataset Loaded: ", X_train.shape, X_test.shape)


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape, BatchNormalization, GlobalAveragePooling2D

# ✅ Define DiabNet Model
model = Sequential()

# ✅ CNN Feature Extraction
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))

# ✅ Global Average Pooling (Reduces Overfitting)
model.add(GlobalAveragePooling2D())

# ✅ Reshape for RNN Input
model.add(Reshape((1, -1)))  

# ✅ LSTM for Temporal Learning
model.add(LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))

# ✅ Fully Connected Layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# ✅ Output Layer (Binary Classification)
model.add(Dense(2, activation='softmax'))  # 2 Classes: Diabetic, Non-Diabetic

# ✅ Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# ✅ Train the model using data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=8), epochs=20, validation_data=(X_test, y_test))


# ✅ Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


import matplotlib.pyplot as plt

# ✅ Plot training history
plt.figure(figsize=(12,4))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()
