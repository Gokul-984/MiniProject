import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# Directory with coarse-grouped images
base_dir = 'data_fine'  # Each subfolder is one coarse group, e.g., group_0, group_1, ...

# Image dimensions and training parameters
img_height, img_width = 400, 400
batch_size = 32

# Data augmentation: only horizontal flip is applied
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2  # 20% for validation
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_coarse_classes = len(train_gen.class_indices)
print("Number of coarse classes:", num_coarse_classes)

# Build a simple CNN for coarse classification
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_coarse_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)

# Save the coarse model and mapping
model.save("coarse_model.h5")
coarse_class_map = train_gen.class_indices
with open("coarse_class_map.pkl", "wb") as f:
    pickle.dump(coarse_class_map, f)

print("Coarse model and mapping saved.")
