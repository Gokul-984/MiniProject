import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# Define the group folders for fine classification
group_folders = {
    "group_0": "data_fine/group_0",  # letters [y, j]
    "group_1": "data_fine/group_1",  # letters [c, o]
    "group_2": "data_fine/group_2",  # letters [g, h]
    "group_3": "data_fine/group_3",  # letters [b, d, f, i, u, v, k, x, l, r, w]
    "group_4": "data_fine/group_4",  # letters [p, q, z]
    "group_5": "data_fine/group_5",  # letters [a, e, m, n, s, t]
}

img_height, img_width = 400, 400
batch_size = 32

for group_name, folder_path in group_folders.items():
    if not os.path.exists(folder_path):
        print(f"Folder not found for {group_name}: {folder_path}")
        continue

    print(f"Training fine model for {group_name} from {folder_path}")
    
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = datagen.flow_from_directory(
        folder_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    val_gen = datagen.flow_from_directory(
        folder_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    num_classes = len(train_gen.class_indices)
    if num_classes <= 1:
        print(f"Skipping {group_name} - not enough classes (found {num_classes}).")
        continue
    
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20
    )
    
    # Save the fine model and its label map.
    model_name = f"{group_name}_model.h5"
    map_name = f"{group_name}_map.pkl"
    model.save(model_name)
    
    label_map = train_gen.class_indices
    with open(map_name, "wb") as f:
        pickle.dump(label_map, f)
    
    print(f"Saved {model_name} and {map_name} for {group_name}")
