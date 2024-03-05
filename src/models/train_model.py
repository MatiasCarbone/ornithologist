import os.path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow import optimizers


# Read JSON files with MFCCS and labels, and generate train/test/validation datasets
def get_data_splits(data_path, val_split=0.25, test_split=0.18):
    # Read JSON and extract X and Y values
    with open(data_path, 'r') as jf:
        data = json.load(jf)
    x = np.array(data['mfccs'])
    y = np.array(data['encoded_labels'])
    label_map = data['label_map']

    # Generate splits
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=test_split, shuffle=True)

    # Add third dimension to MFCC arrays
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    x_val = x_val[..., np.newaxis]

    print(
        f'• Train dataset: {x_train.shape[0]} files \n• Validation dataset: {x_val.shape[0]} files \n• Test dataset: {x_test.shape[0]} files'
    )

    return x_train, x_test, x_val, y_train, y_test, y_val, label_map


def build_model(input_shape, class_count, learning_rate=0.0001):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))

    # Convolutional layer 1
    model.add(
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same',
        )
    )

    # Convolutional layer 2
    model.add(
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
        )
    )
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.BatchNormalization())
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same',
        )
    )

    # Convolutional layer 3
    # model.add(
    #     keras.layers.Conv2D(
    #         filters=32,
    #         kernel_size=(2, 2),
    #         activation='relu',
    #         kernel_regularizer=keras.regularizers.l2(0.001),
    #     )
    # )
    # model.add(keras.layers.Dropout(0.1))
    # model.add(keras.layers.BatchNormalization())
    # model.add(
    #     keras.layers.MaxPooling2D(
    #         pool_size=(2, 2),
    #         strides=(2, 2),
    #         padding='same',
    #     )
    # )

    # Flatten + Dense layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.4))

    # Output classification layer
    model.add(keras.layers.Dense(units=class_count, activation='softmax'))

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.summary()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def training(model, batch_size, epochs, x_train, y_train, x_val, y_val, patience=10):
    earlystop_callback = keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.0001,
        patience=patience,
        verbose=0,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[earlystop_callback],
    )
    return history


def save_history_plot(history, path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.grid()
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.grid()
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    plt.savefig(path, bbox_inches='tight')


BATCH_SIZE = 32
EPOCHS = 200

json_path = './datasets/xeno_canto_birds/mfccs.json'
model_save_path = './models/00-production_model/'

x_train, x_test, x_val, y_train, y_test, y_val, label_map = get_data_splits(json_path)

input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
class_count = len(label_map)

model = build_model(input_shape, class_count, learning_rate=0.00005)

history = training(
    model,
    BATCH_SIZE,
    EPOCHS,
    x_train,
    y_train,
    x_val,
    y_val,
    patience=15,
)

model.save(os.path.join(model_save_path, 'production_model.keras'))
save_history_plot(history, os.path.join(model_save_path, 'training_history.png'))
print(f'\nSaved model and history plot to {model_save_path}!\n')
