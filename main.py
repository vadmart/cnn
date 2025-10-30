import matplotlib.pyplot as plt
import numpy as np

from keras import models, layers, losses, callbacks
from helpers import prepare, visualize, build_plot, rescaling
from config import *


def plot_loss_curves(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # build_plot()

    train_ds = prepare(train_ds, shuffle=True, augment=True)
    validation_ds = prepare(validation_ds)
    build_plot(train_ds)
    model = models.Sequential([
        rescaling,
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), 2),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), 2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(6, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    early_stopping = callbacks.EarlyStopping(patience=5, monitor='val_loss')

    history = model.fit(train_ds,
                        validation_data=validation_ds,
                        epochs=30,
                        callbacks=[early_stopping])
    model.save("model/cnn.keras")
    # plot_loss_curves(history)
    # predictions = model.predict(pred_ds)
    # print(f"Готово! Отримано {len(predictions)} прогнозів")
    # predicted_class_indices = np.argmax(predictions, axis=1)
    # classes = [class_names[i] for i in predicted_class_indices]
    # print(classes[:10])
