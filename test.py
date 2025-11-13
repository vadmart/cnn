import numpy as np
from keras import models
from config import pred_ds, class_names
from helpers import show_images_from_ds


loaded_model = models.load_model("model/cnn.keras")
predictions = loaded_model.predict(pred_ds)
prediction_classes_indices = np.argmax(predictions, axis=1)
prediction_classes = [class_names[i] for i in prediction_classes_indices]
show_images_from_ds(pred_ds, prediction_classes)
print("-----Predictions-----")
for i, pred in enumerate(predictions[:9]):
    print(f"{i}:", [round(val, 2) for val in pred])
