import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Paths
MODEL_PATH = "models/mri_model.keras"
TEST_DIR = "Testing"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open("models/mri_class_names.json", "r") as f:
    class_names = json.load(f)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get actual labels
y_true = np.concatenate([labels.numpy() for images, labels in test_ds])

# Get model predictions
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("MRI Model Confusion Matrix")
plt.tight_layout()
plt.savefig("models/mri_confusion_matrix.png")
plt.show()