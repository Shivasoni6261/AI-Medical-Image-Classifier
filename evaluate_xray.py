import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load trained model
model = tf.keras.models.load_model("models/xray_model.keras")

# Load class names
with open("models/xray_class_names.json", "r") as file:
    class_names = json.load(file)

print("Class names:", class_names)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "chest_xray/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# Get true labels
y_true = np.concatenate([labels.numpy() for images, labels in test_ds])

# Get predictions
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Chest X-ray Confusion Matrix")
plt.tight_layout()

plt.savefig("models/xray_confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved at: models/xray_confusion_matrix.png")