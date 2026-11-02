import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# 1. Configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42

TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
TEST_DIR = "chest_xray/test"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 2. Load datasets
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nX-ray classes:", class_names)
print("Number of classes:", num_classes)

if num_classes != 2:
    raise ValueError(
        f"Expected 2 classes (NORMAL, PNEUMONIA), but found {class_names}"
    )

with open(os.path.join(MODEL_DIR, "xray_class_names.json"), "w") as f:
    json.dump(class_names, f)

# Improve data pipeline performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------------
# 3. Data augmentation
# -----------------------------
data_augmentation = models.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.08),
])

# -----------------------------
# 4. Load MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -----------------------------
# 5. Build model
# -----------------------------
inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# -----------------------------
# 6. Compile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 7. Initial training
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# 8. Fine-tuning
# -----------------------------
print("\nStarting fine-tuning...")

base_model.trainable = True

# Freeze earlier layers; unfreeze only the later layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# -----------------------------
# 9. Evaluate on test dataset
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nX-ray Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("X-ray Test Loss:", round(test_loss, 4))

# -----------------------------
# 10. Save model
# -----------------------------
model_path = os.path.join(MODEL_DIR, "xray_model.keras")
model.save(model_path)

print("\nX-ray model saved at:", model_path)

# -----------------------------
# 11. Save accuracy graph
# -----------------------------
all_accuracy = history.history["accuracy"] + fine_tune_history.history["accuracy"]
all_val_accuracy = (
    history.history["val_accuracy"] + fine_tune_history.history["val_accuracy"]
)

plt.figure(figsize=(10, 5))
plt.plot(all_accuracy, label="Training Accuracy")
plt.plot(all_val_accuracy, label="Validation Accuracy")

plt.title("Chest X-ray Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(MODEL_DIR, "xray_accuracy.png"))
plt.show()