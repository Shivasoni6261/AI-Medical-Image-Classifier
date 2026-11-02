import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# 1. Basic configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

TRAIN_DIR = "Training"
TEST_DIR = "Testing"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 2. Load training dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=SEED
)

# -----------------------------
# 3. Load validation dataset
# -----------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=SEED
)

# -----------------------------
# 4. Load testing dataset
# -----------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nMRI Classes:", class_names)
print("Number of classes:", num_classes)

# Save class names for the app
with open(os.path.join(MODEL_DIR, "mri_class_names.json"), "w") as f:
    json.dump(class_names, f)

# Improve dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -----------------------------
# 5. Data augmentation
# -----------------------------
data_augmentation = models.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
])

# -----------------------------
# 6. Load MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# 7. Build the classifier
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
# 8. Compile the model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 9. Train the model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
# -----------------------------
# 9B. Fine-tune MobileNetV2
# -----------------------------

print("\nStarting fine-tuning...")

# Unfreeze the base model
base_model.trainable = True

# Keep earlier layers frozen; fine-tune only the later layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Continue training
fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)
# -----------------------------
# 10. Evaluate on test dataset
# -----------------------------
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")
print("Test Loss:", round(test_loss, 4))

# -----------------------------
# 11. Save the trained model
# -----------------------------
model_path = os.path.join(MODEL_DIR, "mri_model.keras")
model.save(model_path)

print("\nModel saved successfully at:", model_path)

# -----------------------------
# 12. Plot training results
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("MRI Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(MODEL_DIR, "mri_accuracy.png"))
plt.show()