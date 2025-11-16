"""
Improved Cats vs Dogs CNN Classifier
Uses larger dataset (25K images) and improved architecture
"""

import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
IMG_SIZE = (224, 224)  # Larger than before for more detail
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

print("=" * 70)
print("IMPROVED CATS VS DOGS CNN CLASSIFIER")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Initial learning rate: {LEARNING_RATE}")
print(f"Max epochs: {EPOCHS}")
print("=" * 70)

# Setup paths
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data_large"

print(f"\nData directory: {DATA_DIR}")

# Download larger dataset if needed
if not DATA_DIR.exists():
    print("\n" + "=" * 70)
    print("DOWNLOADING LARGER DATASET (25,000 images)")
    print("=" * 70)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        DATA_DIR.mkdir(exist_ok=True)
        api = KaggleApi()
        api.authenticate()

        # Download a publicly available cats and dogs dataset
        print("Downloading from Kaggle: cats-and-dogs-dataset (25,000 images)...")
        api.dataset_download_files("chetankv/dogs-cats-images", path=str(DATA_DIR), unzip=True)

        print("Dataset downloaded and extracted successfully!")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease download manually from:")
        print("https://www.kaggle.com/c/dogs-vs-cats/data")
        sys.exit(1)
else:
    print("Using existing dataset...")

# Check dataset structure
print("\nChecking dataset structure...")

# The chetankv/dogs-cats-images dataset comes with a "dataset" folder
# that already has train_set and test_set with dog/cat subdirectories
if (DATA_DIR / "dataset").exists():
    print("Found dataset folder, using pre-organized structure")
    # Rename for consistency
    if not (DATA_DIR / "train").exists():
        (DATA_DIR / "dataset" / "training_set").rename(DATA_DIR / "train")
    if not (DATA_DIR / "test").exists() and (DATA_DIR / "dataset" / "test_set").exists():
        (DATA_DIR / "dataset" / "test_set").rename(DATA_DIR / "test")
else:
    print("Dataset structure ready")

# Create train/val/test splits
def create_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split the organized dataset into train/val/test"""
    import shutil

    source_dir = data_dir / "train"

    # Check if splits already exist
    if (data_dir / "train_split").exists():
        print("Splits already exist!")
        return data_dir / "train_split", data_dir / "val", data_dir / "test"

    print("\nCreating train/val/test splits...")

    train_dir = data_dir / "train_split"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / "cats").mkdir(parents=True, exist_ok=True)
        (split_dir / "dogs").mkdir(parents=True, exist_ok=True)

    # Process each class
    for class_name in ["cats", "dogs"]:
        class_dir = source_dir / class_name
        images = list(class_dir.glob("*.jpg"))

        # Shuffle
        np.random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Copy files
        for img in train_imgs:
            shutil.copy(str(img), str(train_dir / class_name / img.name))
        for img in val_imgs:
            shutil.copy(str(img), str(val_dir / class_name / img.name))
        for img in test_imgs:
            shutil.copy(str(img), str(test_dir / class_name / img.name))

        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    return train_dir, val_dir, test_dir

train_dir, val_dir, test_dir = create_splits(DATA_DIR)

# Data augmentation (aggressive)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),  # Increased from 0.05
    layers.RandomZoom(0.2),  # Increased from 0.1
    layers.RandomContrast(0.2),  # New
    layers.RandomBrightness(0.2),  # New
], name="augmentation")

# Build improved CNN architecture
def build_improved_cnn(img_size, num_classes=2):
    """
    Improved CNN with:
    - Batch normalization
    - More conv blocks
    - Deeper network
    - Better regularization
    """

    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

    # Normalization
    x = layers.Rescaling(1./255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 4 (new)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Dense layers
    x = layers.GlobalAveragePooling2D()(x)  # Better than Flatten
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='ImprovedCatsDogsCNN')

    return model

print("\n" + "=" * 70)
print("BUILDING IMPROVED CNN ARCHITECTURE")
print("=" * 70)

model = build_improved_cnn(IMG_SIZE)
model.summary()

# Create datasets
print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED
)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED
)

class_names = train_ds.class_names
print(f"\nClasses: {class_names}")

# Apply augmentation to training set
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Compile model
print("\n" + "=" * 70)
print("COMPILING MODEL")
print("=" * 70)

# Learning rate schedule with warmup
initial_lr = LEARNING_RATE
def lr_schedule(epoch):
    """Warmup + Cosine decay"""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        decay_epochs = EPOCHS - warmup_epochs
        alpha = (epoch - warmup_epochs) / decay_epochs
        cosine_decay = 0.5 * (1 + np.cos(np.pi * alpha))
        return initial_lr * cosine_decay

optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint_path = PROJECT_DIR / "improved_model_catsdogs.keras"
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    verbose=1
)

# Evaluate
print("\n" + "=" * 70)
print("EVALUATING MODEL")
print("=" * 70)

# Load best model
best_model = keras.models.load_model(checkpoint_path)

test_loss, test_acc = best_model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Save history and class names
history_path = PROJECT_DIR / "improved_history.json"
with open(history_path, 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    json.dump(history_dict, f)

class_names_path = PROJECT_DIR / "class_names.txt"
with open(class_names_path, 'w') as f:
    for name in class_names:
        f.write(name + '\n')

print(f"\nModel saved to: {checkpoint_path}")
print(f"History saved to: {history_path}")
print(f"Class names saved to: {class_names_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Final test accuracy: {test_acc:.4f}")
