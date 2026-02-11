import tensorflow as tf
from tensorflow import keras
layers = tf.keras.layers
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple

DATASET_PATH = "tom_and_jerry_training_dataset"
DATASET_TESTING_PATH = "tom_and_jerry_testing_dataset"
IMAGE_SIZE = (128, 128)  # Define the target image size
SINGLE_IMAGE_TESTS_PATH = "single_image_tests"

BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
INTERPOLATION = "bilinear"
EPOCHS = 80

data_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ], name="data_augmentation")

def clean_dataset(dataset_path) -> None:
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))
                print(f"Deleted: {os.path.join(root, file)}")

def debug_image(dataset_path) -> None:
    image = tf.io.read_file(dataset_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    
    plt.imshow(image.numpy().astype("uint8"))  # Convert to uint8 before displaying
    plt.show()

def setup_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    
    # Define class names
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,  # Use 20% for validation
        subset="training",
        seed=123,  # Ensures consistent split
        interpolation=INTERPOLATION, # Use bilinear interpolation for resizing
        label_mode="int",
    ) 

    class_names = train_dataset.class_names  # e.g. ['00','01','10','11'] (check this!)
    name_to_vec = {
        "tom": [1., 0.],
        "jerry": [0., 1.],
        "both": [1., 1.],
        "neither": [0., 0.],
    }
    classname_table = tf.constant([name_to_vec[name] for name in class_names], dtype=tf.float32)
    
    def make_multilabel_ds(ds, table, training = False):

        
        def to_multilabel(x, y):
            y2 = tf.gather(table, y)

            # Apply augmentation only on training set
            if training:
                x = data_aug(x, training=True)

            x = normalization_layer(x)
            return x, y2
        


        return ds.map(to_multilabel, num_parallel_calls=tf.data.AUTOTUNE)\
                .prefetch(tf.data.AUTOTUNE)

    
            
    train_dataset = make_multilabel_ds(train_dataset, classname_table, training=True)        

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,  # Use same split
        subset="validation",
        seed=123,
        interpolation=INTERPOLATION,
        label_mode="int",
    )
    
    val_dataset = make_multilabel_ds(val_dataset, classname_table)
    
    return train_dataset, val_dataset

# Build CNN model

def create_model() -> tf.keras.Model:

    model = keras.Sequential([
        keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        layers.Conv2D(32, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(2, activation="sigmoid")
    ])
    
    steps_per_epoch = len(train_dataset)
    total_steps = steps_per_epoch * EPOCHS

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=total_steps,
        alpha=0.05  # final lr = 0.05 * initial_lr
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,   # ← schedule goes here
        weight_decay=1e-4
    )
    # Compile the model
    model.compile(
        optimizer= optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name = "binary_accuracy"),
                tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=2, name="auc_roc"),
                tf.keras.metrics.AUC(curve="PR",  multi_label=True, num_labels=2, name="auc_pr")
                ]
    )
    model.summary()
    return model

def train_evaluate_save_model(model, train_dataset, val_dataset):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )
    ]

    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)

    model = tf.keras.models.load_model("best_model.keras")

    y_true_val, y_prob_val = collect_probs_and_labels(model, val_dataset)
    (best_tom, best_jerry), stats = tune_thresholds_for_exact_match(y_true_val, y_prob_val)
    print("Best thresholds for exact match:", (best_tom, best_jerry))
    print("Val exact match:", stats["exact"], "Val macro:", stats["macro"])
    best_ts = [best_tom, best_jerry]
    
    results = model.evaluate(val_dataset, verbose=0,return_dict=True)
    print("Validation metrics:", results)

    val_loss = results["loss"]
    # If your metric is BinaryAccuracy, name is usually "binary_accuracy"
    val_acc  = results.get("binary_accuracy", None)
    auc_roc  = results.get("auc_roc", None)
    auc_pr   = results.get("auc_pr", None)

    print(f"Validation Loss: {val_loss:.4f}")
    if val_acc is not None:
        print(f"Validation Binary Accuracy: {val_acc:.4f}")
    if auc_roc is not None:
        print(f"Validation ROC AUC: {auc_roc:.4f}")
    if auc_pr is not None:
        print(f"Validation PR AUC: {auc_pr:.4f}")
    return model, best_ts


def collect_probs_and_labels(model, ds):
    y_true_list = []
    y_prob_list = []
    for x, y in ds:
        probs = model.predict(x, verbose=0)  # (batch, 2)
        y_true_list.append(y.numpy())
        y_prob_list.append(probs)
    y_true = np.concatenate(y_true_list, axis=0).astype(np.int32)
    y_prob = np.concatenate(y_prob_list, axis=0).astype(np.float32)
    return y_true, y_prob

def prf1_from_counts(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def tune_per_label_thresholds(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    best_ts = []
    best_stats = []

    for i in range(y_true.shape[1]):
        best = {"threshold": 0.5, "f1": -1, "p": 0, "r": 0, "tp": 0, "fp": 0, "fn": 0}
        for t in thresholds:
            y_pred_i = (y_prob[:, i] >= t).astype(np.int32)
            tp = int(np.sum((y_true[:, i] == 1) & (y_pred_i == 1)))
            fp = int(np.sum((y_true[:, i] == 0) & (y_pred_i == 1)))
            fn = int(np.sum((y_true[:, i] == 1) & (y_pred_i == 0)))
            p, r, f1 = prf1_from_counts(tp, fp, fn)
            if f1 > best["f1"]:
                best = {"threshold": float(t), "f1": f1, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn}
        best_ts.append(best["threshold"])
        best_stats.append(best)
    return best_ts, best_stats

def tune_thresholds_for_exact_match(y_true, y_prob, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99) 

    best = {"t_tom": 0.5, "t_jerry": 0.5, "exact": -1, "macro": -1}

    for t_tom in grid:
        for t_jerry in grid:
            pred = (y_prob >= np.array([t_tom, t_jerry])).astype(np.int32)

            exact = np.mean(np.all(pred == y_true, axis=1))

            tom_acc   = np.mean(pred[:, 0] == y_true[:, 0])
            jerry_acc = np.mean(pred[:, 1] == y_true[:, 1])
            macro = 0.5 * (tom_acc + jerry_acc)

            # Primary objective: exact match; tie-breaker: macro
            if (exact > best["exact"]) or (exact == best["exact"] and macro > best["macro"]):
                best = {"t_tom": float(t_tom), "t_jerry": float(t_jerry), "exact": float(exact), "macro": float(macro)}

    return (best["t_tom"], best["t_jerry"]), best


def folder_to_vec(folder: str):
    mapping = {
        "tom":     np.array([1, 0], dtype=np.int32),
        "jerry":   np.array([0, 1], dtype=np.int32),
        "both":    np.array([1, 1], dtype=np.int32),
        "neither": np.array([0, 0], dtype=np.int32),
    }
    if folder not in mapping:
        return None
    return mapping[folder]

def predict_images_in_directory_multilabel(directory_path, model, best_ts = [0.5, 0.5]):
    predictions = {}

    # Accuracy counters
    correct_tom = correct_jerry = total = 0
    exact_correct = 0

    # TP / FP / FN / TN counters
    TP = np.zeros(2, dtype=int)  # [Tom, Jerry]
    FP = np.zeros(2, dtype=int)
    FN = np.zeros(2, dtype=int)
    TN = np.zeros(2, dtype=int)

    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)
        if not os.path.isdir(subdirectory_path):
            continue

        true_vec = folder_to_vec(subdirectory)
        if true_vec is None:
            continue

        for filename in os.listdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, filename)
            if not os.path.isfile(file_path):
                continue

            try:
                img = tf.keras.preprocessing.image.load_img(
                    file_path, target_size=IMAGE_SIZE
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
                img_array /= 255.0

                probs = model.predict(img_array, verbose=0)[0]
                thr_tom, thr_jerry = best_ts  # computed earlier on validation

                pred_vec = np.zeros_like(probs, dtype=np.int32)
                pred_vec[0] = int(probs[0] >= thr_tom)
                pred_vec[1] = int(probs[1] >= thr_jerry)

                predictions[file_path] = {
                    "probs": probs.tolist(),
                    "pred": pred_vec.tolist(),
                    "true": true_vec.tolist(),
                }

                # Accuracy metrics
                total += 1
                correct_tom += int(pred_vec[0] == true_vec[0])
                correct_jerry += int(pred_vec[1] == true_vec[1])
                exact_correct += int(np.all(pred_vec == true_vec))

                # TP / FP / FN / TN
                for i in range(2):
                    if true_vec[i] == 1 and pred_vec[i] == 1:
                        TP[i] += 1
                    elif true_vec[i] == 0 and pred_vec[i] == 1:
                        FP[i] += 1
                    elif true_vec[i] == 1 and pred_vec[i] == 0:
                        FN[i] += 1
                    else:
                        TN[i] += 1

            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")

    if total == 0:
        print("No images found / processed.")
        return predictions

    # Accuracy metrics
    tom_acc = correct_tom / total
    jerry_acc = correct_jerry / total
    macro_acc = (tom_acc + jerry_acc) / 2
    exact_acc = exact_correct / total

    print(f"Tom accuracy:   {tom_acc:.4f}")
    print(f"Jerry accuracy:{jerry_acc:.4f}")
    print(f"Macro avg:     {macro_acc:.4f}")
    print(f"Exact match:   {exact_acc:.4f}")

    # Precision / Recall / F1
    labels = ["Tom", "Jerry"]
    for i, name in enumerate(labels):
        precision = TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] > 0 else 0.0
        recall    = TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )

        print(f"\n{name}")
        print(f"TP={TP[i]} FP={FP[i]} FN={FN[i]} TN={TN[i]}")
        print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

    return predictions

def load_model() -> tf.keras.Model:
    return keras.models.load_model("best_model.keras")

def save_model(model) -> None:
    model.save("best_model.keras")

def predict_single_image(image_path, model, best_ts=(0.5, 0.5), threshold=None) -> dict:
    """
    Multi-label single-image prediction.
    - model output expected: [tom_prob, jerry_prob]
    - Use per-label thresholds best_ts, or a single global threshold via `threshold`.
    Returns a dict with probs, pred vector, and a human-readable label.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  

        probs = model.predict(img_array, verbose=0)[0]  # (2,)

        # Decide thresholds
        if threshold is not None:
            thr_tom = thr_jerry = float(threshold)
        else:
            thr_tom, thr_jerry = best_ts

        pred = np.zeros(2, dtype=np.int32)
        pred[0] = int(probs[0] >= thr_tom)  # Tom
        pred[1] = int(probs[1] >= thr_jerry)  # Jerry

        # Human-readable label
        if pred[0] == 1 and pred[1] == 1:
            label = "both"
        elif pred[0] == 1 and pred[1] == 0:
            label = "tom"
        elif pred[0] == 0 and pred[1] == 1:
            label = "jerry"
        else:
            label = "neither"

        result = {
            "image_path": image_path,
            "probs": {"tom": float(probs[0]), "jerry": float(probs[1])},
            "pred": {"tom": int(pred[0]), "jerry": int(pred[1])},
            "label": label,
            "thresholds": {"tom": float(thr_tom), "jerry": float(thr_jerry)},
        }

        print(
            f"{image_path}\n"
            f"  probs:  tom={result['probs']['tom']:.4f}, jerry={result['probs']['jerry']:.4f}\n"
            f"  pred:   tom={result['pred']['tom']}, jerry={result['pred']['jerry']}  -> {label}\n"
            f"  thr:    tom={result['thresholds']['tom']:.2f}, jerry={result['thresholds']['jerry']:.2f}"
        )

        return result

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {}


def try_single_images(model, best_ts=(0.5, 0.5)) -> None:
    for filename in os.listdir(SINGLE_IMAGE_TESTS_PATH):
        file_path = os.path.join(SINGLE_IMAGE_TESTS_PATH, filename)
        predict_single_image(file_path, model, best_ts=best_ts)

clean_dataset(DATASET_PATH)
clean_dataset(DATASET_TESTING_PATH)
clean_dataset(SINGLE_IMAGE_TESTS_PATH)

# 2) Build train/val datasets
train_dataset, val_dataset = setup_datasets()

# 3) Train + save
model = create_model()
best_model, best_ts =train_evaluate_save_model(model, train_dataset, val_dataset)
 
# 6) Evaluate on separate testing dataset directory
print("\n--- Testing set metrics (with tuned thresholds) ---")
predict_images_in_directory_multilabel(DATASET_TESTING_PATH, best_model, best_ts=best_ts)

# 7) Predict ALL images in SINGLE_IMAGE_TESTS_PATH
try_single_images(best_model, best_ts=best_ts)