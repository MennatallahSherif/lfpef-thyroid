"""
tf.data pipelines.

Images are returned in [0, 1]. The models multiply by 255 internally because
Keras EfficientNetV2 contains its own Rescaling(1/255) layer and expects [0, 255].
(Feeding [0, 1] directly was the bug that made the old mean/max-pooling
baseline output a constant ~0.48 for every image.)
"""
import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([img_size, img_size, 3])
    return img


def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.12)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    return tf.clip_by_value(img, 0.0, 1.0)


def class_weights(y):
    y = np.asarray(y).astype(int)
    n, n0, n1 = len(y), max((y == 0).sum(), 1), max((y == 1).sum(), 1)
    return {0: n / (2 * n0), 1: n / (2 * n1)}


# --------------------------------------------------------------------------- frames
def frame_train_ds(df: pd.DataFrame, cfg, use_aux: bool, seed: int):
    """(image, {y, tr}, {y_w, tr_w}) with class weights computed on THIS training set."""
    wy = class_weights(df["y"])
    tr_counts = df["tirads_grp"].clip(lower=0).value_counts().reindex([0, 1, 2]).fillna(0).values
    tr_w = np.where(tr_counts > 0, tr_counts.sum() / (3 * np.maximum(tr_counts, 1)), 0.0).astype("float32")

    paths = df["img_path"].astype(str).values
    y = df["y"].astype("float32").values
    tr = df["tirads_grp"].clip(lower=0).astype("int32").values
    w_y = np.where(y == 0, wy[0], wy[1]).astype("float32")
    w_tr = tr_w[tr]

    ds = tf.data.Dataset.from_tensor_slices((paths, y, tr, w_y, w_tr))
    ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def _map(p, y_, tr_, wy_, wtr_):
        img = augment(load_image(p, cfg.IMG_SIZE))
        if use_aux:
            return img, {"y": y_, "tr": tr_}, {"y": wy_, "tr": wtr_}
        return img, {"y": y_}, {"y": wy_}

    return ds.map(_map, num_parallel_calls=AUTOTUNE).batch(cfg.BATCH_SIZE).prefetch(AUTOTUNE)


def frame_predict_ds(df: pd.DataFrame, cfg):
    ds = tf.data.Dataset.from_tensor_slices(df["img_path"].astype(str).values)
    ds = ds.map(lambda p: load_image(p, cfg.IMG_SIZE), num_parallel_calls=AUTOTUNE)
    return ds.batch(cfg.BATCH_SIZE).prefetch(AUTOTUNE)


# --------------------------------------------------------------------------- bags (AttnMIL)
def make_bag_table(df: pd.DataFrame, t_max: int) -> pd.DataFrame:
    """One row per patient: padded list of paths + frame mask. Padding repeats a real
    frame, and the mask removes it from the attention, so no black images are created."""
    rows = []
    for pid, g in df.sort_values("file_name").groupby("patient_id", sort=True):
        paths = g["img_path"].astype(str).tolist()[:t_max]
        k = len(paths)
        rows.append({"patient_id": pid, "y": int(g["y"].iloc[0]),
                     "paths": paths + [paths[0]] * (t_max - k),
                     "mask": [1.0] * k + [0.0] * (t_max - k), "n_imgs": k})
    return pd.DataFrame(rows)


def bag_ds(bags: pd.DataFrame, cfg, training: bool, seed: int = 0):
    paths = np.stack(bags["paths"].values).astype(str)
    masks = np.stack(bags["mask"].values).astype("float32")
    y = bags["y"].astype("float32").values
    ds = tf.data.Dataset.from_tensor_slices((paths, masks, y))
    if training:
        ds = ds.shuffle(len(bags), seed=seed, reshuffle_each_iteration=True)

    def _map(p, m, y_):
        def _one(path):
            img = load_image(path, cfg.IMG_SIZE)
            return augment(img) if training else img
        imgs = tf.map_fn(_one, p, fn_output_signature=tf.float32)
        return {"images": imgs, "frame_mask": m}, y_

    return ds.map(_map, num_parallel_calls=AUTOTUNE).batch(cfg.MIL_BATCH).prefetch(AUTOTUNE)
