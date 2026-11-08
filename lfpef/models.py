"""Model definitions. Both models share the same backbone family and input scaling."""
import tensorflow as tf
import keras
from keras import layers


def make_backbone(cfg):
    bb = getattr(keras.applications, cfg.BACKBONE)(
        include_top=False, weights=cfg.BACKBONE_WEIGHTS,
        input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
    bb.trainable = False
    return bb


def build_frame_model(cfg, use_aux: bool):
    """Frame-level CNN -> malignancy logit (+ optional TI-RADS group head)."""
    backbone = make_backbone(cfg)
    inp = keras.Input((cfg.IMG_SIZE, cfg.IMG_SIZE, 3), name="image")   # [0, 1]
    x = layers.Rescaling(255.0, name="to_0_255")(inp)                    # backbone expects [0, 255]
    x = backbone(x, training=False)                                      # BN stays in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(cfg.DROPOUT)(x)
    outputs = {"y": layers.Dense(1, name="y")(x)}
    if use_aux:
        outputs["tr"] = layers.Dense(3, activation="softmax", name="tr")(x)
    return keras.Model(inp, outputs, name="frame_cnn"), backbone


@keras.saving.register_keras_serializable(package="lfpef")
class MaskedAttentionPool(layers.Layer):
    """Ilse et al. (2018) attention pooling with a frame mask."""

    def __init__(self, attn_dim=128, **kw):
        super().__init__(**kw)
        self.attn_dim = attn_dim
        self.v = layers.Dense(attn_dim, activation="tanh")
        self.w = layers.Dense(1)

    def call(self, feats, mask):
        a = tf.squeeze(self.w(self.v(feats)), -1)           # (B, T)
        a = a + (1.0 - mask) * -1e9
        a = tf.nn.softmax(a, axis=1)
        return tf.reduce_sum(feats * a[..., None], axis=1), a

    def get_config(self):
        return {**super().get_config(), "attn_dim": self.attn_dim}


def build_attnmil(cfg, t_max: int):
    backbone = make_backbone(cfg)
    images = keras.Input((t_max, cfg.IMG_SIZE, cfg.IMG_SIZE, 3), name="images")
    mask = keras.Input((t_max,), name="frame_mask")
    x = layers.TimeDistributed(layers.Rescaling(255.0))(images)
    x = layers.TimeDistributed(backbone)(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    bag, _ = MaskedAttentionPool(cfg.MIL_ATTN_DIM, name="attn_pool")(x, mask)
    z = layers.Dropout(cfg.DROPOUT)(layers.Dense(128, activation="relu")(bag))
    out = layers.Dense(1, name="y")(z)
    return keras.Model({"images": images, "frame_mask": mask}, out, name="attnmil"), backbone


def unfreeze_last(backbone, n_last: int):
    """Stage-2 fine-tuning: unfreeze the last n layers, keep BatchNorm frozen."""
    backbone.trainable = True
    for l in backbone.layers:
        l.trainable = False
    for l in backbone.layers[-n_last:]:
        if not isinstance(l, layers.BatchNormalization):
            l.trainable = True
