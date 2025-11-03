# plot_vae_tf.py
# TensorFlow β-VAE (2→48→1) with VisualKeras diagram
# Input: (64, 64, 2)  -> Enc: 4 downsamples -> Latent(48) -> Dec: 4 upsamples -> (64, 64, 1)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import visualkeras

LATENT_DIM = 48
INPUT_SHAPE = (64, 64, 2)  # (H, W, C) matching your PyTorch model's 2 channels


# ------------------ Sampling (reparameterization) ------------------
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        std = tf.exp(0.5 * logvar)
        return mu + eps * std  # z


# ------------------ Encoder ------------------
def build_encoder(latent_dim=LATENT_DIM, input_shape=INPUT_SHAPE):
    inp = keras.Input(
        shape=input_shape, name="input_x_and_mask"
    )  # 2 channels: [x*mask, mask]

    x = layers.Conv2D(32, 3, strides=2, padding="same")(inp)  # 64->32
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)  # 32->16
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)  # 16->8
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)  # 8->4
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)  # 128*4*4
    mu = layers.Dense(latent_dim, name="z_mean")(x)
    logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    z = Sampling(name="z")([mu, logvar])

    return keras.Model(inp, [mu, logvar, z], name="Encoder")


# ------------------ Decoder ------------------
def build_decoder(latent_dim=LATENT_DIM):
    z_inp = keras.Input(shape=(latent_dim,), name="z_input")
    x = layers.Dense(128 * 4 * 4)(z_inp)
    x = layers.Reshape((4, 4, 128))(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)  # 4->8
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)  # 8->16
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x)  # 16->32
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(16, 4, strides=2, padding="same")(x)  # 32->64
    x = layers.ReLU()(x)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="sigma_hat")(x)

    return keras.Model(z_inp, out, name="Decoder")


# ------------------ End-to-end VAE (for plotting) ------------------
def build_vae(latent_dim=LATENT_DIM, input_shape=INPUT_SHAPE):
    enc = build_encoder(latent_dim, input_shape)
    dec = build_decoder(latent_dim)

    inp = keras.Input(shape=input_shape, name="vae_input")
    mu, logvar, z = enc(inp)
    xhat = dec(z)
    return keras.Model(inp, xhat, name="BetaVAE"), enc, dec


if __name__ == "__main__":
    vae, enc, dec = build_vae(LATENT_DIM, INPUT_SHAPE)
    vae.summary()  # optional: print textual summary

    # ---- VisualKeras plot (to PNG) ----
    # If you see fonts/icons missing, ensure `Pillow` is installed.
    visualkeras.layered_view(
        vae,
        legend=True,
        to_file="vae_visualkeras.png",  # output image
        scale_xy=1.3,
        scale_z=1,
        type_ignore=[layers.ReLU, layers.Flatten, Sampling],  # simplify the view
    )
    print("Saved visual diagram to 'vae_visualkeras.png'")
