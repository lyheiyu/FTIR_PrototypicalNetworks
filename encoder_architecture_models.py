"""Only the two preregistered CE encoders for the paired architecture study."""
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='FTIR')
class SpectralBinPooling(layers.Layer):
    """Average each of 32 ordered, contiguous spectral bins independently."""
    def __init__(self, bins=32, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins

    def build(self, input_shape):
        length = int(input_shape[1])
        if length < self.bins:
            raise ValueError('Too few spectral positions for the fixed bin count')
        self.bounds = [(i * length // self.bins, (i + 1) * length // self.bins)
                       for i in range(self.bins)]
        super().build(input_shape)

    def call(self, inputs):
        return tf.stack([tf.reduce_mean(inputs[:, start:end, :], axis=1)
                         for start, end in self.bounds], axis=1)

    def get_config(self):
        return {**super().get_config(), 'bins': self.bins}


def build_encoder(architecture, length=1868, embedding_dim=128):
    inputs = layers.Input(shape=(length,), name='spectrum')
    z = layers.Reshape((length, 1))(inputs)
    if architecture == 'M0':
        for filters, kernel in [(32, 15), (64, 9), (64, 5)]:
            z = layers.Conv1D(filters, kernel, padding='same', activation='relu')(z)
            z = layers.MaxPooling1D(2)(z)
        z = layers.GlobalAveragePooling1D()(z)
    elif architecture == 'M1':
        # Multi-scale local features followed by only two factor-2 pools.
        z = layers.Concatenate()([
            layers.Conv1D(16, kernel, padding='same', activation='relu')(z)
            for kernel in [9, 25, 49]])
        for _ in range(2):
            z = layers.MaxPooling1D(2)(z)
            z = layers.Conv1D(32, 5, padding='same', activation='relu')(z)
        z = layers.Conv1D(8, 1, activation='relu', name='channel_compression')(z)
        z = SpectralBinPooling(32, name='ordered_spectral_bins')(z)
        z = layers.Flatten()(z)
    else:
        raise ValueError('Only M0 and M1 are authorized in this comparison')
    embedding = layers.Dense(embedding_dim, activation='relu', name='embedding')(z)
    return tf.keras.Model(inputs, embedding, name=f'{architecture}_encoder')


def capacity_check(length=1868, embedding_dim=128):
    models = {name: build_encoder(name, length, embedding_dim) for name in ['M0', 'M1']}
    counts = {name: model.count_params() for name, model in models.items()}
    delta = counts['M1'] / counts['M0'] - 1
    if abs(delta) > .10:
        raise ValueError(f'M1 capacity is not within 10%: {counts}')
    return {**counts, 'relative_difference': delta}
