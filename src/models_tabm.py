import keras
import tensorflow as tf
from src.models_baseline import build_mlp

# Implement TabM Packed

class TabM_packed(keras.Model):
    """
    TabM_packed model: An ensemble of k independent MLPs.
    """
    def __init__(self, input_shape, num_classes, k=32, num_layers=4, hidden_units=256, dropout_rate=0.2):
        super(TabM_packed, self).__init__()
        self.k = k
        self.mlps = [build_mlp(input_shape, num_classes, num_layers, hidden_units, dropout_rate) for _ in range(k)]

    def call(self, inputs):
        """
        Processes inputs in parallel and averages the predictions.
        """
        predictions = [mlp(inputs) for mlp in self.mlps]
        return tf.reduce_mean(tf.stack(predictions), axis=0)

    def summary(self):
        """
        Prints the summary of the first MLP in the ensemble.
        """
        self.mlps[0].summary()


# Implement TabM naive
import tensorflow as tf
from keras.layers import Layer, Dense, Dropout, ReLU, Input, Lambda
from keras.models import Model

class BatchEnsembleLayer(Layer):
    """
    A parameterized BatchEnsemble layer that supports 'mini', 'naive', and 'final' TabM variants.
    """
    def __init__(self, units, k, activation=None, variant='final', **kwargs):
        super(BatchEnsembleLayer, self).__init__(**kwargs)
        self.units = units
        self.k = k
        self.activation = tf.keras.activations.get(activation)
        self.variant = variant

    def build(self, input_shape):
        """Initializes the layer's weights based on the selected variant."""
        input_dim = input_shape[-1]

        # Shared weight matrix
        self.W = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', trainable=True, name='W')

        # Non-shared adapter 'r'
        r_initializer = 'ones' if self.variant == 'final' else 'glorot_uniform'
        self.r = self.add_weight(shape=(self.k, input_dim), initializer=r_initializer, trainable=True, name='r')

        if self.variant in ['naive', 'final']:
            # Non-shared adapter 's' and bias 'b' for naive and final variants
            s_initializer = 'ones' if self.variant == 'final' else 'glorot_uniform'
            self.s = self.add_weight(shape=(self.k, self.units), initializer=s_initializer, trainable=True, name='s')
            self.b = self.add_weight(shape=(self.k, self.units), initializer='zeros', trainable=True, name='b')
        else: # 'mini' variant
            # Shared bias for the mini variant
            self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='b')

        super(BatchEnsembleLayer, self).build(input_shape)

    def call(self, inputs):
        """Performs the forward pass for the k submodels."""
        x_r = inputs * tf.expand_dims(self.r, axis=1)
        x_r_W = tf.tensordot(x_r, self.W, axes=[[2], [0]])

        if self.variant in ['naive', 'final']:
            output = x_r_W * tf.expand_dims(self.s, axis=1) + tf.expand_dims(self.b, axis=1)
        else: # 'mini' variant
            output = x_r_W + self.b

        if self.activation is not None:
            output = self.activation(output)
        return output

class SharedDenseLayer(Layer):
    """A wrapper to apply a single Dense layer across k parallel inputs."""
    def __init__(self, dense_layer, **kwargs):
        super(SharedDenseLayer, self).__init__(**kwargs)
        self.dense = dense_layer

    def call(self, inputs):
        """Applies the shared Dense layer to each of the k inputs."""
        return tf.map_fn(self.dense, inputs)

def build_tabm(variant, input_shape, num_classes, k=32, num_layers=4, hidden_units=256, dropout_rate=0.2):
    """
    Builds a TabM model based on the specified variant ('mini', 'naive', or 'final').
    """
    inputs = Input(shape=(k,) + input_shape)

    if variant == 'mini':
        # TabM_mini: BatchEnsembleLayer only for the first layer, then shared Dense layers
        x = BatchEnsembleLayer(hidden_units, k, activation='relu', variant='mini')(inputs)
        x = Dropout(dropout_rate)(x)
        for _ in range(num_layers - 1):
            shared_dense = Dense(hidden_units, activation='relu')
            x = SharedDenseLayer(shared_dense)(x)
            x = Dropout(dropout_rate)(x)
    elif variant in ['naive', 'final']:
        # TabM_naive and TabM (final): BatchEnsembleLayer for all hidden layers
        x = BatchEnsembleLayer(hidden_units, k, activation='relu', variant=variant)(inputs)
        x = Dropout(dropout_rate)(x)
        for _ in range(num_layers - 1):
            x = BatchEnsembleLayer(hidden_units, k, activation='relu', variant=variant)(x)
            x = Dropout(dropout_rate)(x)
    else:
        raise ValueError("Unknown variant: {}".format(variant))

    # Shared output layer
    output_dense = Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
    outputs = SharedDenseLayer(output_dense)(x)
    
    # Average the k predictions to get a single prediction per sample
    averaged_outputs = Lambda(lambda x: tf.reduce_mean(x, axis=1))(outputs)
    
    model = Model(inputs, averaged_outputs)
    
    return model