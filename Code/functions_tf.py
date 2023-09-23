import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Multiply, Add, Activation, Lambda, Conv1D, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras as keras


def plot_and_save_history(name: str, history, model, save_path: str, subfolder: str):
    import matplotlib.pyplot as plt
    import pickle
        
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epochs')
    plt.savefig(f'{save_path}\loss_{name}.png', dpi=300)
    pickle.dump(history.history, open(f'{save_path}\history_{name}.pkl', 'wb'))

    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.legend(['top_categorical_accuracy', 'val_top_categorical_accuracy'])
    plt.xlabel('Epochs')
    plt.ylabel('Categorical Accuracy')
    plt.ylim(0,1)
    plt.savefig(f'{save_path}\\acc_{name}.png', dpi=300)

    model.save(f'{save_path}\\{subfolder}\\{name}.h5')
    return None

class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embeddings = tf.keras.layers.Conv1D(filters=hidden_size, kernel_size=patch_size, strides=patch_size)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1, 1, self.hidden_size), trainable=True, name="cls_token")

        num_patches = input_shape[1] // self.patch_size
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.hidden_size), trainable=True, name="position_embeddings"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)  # N,H,W,C
        embeddings = self.patch_embeddings(inputs, training=training)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = tf.repeat(self.cls_token, repeats=inputs_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)

        return embeddings


class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation="gelu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool = False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dim,
        attention_bias,
        mlp_dim,
        # shapes_output,
        attention_dropout=0.0,
        sd_survival_probability=1.0,
        activation="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
            # output_shape= shapes_output
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def build(self, input_shape):
        super().build(input_shape)
        # TODO YONIGO: tf doc says to do this  ¯\_(ツ)_/¯
        self.attn._build_from_signature(input_shape, input_shape)

    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training=training)
        x = self.stochastic_depth([inputs, x], training=training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return self.stochastic_depth([x, x2], training=training)

    def get_attention_scores(self, inputs):
        x = self.norm_before(inputs, training=False)
        _, weights = self.attn(x, x, training=False, return_attention_scores=True)
        return weights


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_dim,
        num_classes,
        # output_shapes,
        dropout=0.0,
        sd_survival_probability=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embeddings = ViTEmbeddings(patch_size, hidden_size, dropout)
        sd = tf.linspace(1.0, sd_survival_probability, depth)
        self.blocks = [
            Block(
                num_heads,
                attention_dim=hidden_size,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                # shapes_output = output_shapes,
                mlp_dim=mlp_dim,
                sd_survival_probability=(sd[i].numpy().item()),
                dropout=dropout,
            )
            for i in range(depth)
        ]

        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(num_classes, 'softmax')
        

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embeddings(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        x = x[:, 0]  # take only cls_token
        return self.head(x)

    def get_last_selfattention(self, inputs: tf.Tensor):
        x = self.embeddings(inputs, training=False)
        for block in self.blocks[:-1]:
            x = block(x, training=False)
        return self.blocks[-1].get_attention_scores(x)


class ChannelAttention(Layer):
    def __init__(self, filters, ratio, **kwargs):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.shared_layer_one = Dense(self.filters // self.ratio, activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_layer_two = Dense(self.filters, kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')

    def call(self, inputs):
        avg_pool = GlobalAveragePooling1D()(inputs)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling1D()(inputs)
        max_pool = Reshape((1, self.filters))(max_pool)  # Reshape here

        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        attention = Add()([avg_pool, max_pool])
        attention = Activation('sigmoid')(attention)

        return Multiply()([inputs, attention])
    
    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'filters': self.filters, 'ratio': self.ratio})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SpatialAttention(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1d = Conv1D(filters=1, kernel_size=self.kernel_size, strides=1, 
                             padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)

    def call(self, inputs):
        avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=2, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=2, keepdims=True))(inputs)

        attention = Concatenate(axis=2)([avg_pool, max_pool])
        attention = self.conv1d(attention)

        return Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
"""DenseNet 1DCNN in Tensorflow-Keras
Reference: Densely Connected Convolutional Networks [https://arxiv.org/abs/1608.06993]
"""



def Conv_1D_Block(x, model_width, kernel, strides):
    # 1D Convolutional Block with BatchNormalization
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def stem(inputs, num_filters):
    # Construct the Stem Convolution Group
    # inputs : input vector
    conv = Conv_1D_Block(inputs, num_filters, 7, 2)
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="same")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    return pool


def conv_block(x, num_filters, bottleneck=True):
    # Construct Block of Convolutions without Pooling
    # x        : input into the block
    # n_filters: number of filters
    if bottleneck:
        num_filters_bottleneck = num_filters * 4
        x = Conv_1D_Block(x, num_filters_bottleneck, 1, 1)

    out = Conv_1D_Block(x, num_filters, 3, 1)

    return out


def dense_block(x, num_filters, num_layers, bottleneck=True):
    for i in range(num_layers):
        cb = conv_block(x, num_filters, bottleneck=bottleneck)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)

    return x


def transition_block(inputs, num_filters):
    x = Conv_1D_Block(inputs, num_filters, 1, 2)
    if x.shape[1] <= 2:
        x = tf.keras.layers.AveragePooling1D(pool_size=1, strides=2, padding="same")(x)
    else:
        x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding="same")(x)

    return x


def classifier(inputs, class_number):
    # Construct the Classifier Group
    # inputs       : input vector
    # class_number : number of output classes
    out = tf.keras.layers.Dense(class_number, activation='softmax')(inputs)

    return out


def regressor(inputs, feature_number):
    # Construct the Regressor Group
    # inputs         : input vector
    # feature_number : number of output features
    out = tf.keras.layers.Dense(feature_number, activation='linear')(inputs)

    return out


class DenseNet:
    def __init__(self, length, num_channel, num_filters, problem_type='Regression',
                 output_nums=1, pooling='avg', dropout_rate=False, bottleneck=True):
        self.length = length
        self.num_channel = num_channel
        self.num_filters = num_filters
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.bottleneck = bottleneck

    def MLP(self, x):
        if self.pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        elif self.pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
        # Final Dense Outputting Layer for the outputs
        x = tf.keras.layers.Flatten(name='flatten')(x)
        if self.dropout_rate:
            x = tf.keras.layers.Dropout(self.dropout_rate, name='Dropout')(x)
        outputs = tf.keras.layers.Dense(self.output_nums, activation='linear')(x)
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Dense(self.output_nums, activation='softmax')(x)

        return outputs

    def DenseNet121(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 24, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 16, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def DenseNet161(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters * 2)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters * 4)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 36, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters * 8)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 24, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def DenseNet169(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters * 2)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters * 4)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 32, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters * 8)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 32, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def DenseNet201(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 48, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 32, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model

    def DenseNet264(self):
        inputs = tf.keras.Input((self.length, self.num_channel))  # The input tensor
        stem_block = stem(inputs, self.num_filters)  # The Stem Convolution Group
        Dense_Block_1 = dense_block(stem_block, self.num_filters * 2, 6, bottleneck=self.bottleneck)
        Transition_Block_1 = transition_block(Dense_Block_1, self.num_filters * 2)
        Dense_Block_2 = dense_block(Transition_Block_1, self.num_filters * 4, 12, bottleneck=self.bottleneck)
        Transition_Block_2 = transition_block(Dense_Block_2, self.num_filters * 4)
        Dense_Block_3 = dense_block(Transition_Block_2, self.num_filters * 8, 64, bottleneck=self.bottleneck)
        Transition_Block_3 = transition_block(Dense_Block_3, self.num_filters * 8)
        Dense_Block_4 = dense_block(Transition_Block_3, self.num_filters * 16, 48, bottleneck=self.bottleneck)
        outputs = self.MLP(Dense_Block_4)
        # Instantiate the Model
        model = tf.keras.Model(inputs, outputs)

        return model
