import tensorflow as tf
'''
Silly learners. 
'''

def learn_on(dataset):
    learner = CAE_MNIST_v2(latent_dim=5, name='silly_learner')
    train(learner, on_dataset = dataset)

def train(learner, on_dataset):
    def train_step(inputs, learner, optimizer):
        with tf.GradientTape() as tape:
            x_, z = learner(inputs)
            loss = tf.keras.losses.binary_crossentropy(inputs, x_, from_logits = True)
        gradients = tape.gradient(loss, learner.trainable_variables)
        optimizer.apply_gradients(zip(gradients, learner.trainable_variables))
        return loss
    optimizer = tf.keras.optimizers.Adam()
    iterations = 1
    for batch_id, batch in enumerate(on_dataset):
        data, label = batch
        for _ in range(iterations):
            loss = train_step(data, learner, optimizer)
        print('Batch {} with loss: {}'.format(batch_id, tf.reduce_sum(loss)))

class CAE_MNIST_v2(tf.keras.Model):
    def __init__(self, latent_dim, log_step=100, **kwargs):
        super(CAE_MNIST_v2, self).__init__(**kwargs)
        self.internal_step = 0
        self.log_step = log_step

        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(28, 28, 1))  # 28x28x1
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=32,
                                               kernel_size=(5, 5),
                                               padding='valid',
                                               activation='relu')  # 24,24,10
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=64,
                                               kernel_size=(5, 5),
                                               padding='valid',
                                               activation='relu')
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=128,
                                               kernel_size=(5, 5),
                                               padding='valid',
                                               activation='relu')

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(
            units=latent_dim)  # activation='sigmoid'

        self.latent_input_layer = tf.keras.layers.InputLayer(
            input_shape=(latent_dim, ))
        self.dense_2 = tf.keras.layers.Dense(units=16 * 16 * 32,
                                             activation='relu')
        self.reshaper = tf.keras.layers.Reshape(target_shape=(16, 16, 32))
        self.conv2dt_1 = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=(5, 5),
            padding='valid',
            data_format='channels_last',
            activation='relu')
        self.conv2dt_2 = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(5, 5),
            padding='valid',
            data_format='channels_last',
            activation='relu')
        self.conv2dt_3 = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(5, 5),
            padding='valid',
            data_format='channels_last',
        )  # activation='sigmoid'

    def encoder(self, inputs, logging=True):
        x = self.input_layer(inputs)
        x = self.conv2d_1(x)
        if (logging):
            tf.summary.histogram(self.conv2d_1.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        x = self.conv2d_2(x)
        if (logging):
            tf.summary.histogram(self.conv2d_2.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        x = self.conv2d_3(x)
        if (logging):
            tf.summary.histogram(self.conv2d_3.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        x = self.flatten(x)
        x = self.dense_1(x)
        if (logging):
            tf.summary.histogram(self.dense_1.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        return x

    def decoder(self, inputs, logging=True):
        x = self.latent_input_layer(inputs)
        x = self.dense_2(x)
        if (logging):
            tf.summary.histogram(self.dense_2.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        x = self.reshaper(x)
        x = self.conv2dt_1(x)
        if (logging):
            tf.summary.histogram(self.conv2dt_1.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        x = self.conv2dt_2(x)
        if (logging):
            tf.summary.histogram(self.conv2dt_2.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        x = self.conv2dt_3(x)
        if (logging):
            tf.summary.histogram(self.conv2dt_3.name,
                                 x,
                                 step=self.internal_step,
                                 description='Visualize Activations')
        return x

    def call(self, inputs, logging=True):
        '''returns input reconstructuction, encoded input.
        '''
        log_flag = (self.internal_step % self.log_step == 0) & (logging)

        embedding = self.encoder(inputs, log_flag)
        reconstruction = self.decoder(embedding, log_flag)

        self.internal_step += 1
        return reconstruction, embedding


class CAE_CIFAR(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(CAE_CIFAR, self).__init__(**kwargs)

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),  # 28x28x1
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(5, 5),
                                   padding='valid',
                                   activation='relu'),  # 24,24,10
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(5, 5),
                                   padding='valid',
                                   activation='relu'),
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(5, 5),
                                   padding='valid',
                                   activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=latent_dim)  # activation='sigmoid'
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim, )),
            tf.keras.layers.Dense(units=16 * 16 * 32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
            tf.keras.layers.Conv2DTranspose(filters=64,
                                            kernel_size=(5, 5),
                                            padding='valid',
                                            data_format='channels_last',
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=32,
                                            kernel_size=(5, 5),
                                            padding='valid',
                                            data_format='channels_last',
                                            activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=(5, 5),
                padding='valid',
                data_format='channels_last',
            )  # activation='sigmoid'
        ])

    def call(self, inputs):
        '''returns input reconstructuction, encoded input
        '''
        embedding = self.encoder(inputs)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding