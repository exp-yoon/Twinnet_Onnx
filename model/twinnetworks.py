from customFn import *
from tensorflow.keras import Model
from tensorflow.keras import Sequential, layers, regularizers

class TwinDiffSkip(Model):
    # With Difference skip connections
    def __init__(self, config):
        super(TwinDiffSkip, self).__init__()
        self.epsilon = config.epsilon
        self.wcce_loss_fn = CustomLoss(config).weighted_categorical_cross_entropy
        self.accuracy_fn = CustomAccuracy(config)

        # Encoder and Decoder of twin network
        weight_decay = regularizers.l2(config.weight_decay)
        self.en_conv1 = Sequential([layers.Conv2D(16, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU(),
                                    layers.Conv2D(16, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU()], name="en_conv1")
        self.maxpool1 = layers.MaxPooling2D((2, 2), padding='SAME')
        self.en_conv2 = Sequential([layers.Conv2D(32, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU(),
                                    layers.Conv2D(32, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU()], name="en_conv2")
        self.maxpool2 = layers.MaxPooling2D((2, 2), padding='SAME')
        self.upsampling1 = Sequential([layers.Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2),
                                                              padding='SAME', kernel_regularizer=weight_decay),
                                       layers.BatchNormalization(), layers.ReLU()], name="upsampling1")
        self.de_conv1 = Sequential([layers.Conv2D(32, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU(),
                                    layers.Conv2D(32, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU()], name="de_conv1")
        self.upsampling2 = Sequential([layers.Conv2DTranspose(filters=16, kernel_size=(5, 5), strides=(2, 2),
                                                              padding='SAME', kernel_regularizer=weight_decay),
                                       layers.BatchNormalization(), layers.ReLU()], name="upsampling2")
        self.de_conv2 = Sequential([layers.Conv2D(16, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU(),
                                    layers.Conv2D(16, (3, 3), padding="SAME", kernel_regularizer=weight_decay),
                                    layers.BatchNormalization(), layers.ReLU(),
                                    layers.Conv2D(2, (3, 3), padding="SAME", kernel_regularizer=weight_decay)], name="de_conv2")

        self.concat = layers.Concatenate(axis=-1)

    def call(self, inputs=None, training=False, **kwargs):
        x1, x2 = self.en_conv1(inputs[0]), self.en_conv1(inputs[1])
        skip_feats_1 = tf.abs(x1 - x2)
        x1, x2 = self.maxpool1(x1), self.maxpool1(x2)
        x1, x2 = self.en_conv2(x1), self.en_conv2(x2)
        skip_feats_2 = tf.abs(x1 - x2)
        x1, x2 = self.maxpool2(x1), self.maxpool2(x2)
        x1 = self.upsampling1(self.concat([x1, x2]))
        x1 = self.de_conv1(self.concat([x1, skip_feats_2]))
        x1 = self.upsampling2(x1)
        x1 = self.de_conv2(self.concat([x1, skip_feats_1]))
        return x1

    def train_step(self, data):
        """
        self.losses : weight_decay loss (lambda 연산 후의 값)
        :param data:
        :return:
        """
        ref_img, insp_img, label_img = data[0][0], data[0][1], data[1]

        with tf.GradientTape() as tape:
            y_pred = self((ref_img, insp_img), training=True, name="Train")  # Forward pass
            wcce_loss = self.wcce_loss_fn(label_img, y_pred)
            weight_decay_loss = tf.math.add_n(self.losses)
            loss = wcce_loss + weight_decay_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        total_acc = self.accuracy_fn.total_acc(label_img, y_pred)
        defect_acc = self.accuracy_fn.defect_acc(label_img, y_pred)
        normal_acc = self.accuracy_fn.normal_acc(label_img, y_pred)

        results = {f'{m.name}': m.result() for m in self.metrics}
        results.update({'loss': loss, 'wcce': wcce_loss, 'weight_decay': weight_decay_loss,
                        'total_acc': total_acc, 'defect_acc': defect_acc, 'normal_acc': normal_acc})

        return results

    def test_step(self, data):
        # Unpack the data
        ref_img, insp_img, label_img = data[0][0], data[0][1], data[1]

        # Compute predictions
        y_pred = self((ref_img, insp_img), training=False)  # Forward pass

        # Compute losses
        wcce_loss = self.wcce_loss_fn(label_img, y_pred)
        weight_decay_loss = tf.math.add_n(self.losses)
        loss = wcce_loss + weight_decay_loss

        # Update metrics
        total_acc = self.accuracy_fn.total_acc(label_img, y_pred)
        defect_acc = self.accuracy_fn.defect_acc(label_img, y_pred)
        normal_acc = self.accuracy_fn.normal_acc(label_img, y_pred)

        results = {f'{m.name}': m.result() for m in self.metrics}
        results.update({'loss': loss, 'wcce': wcce_loss, 'weight_decay': weight_decay_loss,
                        'total_acc': total_acc, 'defect_acc': defect_acc, 'normal_acc': normal_acc})
        return results

