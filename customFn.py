from utils import *
import datetime
import copy
import tensorflow.keras.backend as K
import tensorflow as tf

class CustomLoss():
    # -(y+weight*log(p) + (1-y)*(1-weight)*log(1-p))
    # predictions : without softmax / labels = 0 or 1
    # tf.math.log = log_e(x)
    def __init__(self, config):
        self.class_weight = config.class_weight
        self.epsilon = config.epsilon

    def weighted_categorical_cross_entropy(self, y_true, y_prediction, logits=True):
        total_wcce = 0
        y_prediction = [y_prediction] if str(type(y_prediction)).find('list') < 0 else y_prediction
        for iter in range(len(y_prediction)):
            y_pred = y_prediction[iter]
            labels = tf.image.resize(y_true, [tf.shape(y_pred)[1], tf.shape(y_pred)[2]])
            labels = K.squeeze(K.cast(labels > 0, tf.float32), axis=-1)

            if logits:
                y_pred_softmax = K.softmax(y_pred, axis=-1)
            else:
                y_pred_softmax = y_pred

            wcce = - (1-labels) * self.class_weight[0] * K.log(y_pred_softmax[:, :, :, 0] + self.epsilon) \
                    - labels * self.class_weight[1] * K.log(y_pred_softmax[:, :, :, 1] + self.epsilon)
            total_wcce += K.mean(wcce)
        return total_wcce / len(y_prediction)

class CustomAccuracy():
    # batch 단위 accuracy
    def __init__(self, config):
        self.img_h = config.input_size[0]
        self.img_w = config.input_size[1]
        self.img_c = config.out_classes
        self.eps = config.epsilon

    def defect_acc(self, y_true, y_pred):
        # y_true = (batch, img_h, img_w, 1)
        # y_pred = (batch, img_h, img_w, 2)
        _y_true = K.squeeze(y_true, axis=-1)
        _y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='float32') # (batch, img_h, img_w)
        num_tp = K.sum(_y_true * _y_pred)
        return num_tp / (K.sum(_y_true) + self.eps)

    def normal_acc(self, y_true, y_pred):
        # y_true = (batch, img_h, img_w, 1)
        # y_pred = (batch, img_h, img_w, 2)
        _y_true_normal = 1 - K.squeeze(y_true, axis=-1)
        _y_pred_normal = 1 - K.cast(K.argmax(y_pred, axis=-1), dtype='float32')
        num_tn = K.sum(_y_true_normal * _y_pred_normal)
        return num_tn / (K.sum(_y_true_normal) + self.eps)

    def total_acc(self, y_true, y_pred):
        # y_true = (batch, img_h, img_w, 1)
        # y_pred = (batch, img_h, img_w, 2)
        _y_true = K.squeeze(y_true, axis=-1)
        _y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='float32')  # (batch, img_h, img_w)

        _y_true_normal = 1 - _y_true
        _y_pred_normal = 1 - _y_pred

        num_tp = K.sum(_y_true * _y_pred)
        num_tn = K.sum(_y_true_normal * _y_pred_normal)
        return (num_tp + num_tn) / (K.sum(_y_true_normal) + K.sum(_y_true) + self.eps)

class CustomCallback(tf.keras.callbacks.Callback):
    # early stopping
    def __init__(self, config):
        super(CustomCallback, self).__init__()
        self.es_min_loss, self.es_step = np.inf, 0
        self.es_patience = config.es_patience
        self.valid_loss_set = []
        self.save_dir = config.save_dir

    def on_test_batch_end(self, batch, logs=None):
        self.valid_loss_set.append([logs['loss']])

    def on_epoch_end(self, epoch, logs=None):
        """
        validation : fit.loss 는 valid_loss_set[-1 (마지막 batch)] 과 같음
        -> early stopping 및 model save 할 때는 validation 의 모든 batch 를 고려한 평균 값을 사용하고자 함
        """

        valid_mean_loss = tf.reduce_mean(self.valid_loss_set).numpy()
        if self.es_min_loss <= valid_mean_loss:
            self.es_step += 1
            if self.es_step > self.es_patience:
                print(f'step > patience!, training process is stopped early..\n'
                      f'valid min loss : {self.es_min_loss}')
                self.model.stop_training = True
        else:
            self.es_step = 0
            self.es_min_loss = copy.deepcopy(valid_mean_loss)
            self.model.save(f'{self.save_dir}/min/', save_format='tf')
            self.model.save_weights(f'{self.save_dir}/ckpt/cp.ckpt')
        self.valid_loss_set = []

    def on_train_begin(self, logs=None):
        print(f"{'-'*10} Start training {'-'*10}")

    def on_train_end(self, logs=None):
        print(f"{'-' * 10} Training is finished. Check your results {'-' * 10}")



class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, config):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.decay_epoch = config.lr_decay_patience
        self.decay_rate = config.lr_decay_rate

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
          raise ValueError('Optimizer must have a "lr" attribute.')

        lr = float(K.get_value(self.model.optimizer.lr))
        lr = self.schedule(epoch, lr, self.decay_epoch, self.decay_rate)

        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
          raise ValueError('The output of the "schedule" function '
                           'should be float.')
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
          raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.optimizer.lr, K.get_value(lr))


def callbacks_lr_ReduceLROnPlateau(config, monitor="val_loss"):
    return tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=config.lr_decay_rate,
                                                patience=config.lr_decay_patience, verbose=1,
                                                mode="auto", cooldown=0, min_lr=0)


def callbacks_tensorboard(config):
    log_dir = os.path.join(config.save_dir, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), '')
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="epoch")


def lr_scheduler(epoch, lr, decay_epoch, decay_rate):
    if (epoch // decay_epoch) > 0 and np.mod(epoch, decay_epoch) == 0:
        lr = lr * decay_rate
        print('\nEpoch %03d: LearningRateScheduler reducing learning rate to %s.' % (epoch + 1, lr))
    else:
        lr = lr
    return lr
