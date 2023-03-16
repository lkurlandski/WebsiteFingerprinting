"""
Convolutional Neural Network classifier.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l2


class CNN:
    """
    This class contains a CNN architecture to model Website Traffic
    Fingerprinting using the direction information from undefended data
    """

    def __init__(self, num_features, num_classes, filter_size, act_func, dropout_rate, optimizer):
        # Added some input arguments for the desired hyperparameters to tune
        """
        :param num_features: number of features (columns) in the data (X)
        :param num_classes: number of unique labels in the data (number of
            websites)
        """
        model = Sequential()
        num_filters = [32, 32]
        filter_sizes = [filter_size, filter_size]
        activation = act_func
        l2_lambda = 0.0001

        # layer 1
        model.add(
            Conv1D(
                num_filters[0],
                filter_sizes[0],
                input_shape=(num_features, 1),
                padding="same",
                activation=activation,
                kernel_regularizer=l2(l2_lambda),
            )
        )
        model.add(BatchNormalization())
        model.add(Conv1D(num_filters[1], filter_sizes[1], activation=activation, kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(4))

        # layer 2
        model.add(Conv1D(64, filter_sizes[0], padding="same", activation=activation, kernel_regularizer=l2(l2_lambda)))
        model.add(BatchNormalization())
        model.add(Conv1D(64, filter_sizes[1], activation=activation, kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(4))

        # layer 3
        model.add(Conv1D(128, filter_sizes[0], padding="same", activation=activation, kernel_regularizer=l2(l2_lambda)))
        model.add(BatchNormalization())
        model.add(Conv1D(128, filter_sizes[1], activation=activation, kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(4))

        # layer 4
        model.add(Conv1D(256, filter_sizes[0], padding="same", activation=activation, kernel_regularizer=l2(l2_lambda)))
        model.add(BatchNormalization())
        model.add(Conv1D(256, filter_sizes[1], activation=activation, kernel_regularizer=l2(l2_lambda)))
        model.add(MaxPooling1D(4))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))  # Sirinam et al.
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))

        opti = optimizer
        metrics = [
                tf.keras.metrics.Accuracy(),
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                ]
        model.compile(loss="categorical_crossentropy", optimizer=opti, metrics=metrics)
        self.model = model

    def fit(
        self,
        X_train,
        Y_train,
        batch_size,
        epochs,
        verbose=1,
        validation_split=None,
        validation_data=None,
        board=True,
        patience=None,
    ):
        """
        :param X_train: a numpy ndarray of dimension (k x n) containing
            training data
        :param Y_train: a numpy ndarray of dimension (k x 1) containing
            labels for X_train
        :param batch_size: batch size to use for training
        :param epochs: number of epochs for training
        :param verbose: Console print options for training progress.
            0 - silent mode,
            1 - progress bar,
            2 - one line per epoch
        :return: None

        This method start training the model with the given data. The
        training options are configured with tensorboard and early stopping
        callbacks.

        Tensorboard could be launched by navigating to the directory
        containing this file in terminal and running the following command.
            > tensorboard --logdir graph
        """

        callbacks = []
        if board:
            callbacks.append(TensorBoard(log_dir="./graph", histogram_freq=0, write_graph=True, write_images=True))
        if patience:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=patience))

        history = self.model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
        )
        return history
