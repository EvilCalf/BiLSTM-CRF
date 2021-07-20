import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python import keras

from CRF import CRF

# from CRF import CRF


class BiLSTMCRF:
    def __init__(self, vocabSize, maxLen, tagIndexDict, tagSum, sequenceLengths=None, vecSize=100, learning_rate=0.01):
        keras.backend.clear_session()
        self.vocabSize = vocabSize
        self.vecSize = vecSize
        self.maxLen = maxLen
        self.tagSum = tagSum
        self.sequenceLengths = sequenceLengths
        self.tagIndexDict = tagIndexDict
        self.learning_rate = learning_rate

        self.buildBiLSTMCRF()

    def getTransParam(self, y, tagIndexDict):
        self.trainY = np.argmax(y, axis=-1)
        yList = self.trainY.tolist()
        transParam = np.zeros(
            [len(list(tagIndexDict.keys())), len(list(tagIndexDict.keys()))])
        for rowI in range(len(yList)):
            for colI in range(len(yList[rowI])-1):
                transParam[yList[rowI][colI]][yList[rowI][colI+1]] += 1
        for rowI in range(transParam.shape[0]):
            transParam[rowI] = transParam[rowI]/np.sum(transParam[rowI])
        return transParam

    def buildBiLSTMCRF(self):

        model = Sequential()
        model.add(tf.keras.layers.Input(shape=(self.maxLen,)))
        model.add(tf.keras.layers.Embedding(self.vocabSize, self.vecSize))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.tagSum, return_sequences=True, activation="tanh"), merge_mode='sum'))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            self.tagSum, return_sequences=True, activation="softmax"), merge_mode='sum'))
        crf = CRF(self.tagSum, name='crf_layer')
        model.add(crf)
        model.compile(Adam(learning_rate=self.learning_rate), loss={
            'crf_layer': crf.get_loss}, metrics=[crf.get_accuracy])
        self.net = model

    def fit(self, X, y, epochs=100, batchsize=32):
        if len(y.shape) == 3:
            y = np.argmax(y, axis=-1)
        if self.sequenceLengths is None:
            self.sequenceLengths = [row.shape[0] for row in y]
        callbacks_list = [
            tf.keras.callbacks.History(),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5,
                                                 verbose=1, mode='auto', min_lr=1e-9),
            tf.keras.callbacks.ModelCheckpoint("model/model.h5", monitor='get_accuracy',
                                               verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=1e-5, patience=10),
            TensorBoard(log_dir="logs", histogram_freq=1)
            # WeightsSaver(1)
        ]
        history = self.net.fit(
            X, y, epochs=epochs, callbacks=callbacks_list, batch_size=batchsize)

        return history

    def predict(self, X):
        preYArr = self.net.predict(X)
        return preYArr

    def load_weights(self, model_path):
        self.net.load_weights(model_path)
