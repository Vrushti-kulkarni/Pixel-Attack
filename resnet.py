import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D, Dense, Input, add, Activation,
    GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers, regularizers
import tensorflow.keras as keras
import os

class ResNet:
    def __init__(self, epochs=200, batch_size=128, load_weights=True):
        self.name = 'resnet'
        # Full path in Colab (assuming you have the file in Drive)
        self.model_filename = 'networks/models/resnet.keras'
        self.legacy_filename = 'networks/models/resnet.h5'
        
        self.stack_n = 5    
        self.num_classes = 10
        self.img_rows, self.img_cols = 32, 32
        self.img_channels = 3
        self.batch_size = batch_size
        self.epochs = epochs
        self.iterations = 50000 // self.batch_size
        self.weight_decay = 0.0001
        
        self._model = None  # Initialize model as None
        
        if load_weights:
            try:
                self._model = load_model(self.model_filename)
                print('Successfully loaded', self.name)
            except (ImportError, ValueError, OSError):
                print('Failed to load', self.name)
    
    def count_params(self):
        if self._model is None:
            print("Model not loaded or built yet")
            return 0
        return self._model.count_params()

    def color_preprocessing(self, x_train, x_test):
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        return x_train, x_test

    def scheduler(self, epoch):
        if epoch < 80:
            return 0.1
        if epoch < 150:
            return 0.01
        return 0.001

    def residual_network(self, img_input, classes_num=10, stack_n=5):
        def residual_block(intput, out_channel, increase=False):
            if increase:
                stride = (2,2)
            else:
                stride = (1,1)

            pre_bn = BatchNormalization()(intput)
            pre_relu = Activation('relu')(pre_bn)

            conv_1 = Conv2D(out_channel, kernel_size=(3,3), strides=stride, padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(pre_relu)
            bn_1 = BatchNormalization()(conv_1)
            relu1 = Activation('relu')(bn_1)
            conv_2 = Conv2D(out_channel, kernel_size=(3,3), strides=(1,1), padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(self.weight_decay))(relu1)
            if increase:
                projection = Conv2D(out_channel,
                                    kernel_size=(1,1),
                                    strides=(2,2),
                                    padding='same',
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=regularizers.l2(self.weight_decay))(intput)
                block = add([conv_2, projection])
            else:
                block = add([intput, conv_2])
            return block

        # build model
        x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(img_input)

        for _ in range(stack_n):
            x = residual_block(x, 16, False)

        x = residual_block(x, 32, True)
        for _ in range(1, stack_n):
            x = residual_block(x, 32, False)
        
        x = residual_block(x, 64, True)
        for _ in range(1, stack_n):
            x = residual_block(x, 64, False)

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(classes_num, activation='softmax',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        return x

    def build_model(self):
        """Build the model architecture without training"""
        img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels))
        output = self.residual_network(img_input, self.num_classes, self.stack_n)
        model = Model(img_input, output)
        
        sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        self._model = model
        return model

    def train(self):
        """Train the model and save in .keras format"""
        # load data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        # build network
        img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels))
        output = self.residual_network(img_input, self.num_classes, self.stack_n)
        resnet = Model(img_input, output)
        resnet.summary()

        # set optimizer
        sgd = optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
        resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # set callback
        change_lr = LearningRateScheduler(self.scheduler)
        checkpoint = ModelCheckpoint(self.model_filename, 
                monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        
        cbks = [change_lr, checkpoint]

        # set data augmentation
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=0.125,
                                    height_shift_range=0.125,
                                    fill_mode='constant', cval=0.)

        datagen.fit(x_train)

        # start training
        history = resnet.fit(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                  steps_per_epoch=self.iterations,
                  epochs=self.epochs,
                  callbacks=cbks,
                  validation_data=(x_test, y_test),
                  verbose=1)
        
        # Save final model in .keras format
        resnet.save(self.model_filename)
        print(f"Model saved to {self.model_filename} (recommended format)")
        
        # Optionally save in legacy format if needed
        try:
            resnet.save(self.legacy_filename)
            print(f"Legacy model also saved to {self.legacy_filename}")
        except Exception as e:
            print(f"Could not save legacy format: {e}")

        self._model = resnet
        return history

    def save_model(self):
        """Save the current model in .keras format"""
        if self._model is None:
            raise ValueError("No model to save. Build or load a model first.")
        try:
            self._model.save(self.model_filename)
            print(f"Model saved as {self.model_filename} (recommended format)")
        except Exception as e:
            print(f"Could not save model: {e}")

    def save_legacy(self):
        """Save the current model in legacy .h5 format"""
        if self._model is None:
            raise ValueError("No model to save. Build or load a model first.")
        try:
            self._model.save(self.legacy_filename)
            print(f"Legacy model saved as {self.legacy_filename}")
        except Exception as e:
            print(f"Could not save legacy format: {e}")

    def color_process(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return imgs

    def predict(self, img):
        if self._model is None:
            raise ValueError("Model not loaded. Either load a trained model or build a new one using build_model()")
        processed = self.color_process(img)
        return self._model.predict(processed, batch_size=self.batch_size)
    
    def predict_one(self, img):
        return self.predict(img)[0]

    def accuracy(self):
        if self._model is None:
            raise ValueError("Model not loaded. Either load a trained model or build a new one using build_model()")
            
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        # color preprocessing
        x_train, x_test = self.color_preprocessing(x_train, x_test)

        return self._model.evaluate(x_test, y_test, verbose=0)[1]