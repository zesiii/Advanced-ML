# preprocess, augmentation, base model cnn2, load weights, predict

import os
import numpy as np
from PIL import Image
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

class COVID19RadiographyDataset:
    def __init__(self, base_path, categories):
        self.base_path = base_path
        self.categories = categories

    def load_filenames(self):
        fnames = []
        for category in self.categories:
            image_folder = os.path.join(self.base_path, category)
            file_names = os.listdir(image_folder)
            full_path = [os.path.join(image_folder, file_name) for file_name in file_names]
            fnames.extend(full_path)
        return fnames

    @staticmethod
    def preprocessor(img_path):
        img = Image.open(img_path).convert("RGB").resize((192, 192))
        img = (np.float32(img) - 1.) / (255 - 1.)
        img = img.reshape((192, 192, 3))
        return img

    def preprocess_images(self, image_filepaths):
        preprocessed_image_data = list(map(self.preprocessor, image_filepaths))
        return np.array(preprocessed_image_data)

    def create_labels(self, num_samples_each):
        labels = []
        for category in self.categories:
            labels.extend([category.split('/')[0]] * num_samples_each)
        return pd.get_dummies(labels), labels


class AugmentedCOVID19RadiographyDataset:
    def __init__(self, X, y, image_filepaths, y_labels):
        self.X = X
        self.y = y
        self.image_filepaths = image_filepaths
        self.y_labels = y_labels
        self.augmentation_params = {
            'rescale': 1./255,
            'rotation_range': 40,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        }

    def augment_data(self, n_augmentations=1):
        datagen = ImageDataGenerator(**self.augmentation_params)

        data = pd.DataFrame({'X': self.image_filepaths, 'y': self.y_labels})
        augmented_images = []
        augmented_labels = []

        for index, row in data.iterrows():
            img = load_img(row['X'], target_size=(192, 192))
            img_array = img_to_array(img) / 255.0
            img_array = img_array.reshape((1,) + img_array.shape)

            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                augmented_images.append(batch[0])
                augmented_labels.append(row['y'])
                i += 1
                if i >= n_augmentations:
                    break

        X_augmented = np.concatenate((np.stack(augmented_images), self.X), axis=0)
        y_augmented = pd.concat([pd.get_dummies(augmented_labels), self.y])

        return X_augmented, y_augmented


class CNNModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):

        model = tf.keras.Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(192, 192, 3)))
        model.add(Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', strides=(2, 2), padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model

    def load_trained_model(self, model_path):
        self.model = load_model(model_path)
        return self.model

class Prediction:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    base_path = 'COVID-19_Radiography_Dataset'
    categories = ['COVID/images', 'Normal/images', 'Viral Pneumonia/images']
    dataset = COVID19RadiographyDataset(base_path, categories)
    print(1)
    image_filepaths = dataset.load_filenames()
    X = dataset.preprocess_images(image_filepaths)
    print(2)
    y, y_labels = dataset.create_labels(len(X) // len(categories))
    augmented_dataset = AugmentedCOVID19RadiographyDataset(X, y, image_filepaths, y_labels)
    X_augmented, y_augmented = augmented_dataset.augment_data(n_augmentations=1)
    print(3)
    cnn_model = CNNModel()
    print(4)
    trained_model = cnn_model.load_trained_model('bes_model_cnn2.h5')
    print(5)
    predictor = Prediction(trained_model)
    print(6)
    predictions = predictor.predict(X_augmented)
    print(predictions)
