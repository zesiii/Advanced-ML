import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, Input
from keras.optimizers import Adam
from transformers import TFBertModel
import numpy as np

class BasicCNN:
    """
    A basic CNN for text classification.

    This class defines a simple CNN model suitable for classification tasks with textual data. It uses an embedding layer,
    followed by a flattening layer, dense layers, and dropout for regularization.

    Attributes:
        model (tf.keras.Model): The Keras model for the network.

    Methods:
        build_model: Constructs the CNN architecture.
        load_weights: Loads pre-trained weights into the model.
        predict: Makes predictions using the model.
    """

    def __init__(self, vocab_size):
        self.model = self.build_model(vocab_size)

    def build_model(self, vocab_size):
        """
        Builds a basic CNN model for text classification.

        Parameters:
        - vocab_size (int): The size of the vocabulary.

        Returns:
        - model (tf.keras.Model): A compiled Keras model with the CNN architecture.
        """
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=200, input_length=40),
            Flatten(),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_weights(self, path):
        """
        Loads the weights into the model from a specified file.

        Parameters:
        - path (str): The path to the weights file.
        """
        self.model.load_weights(path)

    def predict(self, x):
        """
        Predicts the class labels for the given input.

        Parameters:
        - x (array-like): Input data for which predictions are to be made.

        Returns:
        - predictions (numpy.ndarray): Predictions made by the model.
        """
        predictions = self.model.predict(x, verbose=0)
        return predictions

class GloveModel(BasicCNN):
    """
    A CNN model with pre-trained GloVe embeddings for text classification.

    This model extends BasicCNN by incorporating a pre-trained GloVe embedding layer,
    making it suitable for tasks where incorporating external semantic knowledge is beneficial.

    The GloVe embeddings are set as non-trainable to preserve their semantic properties.
    """

    def build_model(self, vocab_size):
        """
        Builds a CNN model with GloVe embeddings for text classification.

        Parameters:
        - vocab_size (int): The size of the vocabulary.

        Returns:
        - model (tf.keras.Model): A compiled Keras model with the CNN architecture using GloVe embeddings.
        """
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=200, input_length=40, weights=[np.zeros((vocab_size, 200))], trainable=False),
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

class BertModel:
    """
    A model based on BERT (Bidirectional Encoder Representations from Transformers) for text classification.

    This class encapsulates a pre-trained BERT model tailored for binary classification tasks, with specific
    layers frozen to maintain transformer encoding capabilities.

    Attributes:
        max_length (int): Maximum sequence length for BERT input.
        model (tf.keras.Model): The BERT-based Keras model.

    Methods:
        build_model: Constructs the BERT model architecture.
        load_weights: Loads pre-trained weights into the model.
        predict: Makes predictions using the model.
    """

    def __init__(self):
        self.max_length = 128
        self.model = self.build_model()

    def build_model(self):
        """
        Constructs and compiles a BERT model for binary text classification.

        Returns:
        - model (tf.keras.Model): A compiled Keras model based on BERT with tailored architecture for classification.
        """
        bert = TFBertModel.from_pretrained('bert-base-uncased')
        for layer in bert.layers:
            layer.trainable = False

        input_ids = Input(shape=(self.max_length,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_length,), dtype=tf.int32, name="attention_mask")
        outputs = bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        output = Dense(1, activation='sigmoid')(sequence_output)
        model = Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_weights(self, path):
        """
        Loads the weights into the model from a specified file.

        Parameters:
        - path (str): The path to the weights file.
        """
        self.model.load_weights(path)

    def predict(self, x):
        """
        Predicts the probability of the positive class for the given input based on BERT model.

        Parameters:
        - x (dict): Input data for the model containing 'input_ids' and 'attention_mask'.

        Returns:
        - predictions (numpy.ndarray): Probability of the positive class from the model.
        """
        predictions = self.model.predict(x, verbose=0)
        return predictions
