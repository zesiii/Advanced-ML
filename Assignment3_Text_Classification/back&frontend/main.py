from data_processing import DataPreprocessor
from models import BasicCNN, GloveModel, BertModel
from performance_metrics import display_classification_report, plot_confusion_matrix
import numpy as np
from tqdm import tqdm

class ModelRunner:
    """
    Manages the process of running predictions using different types of machine learning models on text data.

    This class encapsulates the entire workflow of loading a model
    (with options for different types such as GloVe, BERT, or a basic CNN),
    preprocessing data appropriate for the chosen model, loading model weights,
    running predictions, and processing the output.

    Attributes:
        model_name (str): The name of the model to be used. Choices are 'Glove200d', 'BERT', or 'CNN'.
        model (tf.keras.Model): The machine learning model instance for running predictions.
        tokenizer (Tokenizer, optional): Tokenizer instance used for text data preprocessing.
        dp (DataPreprocessor): An instance of DataPreprocessor for handling data preprocessing tasks.
        vocab_size (int): The size of the vocabulary used in the model, preset to 13836.

    Methods:
        get_model_and_weights: Retrieves the model class and associated weights file based on the model name.
        preprocess_data: Prepares input data for prediction based on the specified model type.
        load_and_prepare_model: Loads the appropriate model and its weights.
        run: Executes the model prediction on provided data, processes and displays the prediction results.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.dp = DataPreprocessor()
        self.vocab_size = 13836

    def get_model_and_weights(self):
        """
        Retrieves the appropriate model class and associated weights file path based on the model name.

        Raises:
        - ValueError: If the model name provided is not supported.

        Returns:
        - tuple: A tuple containing the model class and the path to the weights file.
        """
        if self.model_name == "Glove200d":
            return GloveModel, 'glove_model_weights.h5'
        elif self.model_name == "BERT":
            return BertModel, 'bert_model_weights.h5'
        elif self.model_name == "CNN":
            return BasicCNN, 'basicnn_model_weights.h5'
        else:
            raise ValueError("Unsupported model type")

    def preprocess_data(self, X_test):
        """
        Preprocesses the data according to the model type specified.

        Parameters:
        - X_test (pd.DataFrame or array-like): The input data to preprocess.

        Returns:
        - tuple: Depending on the model, returns the preprocessed data and possibly an embedding matrix.
        """
        if self.model_name == "Glove200d":
            self.tokenizer = self.dp.preprocess_for_cnn(X_test)[1]
            vocab_size = len(self.tokenizer.word_index) + 1
            embeddings_index = self.dp.load_glove_embeddings("glove.6B.200d.txt", 200)
            embedding_matrix = np.zeros((vocab_size, 200))
            for word, i in self.tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            return self.dp.preprocess_for_cnn(X_test)[0], embedding_matrix
        elif self.model_name == "BERT":
            X_test_bert = X_test.to_list()
            return self.dp.prepare_data_bert(X_test_bert), None
        elif self.model_name == "CNN":
            self.tokenizer = self.dp.preprocess_for_cnn(X_test)[1]
            return self.dp.preprocess_for_cnn(X_test)[0], None

    def load_and_prepare_model(self, embedding_matrix=None):
        """
        Loads the specified model and its weights, preparing it for prediction.

        Parameters:
        - embedding_matrix (np.ndarray, optional): The embedding matrix to use, if applicable.

        """
        ModelClass, weight_file = self.get_model_and_weights()
        if self.model_name == "Glove200d":
            self.model = ModelClass(self.vocab_size)
        elif self.model_name == "BERT":
            self.model = ModelClass()
        else:
            # print(self.tokenizer)
            # self.model = ModelClass(len(self.tokenizer.word_index) + 1)
            self.model = ModelClass(self.vocab_size)
        self.model.load_weights(weight_file)

    # def run(self, X_test, y_test):
    #     preprocessed_data, embedding_matrix = self.preprocess_data(X_test)
    #     self.load_and_prepare_model(embedding_matrix)
    #
    #     if self.model_name == "BERT":
    #         predictions = self.model.predict(preprocessed_data)
    #         predictions = np.where(predictions < 0.5, 1, 0).flatten()
    #     else:
    #         predictions = self.model.predict(preprocessed_data)
    #         predictions = np.argmax(predictions, axis=1)
    #     label_map = {0: 'negative', 1: 'positive'}
    #     predictions = [label_map[int(label)] for label in predictions]
    #
    #     display_classification_report(y_test, predictions)
    #     plot_confusion_matrix(y_test, predictions)
    #     return predictions
    def run(self, X_test, y_test):
        """
        Executes the model's prediction on the provided test data and processes the outputs to display performance metrics.

        Parameters:
        - X_test (pd.DataFrame or array-like): The test data for prediction.
        - y_test (array-like): The true labels against which to evaluate the predictions.

        Returns:
        - list: A list of predicted labels for the test data.
        """
        preprocessed_data, embedding_matrix = self.preprocess_data(X_test)
        self.load_and_prepare_model(embedding_matrix)


        num_samples = len(X_test)
        batch_size = 1
        predictions = []


        with tqdm(total=num_samples, desc="Predicting", unit="sample") as pbar:
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)

                if isinstance(preprocessed_data, dict):
                    batch_data = {key: val[start_idx:end_idx] for key, val in preprocessed_data.items()}
                else:
                    batch_data = preprocessed_data[start_idx:end_idx]

                if self.model_name == "BERT":
                    batch_predictions = self.model.predict(batch_data)
                    batch_predictions = np.where(batch_predictions < 0.5, 1, 0).flatten()
                else:
                    batch_predictions = self.model.predict(batch_data)
                    batch_predictions = np.argmax(batch_predictions, axis=1)

                predictions.extend(batch_predictions)
                pbar.update(end_idx - start_idx)

        label_map = {0: 'negative', 1: 'positive'}
        predictions_labels = [label_map[int(label)] for label in predictions]

        display_classification_report(y_test, predictions_labels)
        plot_confusion_matrix(y_test, predictions_labels)
        return predictions_labels


# if __name__ == '__main__':
#     pass
    # import pandas as pd
    # test_data = pd.read_csv("test_sample.csv")
    # test_reviews = test_data.review
    # test_labels = test_data.sentiment
    # runner2 = ModelRunner('Glove200d')
    #
    # runner2.run(test_reviews, test_labels)