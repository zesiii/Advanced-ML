import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from transformers import BertTokenizer

class DataPreprocessor:
    """
        A class for preprocessing textual data for various types of deep learning models.

        This class provides methods to tokenize and pad textual data for convolutional neural networks (CNNs),
        load pre-trained GloVe embeddings, and preprocess text data for BERT models using the Transformers library.

        Attributes:
            tokenizer (Tokenizer, optional): Tokenizer for converting text to sequences. Initialized as None.
            max_len (int): Maximum length of sequences for padding. Defaults to 40 for CNN processing.
    """
    def __init__(self):
        self.tokenizer = None
        self.max_len = 40

    def preprocess_for_cnn(self, txt_in):
        """
        Preprocesses text input for CNN models.

        This method tokenizes textual input and pads the resulting sequences to a uniform length to prepare
        for input into CNN models.

        Parameters:
        - txt_in (list of str): The text input to process. Each entry is a single string (a document or sentence).

        Returns:
        - tuple: A tuple containing the following elements:
            - numpy.ndarray: Padded sequences of tokens.
            - Tokenizer: The tokenizer used to process the text.
        """
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(txt_in)
        sequences = self.tokenizer.texts_to_sequences(txt_in)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return padded_sequences, self.tokenizer

    def load_glove_embeddings(self, path, embedding_dim=200):
        """
        Loads GloVe word embeddings from a file.

        Parameters:
        - path (str): Path to the GloVe embeddings file.
        - embedding_dim (int): Dimension of the embeddings. Defaults to 200.

        Returns:
        - dict: A dictionary where keys are words and values are their corresponding embedding vectors as numpy arrays.
        """
        embeddings_index = {}
        with open(path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def prepare_data_bert(self, texts, max_length=128):
        """
        Preprocesses text data for BERT models.

        This method tokenizes text input according to BERT model requirements and prepares necessary inputs like
        'input_ids' and 'attention_mask'.

        Parameters:
        - texts (list of str): The text input to process. Each entry is a single string (a document or sentence).
        - max_length (int): Maximum length of sequences after tokenization. Defaults to 128.

        Returns:
        - dict: A dictionary containing the tokenized data with keys 'input_ids' and 'attention_mask'.
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized = tokenizer(texts, max_length=max_length, truncation=True, padding='max_length', return_tensors="tf")
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        }

# if __name__ == '__main__':
#     pass
#     test_sample = pd.read_csv("test_sample.csv")
#     X_test_sample = test_sample['review']
#     dp = DataPreprocessor()
#     a= dp.load_glove_embeddings("glove.6B.200d.txt", 200)
#     print(a)
#     print("!!!!!!!")
    # print(_)

