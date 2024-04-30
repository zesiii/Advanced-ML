from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def display_classification_report(y_true, y_pred):
    """
    Generates and prints a classification report for the predictions.

    This function takes the true labels and the predicted labels of a classification model
    and prints out a report showing the main classification metrics such as precision, recall,
    and f1-score for each class.

    Parameters:
    - y_true (list): The true labels of the data.
    - y_pred (list): The predicted labels as returned by a classifier.

    Returns:
    None, but prints the classification report in the front end notebook.
    """
    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a confusion matrix for the predictions using seaborn's heatmap.

    This function takes the true labels and the predicted labels and computes a confusion matrix,
    which it then plots using a heatmap. This visualization helps in understanding the classification
    performance of a model across different classes.

    Parameters:
    - y_true (list): The true labels of the data.
    - y_pred (list): The predicted labels as returned by a classifier.

    Returns:
    None, but displays the confusion matrix as a heatmap in a front end notebook.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
