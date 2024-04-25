import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns

class Visualizer:
    def __init__(self):
        pass

    def visualize_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(set(y_test)))
        plt.xticks(tick_marks, sorted(set(y_test)), rotation=45)
        plt.yticks(tick_marks, sorted(set(y_test)))
        plt.tight_layout()
        plt.show()

    def visualize_classification_report(self, y_test, y_pred):
        report = classification_report(y_test, y_pred)
        print(report)

    def visualize_accuracy_score(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

# Usage
visualizer = Visualizer()
y_test = [0, 1, 1, 0]  # Actual test labels
y_pred = [0, 1, 0, 1]  # Predicted test labels

visualizer.visualize_confusion_matrix(y_test, y_pred)
visualizer.visualize_classification_report(y_test, y_pred)
visualizer.visualize_accuracy_score(y_test, y_pred)
