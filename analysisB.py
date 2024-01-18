import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from B.utils import load_data
from B.train import train_model, NiN  # Import NiN model class

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):  
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def analyze_model(model, data_path, batch_size=32):
    _, _, test_loader = load_data(data_path, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Class 1', 'Class 2', 'Class 3', ...] 
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, classes=class_names, normalize=True)
    plt.show()

    # Classification Report
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Example usage
model_b_path = './B/pathmnist_model.pth'
data_path = "./Datasets/pathmnist.npz"
model_b = NiN(9)
model_b.load_state_dict(torch.load(model_b_path, map_location=torch.device('cpu')))
analyze_model(model_b, data_path)
