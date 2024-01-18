import torch
from A.model import BinaryClassifierCNN
from A.utils import load_npz_data
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model_path='A/model.pth', npz_file_path='Datasets/pneumoniamnist.npz'):
    _, _, test_loader = load_npz_data(npz_file_path)

    model = BinaryClassifierCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded for evaluation.")

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(conf_matrix)

    # Generate a classification report
    class_report = classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia'])
    print('Classification Report:')
    print(class_report)

if __name__ == "__main__":
    evaluate()
