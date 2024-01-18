import os
import torch
from A.train import train as train_A
from A.evaluate import evaluate as evaluate_A
from B.train import train_model, NiN  # Import NiN model class
from B.evaluate import evaluate_model

if __name__ == "__main__":
    # For model A
    model_a_path = './A/model.pth'
    if not os.path.exists(model_a_path):
        train_A()
    evaluate_A()

    # For model B
    model_b_path = './B/pathmnist_model.pth'
    data_path = "./Datasets/pathmnist.npz"
    num_classes = 9  # Adjust based on your dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_b_path):
        model_b = train_model(data_path, num_classes, epochs=30, batch_size=64, lr=0.001)
        torch.save(model_b.state_dict(), model_b_path)
    else:
        # Load the existing model with appropriate map_location
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        model_b = NiN(num_classes)
        model_b.load_state_dict(torch.load(model_b_path, map_location=map_location))
        model_b.to(device)
    
    evaluate_model(model_b, data_path, batch_size=32)
