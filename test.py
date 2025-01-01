import os
import numpy as np
from utils.data import MNISTLoader
import argparse
import pickle

def test(test_images_path, test_labels_path, model_path):
    """
    Test the Convolutional Neural Network
    
    Parameters:
    - test_images_path: Path to test images
    - test_labels_path: Path to test labels
    
    Returns:
    Test accuracy
    """
    # Load test data
    test_loader = MNISTLoader(test_images_path, test_labels_path)
    
    # Load model
    with open(f'{model_path}/model_state_dict.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Prediction and accuracy tracking
    correct_predictions = 0
    total_samples = len(test_loader.labels)
    
    # Batch processing for memory efficiency
    batch_size = 64
    for batch_images, batch_labels in test_loader.get_batches(batch_size, shuffle=False):
        # Forward propagation
        output = model.forward(batch_images)
        
        # Compute predictions
        predictions = np.argmax(output, axis=1)
        true_labels = np.argmax(batch_labels, axis=1)
        
        # Count correct predictions
        correct_predictions += np.sum(predictions == true_labels)
        
        # Print progress
        print(f"Testing progress: {len(predictions)}/{total_samples}", end='\r')
    
    # Compute accuracy
    accuracy = correct_predictions / total_samples
    
    # Print results
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    # Paths to MNIST dataset
    test_images_path = os.path.join(args.data_path, 'train-images.idx3-ubyte')
    test_labels_path = os.path.join(args.data_path, 'train-labels.idx1-ubyte')
    model_path = os.path.join(args.model_path)
    
    # Start testing
    test(test_images_path, test_labels_path, model_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./datasets', help="Path to MNIST dataset")
    parser.add_argument("--model_path", type=str, default="./model_ckpt", help="Path to save/load model")
    args = parser.parse_args()
    
    main(args)
    main()
