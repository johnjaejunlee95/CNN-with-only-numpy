import os
from utils.data import MNISTLoader
from utils.model import CNN
import argparse
import pickle

def train(args, train_images_path, train_labels_path, model_path):

    # Load training data and model with hyperparmameters
    lr = args.lr # 0.001
    epochs=args.epochs # 10
    batch_size= args.batch_size # 256
    model = CNN(learning_rate=lr)
    train_loader = MNISTLoader(train_images_path, train_labels_path)
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # Iterate through batches
        for batch_images, batch_labels in train_loader.get_batches(batch_size):
            # Forward propagation
            output = model.forward(batch_images)
            
            # Backward propagation
            loss = model.backward(batch_images, batch_labels, output)
            epoch_loss += loss
            batch_count += 1
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}], Batch: {batch_count}/{len(train_loader)//batch_size}, Loss: {loss:.4f}", end='\r')
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{epochs}], Overall Loss: {epoch_loss/batch_count:.4f}")
    
    # Save checkpoint
    os.makedirs(f'{model_path}', exist_ok=True)
    with open(f'{model_path}/model_state_dict.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Training completed")

def main(args):
    # Paths to MNIST dataset
    train_images_path = os.path.join(args.data_path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(args.data_path, 'train-labels.idx1-ubyte')
    model_path = os.path.join(args.model_path)
    
    # Start training
    train(args, train_images_path, train_labels_path, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--data_path", type=str, default='./datasets', help="Path to MNIST dataset")
    parser.add_argument("--model_path", type=str, default="./model_ckpt", help="Path to save/load model")
    args = parser.parse_args()
    
    main(args)
