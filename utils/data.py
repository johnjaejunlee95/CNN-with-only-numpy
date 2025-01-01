import numpy as np
import struct

class MNISTLoader:
    def __init__(self, images_path, labels_path):

        # Load images
        with open(images_path, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            self.images = images.reshape(num_images, 1, rows, cols).astype(np.float32) / 255.0
        
        # Load labels
        with open(labels_path, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        
        # One-hot encode labels
        self.one_hot_labels = self.to_one_hot(self.labels)
    
    def to_one_hot(self, labels, num_classes=10):

        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot
    

    def get_batches(self, batch_size, shuffle=True):

        # Create indices
        indices = np.arange(len(self.images))
        
        # Shuffle if specified
        if shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield (
                self.images[batch_indices],
                self.one_hot_labels[batch_indices]
            )

    def __len__(self):
        return len(self.images)