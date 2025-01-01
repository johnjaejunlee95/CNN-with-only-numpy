import numpy as np

import numpy as np

class ConvLayer:
    def __init__(self, learning_rate, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        
        # Initialize weights and biases
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros((output_channels, 1))
        
        # Hyperparameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Cache for backpropagation
        self.cache = None
        self.learning_rate = learning_rate
    
    def _pad_input(self, inputs):

        batch_size, channels, height, width = inputs.shape
        padded_height = height + 2 * self.padding
        padded_width = width + 2 * self.padding
        
        padded_inputs = np.zeros((batch_size, channels, padded_height, padded_width))
        padded_inputs[:, :, 
            self.padding:padded_height-self.padding, 
            self.padding:padded_width-self.padding
        ] = inputs
        
        return padded_inputs
    
    def forward(self, inputs):

        batch_size, input_channels, input_height, input_width = inputs.shape
        
        # Pad inputs
        padded_inputs = self._pad_input(inputs)
        
        # Output dimensions
        output_height = (input_height + 2*self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Create view of input patches
        patches = np.lib.stride_tricks.as_strided(
            padded_inputs, 
            shape=(batch_size, input_channels, output_height, output_width, self.kernel_size, self.kernel_size),
            strides=(
                padded_inputs.strides[0], 
                padded_inputs.strides[1], 
                padded_inputs.strides[2] * self.stride, 
                padded_inputs.strides[3] * self.stride, 
                padded_inputs.strides[2], 
                padded_inputs.strides[3]
            )
        )
        
        # Reshape weights for broadcasting
        reshaped_weights = self.weights.reshape(self.weights.shape[0], -1)
        
        # Vectorized convolution
        # outputs = np.einsum('ijklmn,fo->ifkl', patches, reshaped_weights, optimize=True)
        outputs = np.matmul(patches.reshape(-1, self.kernel_size * self.kernel_size * input_channels), reshaped_weights.T).reshape(batch_size, self.weights.shape[0], output_height, output_width)
        outputs += self.biases.reshape(-1, 1, 1)
        
        # Cache for backpropagation
        self.cache = (inputs, padded_inputs)
        
        return outputs
    
    
    def backward(self, grad_output):
        inputs, padded_inputs = self.cache
        batch_size, input_channels, input_height, input_width = inputs.shape
        output_channels, _, kernel_size, _ = self.weights.shape
        output_height, output_width = grad_output.shape[2], grad_output.shape[3]

        # Initialize gradients
        grad_inputs = np.zeros_like(padded_inputs)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)

        # Compute gradient for biases
        grad_biases += np.sum(grad_output, axis=(0, 2, 3)).reshape(-1,1)

        # Unfold the input into patches (im2col-like approach)
        patches = np.lib.stride_tricks.as_strided(
            padded_inputs,
            shape=(
                batch_size,
                input_channels,
                kernel_size,
                kernel_size,
                output_height,
                output_width,
            ),
            strides=(
                padded_inputs.strides[0],
                padded_inputs.strides[1],
                padded_inputs.strides[2] * self.stride,
                padded_inputs.strides[3] * self.stride,
                padded_inputs.strides[2],
                padded_inputs.strides[3],
            ),
            writeable=False,
        )  # Shape: (batch_size, input_channels, kernel_size, kernel_size, output_height, output_width)

        # Reshape patches for vectorized computation
        patches = patches.reshape(batch_size, input_channels, kernel_size, kernel_size, -1)  # (B, C, K, K, OH*OW)
        grad_output = grad_output.reshape(batch_size, output_channels, -1)  # (B, F, OH*OW)


        patches_reshaped = patches.transpose(1, 2, 3, 0, 4).reshape(input_channels * kernel_size * kernel_size, -1)  # Shape: (C * K * K, B * OH * OW)
        grad_output_reshaped = grad_output.transpose(1, 0, 2).reshape( output_channels, -1)  # Shape: (F, B * OH * OW)
        grad_weights = grad_output_reshaped @ patches_reshaped.T  # Shape: (F, C * K * K)
        grad_weights = grad_weights.reshape(output_channels, input_channels, kernel_size, kernel_size)  # Shape: (F, C, K, K)

        # Compute grad_input_patch using matmul
        weights_reshaped = self.weights.reshape(output_channels, -1)  # Shape: (F, C * K * K)
        grad_input_patch = weights_reshaped.T @ grad_output_reshaped  # Shape: (C * K * K, B * OH * OW)
        grad_input_patch = grad_input_patch.reshape(input_channels, kernel_size, kernel_size, batch_size, output_height, output_width).transpose(3, 0, 1, 2, 4, 5)  # Shape: (B, C, K, K, OH, OW)


        # Accumulate gradients back to padded inputs --> not parallelizable
        for h in range(kernel_size):
            for w in range(kernel_size):
                grad_inputs[:, :, h:h + output_height * self.stride:self.stride, w:w + output_width * self.stride] += grad_input_patch[:, :, h, w, :, :]

        # Remove padding from grad_inputs
        if self.padding > 0:
            grad_inputs = grad_inputs[:, :, self.padding:-self.padding, self.padding:-self.padding]

        # Update parameters with gradient clipping

        grad_weights = np.clip(grad_weights, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases

        return grad_inputs

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        """
        Max Pooling Layer
        
        Parameters:
        - pool_size: Size of the pooling window
        - stride: Stride of the pooling operation
        """
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
    
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.shape
        
        # Output dimensions
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        # Create windows
        windows = np.lib.stride_tricks.as_strided(
            inputs,
            shape=(batch_size, channels, output_height, output_width, self.pool_size, self.pool_size),
            strides=(
                inputs.strides[0], 
                inputs.strides[1], 
                inputs.strides[2] * self.stride, 
                inputs.strides[3] * self.stride, 
                inputs.strides[2], 
                inputs.strides[3]
            )
        )
        
        # Max pooling
        outputs = windows.max(axis=(4, 5))
        
        # Create mask of max values for backprop
        max_mask = windows == outputs[:, :, :, :, np.newaxis, np.newaxis]
        
        # Cache for backpropagation
        self.cache = (inputs, max_mask)
        
        return outputs
    
    def backward(self, grad_output):
        inputs, max_mask = self.cache
        batch_size, channels, height, width = inputs.shape
        
        # Create output gradient shape with pooling window dimensions
        grad_windows = np.zeros_like(max_mask, dtype=float)
        
        # Broadcast gradient through max mask
        grad_windows[max_mask] = np.repeat(grad_output.ravel(), max_mask.sum(axis=(4,5)).ravel())
        grad_inputs = grad_windows.reshape(batch_size, channels, height, width)
        
        return grad_inputs
    
   
    

class FullyConnectedLayer:
    def __init__(self, learning_rate, input_size, output_size):

        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        
        # Cache for backpropagation
        self.cache = None
        self.learning_rate = learning_rate
    
    def forward(self, inputs):

        batch_size = inputs.shape[0]
        
        # Flatten input if needed
        if len(inputs.shape) > 2:
            inputs = inputs.reshape(batch_size, -1)
        
        # Linear transformation
        outputs = np.dot(inputs, self.weights.T) + self.biases.T
        
        # Cache for backpropagation
        self.cache = (inputs, outputs)
        
        return outputs
    
    def backward(self, grad_output):

        inputs, _ = self.cache
        batch_size = inputs.shape[0]
        
        # Flatten input if needed
        if len(inputs.shape) > 2:
            inputs = inputs.reshape(batch_size, -1)
        
        # Compute gradients
        grad_inputs = np.dot(grad_output, self.weights)
        grad_weights = np.dot(grad_output.T, inputs)
        grad_biases = np.sum(grad_output, axis=0).reshape(-1, 1)
        
        # Update parameters with gradient clipping
        grad_weights = np.clip(grad_weights, -1, 1)
        grad_biases = np.clip(grad_biases, -1, 1)
        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases
        
        return grad_inputs

def relu(x):

    return np.maximum(0, x)

def relu_derivative(x):
 
    return (x > 0).astype(float)

def softmax(x):

    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class CNN:
    def __init__(self, learning_rate=0.01):
        """
        Convolutional Neural Network Architecture
        """
        # Layers
        self.conv1 = ConvLayer(learning_rate, input_channels=1, output_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.conv2 = ConvLayer(learning_rate, input_channels=32, output_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.classifier = FullyConnectedLayer(learning_rate, 32 * 7 * 7, 10)
    
    def forward(self, x):
       
        # 2 Convolution and pooling layers
        x = self.conv1.forward(x)
        x = relu(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = relu(x)
        x = self.pool2.forward(x)
        
        # Fully connected layers
        x = self.classifier.forward(x)
        
        return softmax(x)
    
    def backward(self, x, y, output):
       
        # Compute loss (cross-entropy)
        loss = -np.sum(y * np.log(output + 1e-15)) / x.shape[0]
        
        # Compute initial gradient
        grad = output - y
        
        # Backward through fully connected layers
        grad = self.classifier.backward(grad).astype(float)
        
        # Backward through pooling and convolution layers
        grad = grad.reshape((-1, 32, 7, 7))
        grad = self.pool2.backward(grad).astype(float)
        grad = self.conv2.backward(grad * relu_derivative(self.conv2.cache[0])).astype(float)
        grad = self.pool1.backward(grad).astype(float)
        self.conv1.backward(grad * relu_derivative(self.conv1.cache[0]))
        
        return loss
