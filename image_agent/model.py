# https://docs.python.org/3/reference/index.html
import torch
import torch.nn.functional as F

# Define a function to compute the soft-argmax of a heatmap.
def spatial_argmax(logit):
    # Apply softmax to flatten the input into a 1D probability distribution, then reshape it back to the original shape.
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    # Calculate the weighted sum of the indices along the width (horizontal direction), representing the x-coordinates.
    x_coords = (weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1)
    # Calculate the weighted sum of the indices along the height (vertical direction), representing the y-coordinates.
    y_coords = (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)
    # Return a stack of the x and y coordinates, providing a soft estimate of the position.
    return torch.stack((x_coords, y_coords), 1)

# Define the main model for detecting the puck position.
class PuckDetector(torch.nn.Module):
    # Nested Block class defining the individual layers of the network
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            # Convolutional layer with specified number of inputs and outputs, kernel size, and stride
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            # Batch normalization layers to normalize the outputs of the convolutional layers
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            # Skip connection to add input directly to the output for training stability
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        # Forward method defines how the data flows through the block
        def forward(self, x):
            # Apply convolution, batch normalization, and ReLU activation sequentially
            x = F.relu(self.b1(self.c1(x)))
            x = F.relu(self.b2(self.c2(x)))
            x = F.relu(self.b3(self.c3(x)))
            # Add the result of the skip connection to the processed output
            return x + self.skip(x)

    # Nested UpBlock class for upsampling the feature maps
    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            # Transposed convolution for upsampling
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride, output_padding=1)

        def forward(self, x):
            # Apply ReLU activation after the transposed convolution
            return F.relu(self.c1(x))

    # Initialize the PuckDetector model
    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=1, kernel_size=3, use_skip=True):
        super().__init__()
        # Mean and standard deviation for normalizing input images
        self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
        self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

        c = 3  # Initial channel size
        self.use_skip = use_skip  # Flag to determine if skip connections are used
        self.n_conv = len(layers)  # Number of convolutional layers
        skip_layer_size = [3] + layers[:-1]  # Define the size of each skip layer
        
        # Create convolutional and upsampling layers dynamically based on the 'layers' list
        for i, l in enumerate(layers):
            setattr(self, f'conv{i}', self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            setattr(self, f'upconv{i}', self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        # Final classifier convolution that outputs the heatmap
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

    # Define how the input image is processed through the network to produce the puck location
    def forward(self, img):
        # Normalize the input image
        img = (img - self.input_mean[None, :, None, None].to(img.device)) / self.input_std[None, :, None, None].to(img.device)
        activations = []
        # Pass the image through each convolutional and upsampling layer
        for i in range(self.n_conv):
            activations.append(img)
            img = getattr(self, f'conv{i}')(img)
        for i in reversed(range(self.n_conv)):
            img = getattr(self, f'upconv{i}')(img)
            img = img[:, :, :activations[i].size(2), :activations[i].size(3)]
            if self.use_skip:
                img = torch.cat([img, activations[i]], dim=1)
        # Apply the classifier and compute the spatial argmax to find the puck's location
        return spatial_argmax(self.classifier(img).squeeze(1))

def save_model(model):
    torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'MSE4.th'))

def load_model():
    model = PuckDetector()
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'MSE4.th'), map_location='cpu'))
    return model
