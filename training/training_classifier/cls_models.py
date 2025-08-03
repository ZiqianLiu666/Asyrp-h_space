import torch.nn as nn

class CustomLatentClassifier(nn.Module):
    def __init__(self, input_dim=18*512, num_layers=1, hidden_dim=None):
        """
        A flexible classifier for high-dimensional latent vectors.

        Args:
            input_dim (int): Dimension of input features (default: 18x512).
            num_layers (int): Number of hidden layers (1 = logistic regression).
            hidden_dim (int or None): If None, uses input_dim // 2.
        """
        super(CustomLatentClassifier, self).__init__()

        # Restrict num_layers between 1-3
        if num_layers < 1 or num_layers > 3:
            raise ValueError("num_layers must be 1 (logistic regression), 2, or 3.")

        # Automatically determine hidden dimensions if not set
        if hidden_dim is None:
            hidden_dim1 = input_dim // 2  # First layer (9216 → 4608)
            hidden_dim2 = hidden_dim1 // 2  # Second layer (4608 → 2048)
        else:
            hidden_dim1 = hidden_dim
            hidden_dim2 = hidden_dim // 2

        layers = []

        # If num_layers = 1, use logistic regression (no hidden layers)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        elif num_layers == 2:
            # One hidden layer: 9216 → 4608 → 1
            layers.append(nn.Linear(input_dim, hidden_dim1))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim1, 1))
        elif num_layers == 3:
            # Two hidden layers: 9216 → 4608 → 2048 → 1
            layers.append(nn.Linear(input_dim, hidden_dim1))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim1, hidden_dim2))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim2, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # Output is raw logits (for BCEWithLogitsLoss)


# # Logistic Regression (no hidden layers)
# model_1 = CustomLatentClassifier(input_dim=18*512, num_layers=1)
# print(model_1)

# # One hidden layer (9216 → 4608 → 1)
# model_2 = CustomLatentClassifier(input_dim=18*512, num_layers=2)
# print(model_2)

# # Two hidden layers (9216 → 4608 → 2048 → 1)
# model_3 = CustomLatentClassifier(input_dim=18*512, num_layers=3)
# print(model_3)

# class CustomLatentClassifier(nn.Module):
#     def __init__(self, input_dim=18*512, num_layers=1, hidden_dim=256):
#         """
#         A flexible classifier for high-dimensional latent vectors.

#         Args:
#             input_dim (int): Dimension of input features (default: 18x512).
#             num_layers (int): Number of hidden layers. If 0, it's logistic regression.
#             hidden_dim (int): Number of neurons per hidden layer.
#         """
#         super(CustomLatentClassifier, self).__init__()
#         layers = []

#         # If num_layers = 0, use logistic regression (no hidden layers)
#         if num_layers == 0:
#             layers.append(nn.Linear(input_dim, 1))
#         else:
#             # First hidden layer
#             layers.append(nn.Linear(input_dim, hidden_dim))
#             layers.append(nn.ReLU())

#             # Additional hidden layers
#             for _ in range(num_layers - 1):
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#                 layers.append(nn.ReLU())

#             # Output layer
#             layers.append(nn.Linear(hidden_dim, 1))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)  # Output is raw logits (for BCEWithLogitsLoss)
    


