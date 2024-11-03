import numpy as np

# Load the saved .npy files
train_states = np.load('final_stock_and_macro_data.npy')


# Print the shape of the loaded arrays to check the dimensions
print(f"Training set shape: {train_states.shape}")


# Display a sample from the training set (e.g., first rolling window state)
print(f"First sample from training set:\n{train_states[0]}")

# Display the first 5 samples for a broader view (optional)
print(f"First 5 samples from training set:\n{train_states[:5]}")