import os
import shutil
import numpy as np
import random
import glob
import matplotlib.pyplot as plt
from PIL import Image

# Delete all files in a directory (for cleanup)
def delete_files(folder):
    """Removes all files and subdirectories inside a folder."""
    if os.path.exists(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # Delete file or symlink
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Delete subfolder
            except Exception as e:
                print(f"Couldn't remove {item_path}: {e}")

# Convert images to PBM and return dataset
def process_images(input_folder, output_folder, size=(16, 16)):
    """Converts images (PNG/BMP) into PBM format and loads them into an array."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = []
    print("Processing images...")  # Debug message

    for filepath in glob.glob(os.path.join(input_folder, "*")):
        if filepath.endswith(".png") or filepath.endswith(".bmp"):
            try:
                img = Image.open(filepath).convert("L")  # Convert to grayscale
                img = img.resize(size)

                binary_data = []
                for pixel in np.array(img).flatten():
                    binary_data.append(1 if pixel < 128 else -1)  # Convert to -1,1

                images.append(binary_data)

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print(f"Loaded {len(images)} images.")  # Debug message
    return np.array(images)

# Hopfield Network Class
class HopfieldNetwork:
    def __init__(self, size, mode="async"):
        self.size = size
        self.weights = np.zeros((size, size), dtype=float)
        self.mode = mode  # "sync" or "async"

    def train(self, patterns):
        """Train using Hebbian learning."""
        patterns_matrix = np.array(patterns)
        self.weights = np.dot(patterns_matrix.T, patterns_matrix)  # Hebbian learning
        np.fill_diagonal(self.weights, 0)  # No self-connections

    def update(self, state):
        """Updates neurons based on selected mode."""
        if self.mode == "sync":
            return np.sign(np.dot(self.weights, state))
        else:  # Asynchronous update
            new_state = state.copy()
            for i in random.sample(range(self.size), self.size):  # Shuffle indices manually
                new_state[i] = 1 if np.dot(self.weights[i], new_state) >= 0 else -1
            return new_state

# Introduce noise by flipping pixels
def add_noise(state, p):
    """Randomly flips some pixels."""
    noisy = state.copy()
    for i in range(len(state)):
        if random.random() < p:
            noisy[i] *= -1  # Flip pixel
    return noisy

# Introduce corruption with a 10x10 preserved square
def add_border(state, size=(16,16)):
    """Sets all pixels to black except for a 10x10 central region."""
    corrupted = np.full(len(state), -1)  # Start with black image
    start_x, start_y = 3, 3  # Position of the preserved square
    for i in range(start_y, start_y + 10):
        for j in range(start_x, start_x + 10):
            corrupted[i * size[0] + j] = state[i * size[0] + j]
    return corrupted

# Run the Hopfield network until it stabilizes
def run_network(initial, net, max_steps=40):
    """Runs updates until the network stabilizes."""
    state = initial.copy()
    for step in range(max_steps):
        new_state = net.update(state)
        if np.array_equal(new_state, state):
            return new_state, step + 1  # Converged
        state = new_state
    return state, max_steps

# Display images for comparison
def show_images(original, noisy, recovered):
    """Displays original, noisy, and recovered images."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [original, noisy, recovered]
    titles = ["Original", "Noisy", "Recovered"]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.reshape(16, 16), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    
    plt.show()

# Plotting results
def plot_results(results, noise_levels):
    accuracy = {p: (results[p]["correct"]/20)* 100 for p in noise_levels}
    avg_steps = {p: (sum(results[p]["steps"]) / len(results[p]["steps"]) if results[p]["steps"] else 0) for p in noise_levels}
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(noise_levels, [accuracy[p] for p in noise_levels], color='g', alpha=1, label='Accuracy (%)', width=0.01)
    ax2.plot(noise_levels, [avg_steps[p] for p in noise_levels], 'blue', label='Avg Steps')
    
    ax1.set_xlabel('Corruption Probability (p)')
    ax1.set_ylabel('Accuracy (%)', color='g')
    ax2.set_ylabel('Average Steps to Convergence', color='b')
    plt.title('Hopfield Network Performance')
    
    fig.legend(loc='upper right')
    plt.show()

def plothistogram(results, noise_levels):
    for p in noise_levels:
        if results[p]["steps"]:
            plt.figure()
            plt.hist(results[p]["steps"], bins=10, alpha=0.5, color='blue', edgecolor='black')
            plt.xlabel('Update Steps')
            plt.ylabel('Frequency')
            plt.title(f'Convergence Steps Histogram for p={p}')
            plt.show()
            
            
# Main program execution
if __name__ == "__main__":
    delete_files("pbm_images")

    dataset = process_images("png_bmp_images", "pbm_images")
    if len(dataset) == 0:
        raise ValueError("No PBM images found!")

    net = HopfieldNetwork(dataset.shape[1], mode="sync")
    net.train(dataset)

    noise_levels = np.linspace(0.2,0.9,8)
    results = {p: {"correct": 0, "steps": []} for p in noise_levels}

    corruption_mode = ("flip") #input either flip or border for two corruption modes
    

    for p in noise_levels:
        for _ in range(10000):
            original = random.choice(dataset)
            corrupted = add_noise(original, p) if corruption_mode == "flip" else add_border(original)
            recovered, steps = run_network(corrupted, net)
            if np.array_equal(recovered, original):
                results[p]["correct"] += 1
            results[p]["steps"].append(steps)
            
            #comment out the code below to prevent seperate histograms from coming up
            #plothistogram(results, noise_levels) 
            
            #comment out the code below to prevent debug images from popping up
            #show_images(original, corrupted, recovered) 
    plothistogram(results, noise_levels)
    #plot_results(results, noise_levels)
    print("Experiment complete!")