import numpy as np
import matplotlib.pyplot as plt

def load_image_from_txt(file_path):
    """
    Load image data from a .txt file.
    Assumes the file contains a 2D grid of pixel values.
    """
    try:
        # Load the data as a NumPy array
        image_data = np.loadtxt(file_path)
        return image_data
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def display_multiple_images(file_paths):
    """
    Display multiple images in one window using subplots.
    """
    num_images = len(file_paths)
    cols = 3  # Number of columns in the grid
    rows = (num_images + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, file_path in enumerate(file_paths):
        image_data = load_image_from_txt(file_path)
        if image_data is not None:
            # Flip the image vertically unless it's "phantom.txt"
            if "phantom_256.txt" not in file_path:
                image_data = np.flipud(image_data)
            axes[i].imshow(image_data, cmap='magma', interpolation='nearest')
            axes[i].set_title(f"Image: {file_path}")
            axes[i].axis('off') 
        else:
            axes[i].set_title(f"Failed to load: {file_path}")
            axes[i].axis('off')

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # List of .txt file paths
    file_paths = [
        "metal_10.txt",
        "metal_100.txt",
        "metal_500.txt",
        "metal_1000.txt",
        "metal_2000.txt",
        "phantom_256.txt"
    ]

    # Display all images in one window
    display_multiple_images(file_paths)