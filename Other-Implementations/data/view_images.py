import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import zoom

rcParams['font.family'] = 'Courier New'

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

def resize_image(image_data, new_shape):
    """
    Resize the image using interpolation.
    """
    zoom_factors = (new_shape[0] / image_data.shape[0], new_shape[1] / image_data.shape[1])
    resized_image = zoom(image_data, zoom_factors, order=1)  # order=1 for bilinear interpolation
    return resized_image

def display_multiple_images(file_paths):
    """
    Display multiple images in one window using subplots.
    """
    num_images = len(file_paths)
    cols = np.ceil(num_images/2).astype(int) 
    rows = (num_images + cols - 1) // cols 

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    if num_images == 1:
        axes = [axes]  # Wrap single Axes object in a list
    else:
        axes = axes.flatten()

    for i, file_path in enumerate(file_paths):
        image_data = load_image_from_txt(file_path)
        if image_data is not None:
            # Flip the image vertically unless "phantom" is in the file name
            if "phantom" not in file_path:
                image_data = np.flipud(image_data)
            if "sinogram" in file_path:
                image_data = resize_image(image_data, (512, 512))
            axes[i].imshow(image_data, cmap='magma', interpolation='nearest')
            axes[i].set_aspect('equal')

            # Set title based on file name
            if "phantom" not in file_path and "sinogram" not in file_path:
                iteration_count = file_path.split("_")[1]
                axes[i].set_title(
                    f"Iterations: {iteration_count}",
                    fontdict={'fontname': 'Courier New', 'fontsize': 12, 'weight': 'bold'}
                )
            else:
                title = file_path.split("_")[0].capitalize()
                axes[i].set_title(
                    f"{title}",
                    fontdict={'fontname': 'Courier New', 'fontsize': 12, 'weight': 'bold'}
                )
            axes[i].axis('off') 
        else:
            axes[i].set_title(f"Failed to load: {file_path}", fontdict={'fontname': 'Courier New', 'fontsize': 12})
            axes[i].axis('off')

            # Turn off any unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # List of .txt file paths
    file_paths = [
        "image_seq_1.txt",  
        "image_seq_10.txt",
        "image_seq_100.txt",
        "image_seq_500.txt",
        "sinogram_seq_4096.txt"
    ]

    # Display all images in one window
    display_multiple_images(file_paths)