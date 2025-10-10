import tomophantom
from tomophantom import TomoP2D
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the 2D phantom library definitions
libpath = os.path.join(os.path.dirname(tomophantom.__file__), "phantomlib", "Phantom2DLibrary.dat")

# Generate a 2D phantom (model 11, size 256×256)
phantom = TomoP2D.Model(11, 256, libpath)

# Normalise the phantom to [0, 1]
phantom = (phantom - np.min(phantom)) / (np.max(phantom) - np.min(phantom))

# Save the phantom to a txt file
output_filename = os.path.join(os.path.dirname(__file__), "phantom_model11.txt")
np.savetxt(output_filename, phantom, fmt="%.8f")