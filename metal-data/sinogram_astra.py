# Generate sinogram for normalisation timing tests
# This script generates a sinogram from a Shepp-Logan phantom using the ASTRA toolbox
# and saves it to a .txt file.

import os
import astra
import numpy as np
from scipy.sparse import csr_matrix

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the volume geometry
image_size = 4096
vol_geom = astra.create_vol_geom(image_size, image_size)

# Define projection geometry
# Number of detectors is based on image size
num_detectors = int(np.ceil(2 * np.sqrt(2) * image_size))

# Print number of detectors
print(f"Number of detectors: {num_detectors}")

# Set number of angles and angle total degrees
angles = np.linspace(0, np.pi, 720, False)
num_angles = len(angles)
proj_geom = astra.create_proj_geom(
    'parallel', 1.0, num_detectors, angles
)

# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)

phantom = astra.data2d.shepp_logan(vol_geom)
phantom = phantom[1]
phantom = np.array(phantom)
id, sinogram = astra.create_sino(phantom, projector_id)
id = astra.create_sino(phantom, projector_id, returnData=True)

# Save the sinogram to a .txt file
output_filename = os.path.join(script_dir, f"sinogram_{image_size}_{num_angles}_test.txt")
np.savetxt(output_filename, sinogram, fmt='%.6f')

# Print confirmation
print(f"Sinogram saved to {output_filename}")