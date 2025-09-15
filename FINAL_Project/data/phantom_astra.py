# This script generates a Shepp-Logan phantom using the ASTRA toolbox
# and saves it to a text file.

import astra
import numpy as np

# File to save the phantom
output_filename = "phantom_256.txt"

# Define the volume geometry
image_size = 256
vol_geom = astra.create_vol_geom(image_size, image_size)

# Create Shepp-Logan phantom using ASTRA
phantom = astra.data2d.shepp_logan(vol_geom)
phantom = phantom[1]
phantom = np.array(phantom)

# Save phantom to text file
np.savetxt(output_filename, phantom, fmt="%.8f")

print(f"Shepp-Logan phantom saved to '{output_filename}'")