# This script generates and saves a sparse projection matrix for the following geometry:
# - Image size: 256x256
# - Number of detectors: calculated to cover the diagonal of the image
# - Number of angles: 90 
# - Projection type: Parallel beam strip projector
# The matrix is saved in a binary format with metadata (number of rows, columns, and non-zero entries) 
# followed by the CSR representation arrays.

import os
import astra
import numpy as np
from scipy.sparse import csr_matrix

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the volume geometry
image_size = 256
vol_geom = astra.create_vol_geom(image_size, image_size)

# File to save the projection matrix - must be saved in metal-data directory
output_filename = os.path.join(script_dir, "projection_" + str(image_size) + ".bin")

# Define projection geometry
num_detectors = int(np.ceil(2 * np.sqrt(2) * image_size))
print(f"Number of detectors: {num_detectors}")

# Set number of angles and angle total degrees
angles = np.linspace(0, np.pi, 90, False)
proj_geom = astra.create_proj_geom(
    'parallel', 1.0, num_detectors, angles
)

# Create projector
projector_id = astra.create_projector('strip', proj_geom, vol_geom)
matrix_id = astra.projector.matrix(projector_id)
A_csr = astra.matrix.get(matrix_id)

# Print dimensions and number of non-zeros
print(f"Sparse matrix shape: {A_csr.shape}")
print(f"Number of non-zero entries: {A_csr.nnz}")


with open(output_filename, "wb") as f:
    # Get dimensions and nnz
    num_rows, num_cols = A_csr.shape
    num_non_zero = A_csr.nnz

    # Write header: numRows, numCols, nnz
    f.write(np.uint64(num_rows))
    f.write(np.uint64(num_cols))
    f.write(np.uint64(num_non_zero))

    # Write the CSR arrays
    f.write(A_csr.indptr.astype(np.int32).tobytes())
    f.write(A_csr.indices.astype(np.int32).tobytes())
    f.write(A_csr.data.astype(np.float32).tobytes())

print(f"Successfully saved CSR matrix to {output_filename}")