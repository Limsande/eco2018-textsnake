"""
Compute RGB channel-wise mean and std of given images.

Usage:
    img_mean_and_std.py IMG...
"""

import numpy as np
import os
import sys

from PIL import Image

if len(sys.argv[1:]) == 0:
    sys.exit(__doc__)

for f in sys.argv[1:]:
    if not os.path.isfile(f):
        sys.exit(f"Is not a file: {f}")

# 3 channels
c1 = []
c2 = []
c3 = []
for f in sys.argv[1:]:
    img = np.array(Image.open(f))
    print(f"Img {f}: shape: {img.shape}")
    c1.extend(img[:, :, 0].flatten())
    c2.extend(img[:, :, 1].flatten())
    c3.extend(img[:, :, 2].flatten())

print(f"Mean: {np.mean(c1)} {np.mean(c2)} {np.mean(c3)}")
print(f"Std: {np.std(c1)} {np.std(c2)} {np.std(c3)}")

