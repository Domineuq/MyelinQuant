# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:29:40 2023
@author: Dominique Neuhaus


Quantification of myelin density based on histology scans

# Description of main steps
1. Grayscale and Otsu's threshold: binary mask -> tissue: 1, background: 0
2. Splitting original image and binary mask into blocks (e.g., 35x35 pixels)
3. If the block contains enough tissue (more than half of the pixels), it is processed further:
    - The block is converted to the HSV colour space
    - Colour thresholds are calculated to identify stained areas
    - A mask is created to isolate these areas
    - The masked area is converted to grayscale and blurred
    - Otsu's thresholding method is applied to create a binary image where myelin areas are identified
4. The fraction of the block area covered by myelin is calculated

"""


# %% Packages
import cv2
from skimage import filters, morphology
import numpy as np
from matplotlib import pyplot as plt


# %% Define output and input
# Specify the output directory and filename for the area fraction map
directory = r'path/out/dir'
filename = 'filename'

# Load the image
image_loc = r"path/to/input/image"
image = cv2.imread(image_loc)



# %% Processing

# Create a tissue mask ------------------------------------------------
# Convert the image to grayscale
image_gray_4mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding method to create a binary mask
ret, binary_mask = cv2.threshold(image_gray_4mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# --> Values are either 0 or 255 (not 0 and 1)


# Invert the mask (-> background = 0)
# --> Pixels values > 0 shall be used for evaluation
binary_mask_inverted = cv2.bitwise_not(binary_mask)




# Split in block and process: myelin area fraction ---------------------

# Define the block size
block_size = 35

# Get the image size
height, width, _ = image.shape

# Calculate new dimensions that are multiples of block_size
new_height = height - (height % block_size)
new_width = width - (width % block_size)

# Crop the image to the new dimensions
image_cropped = image[:new_height, :new_width]

# Crop the tissue mask to the new dimensions
tissue_mask_cropped = binary_mask_inverted[:new_height, :new_width]

# Initialize an empty array to store the myelin area fractions
myelin_fractions = np.zeros((new_height // block_size, new_width // block_size))


# Process each block of the image
for i in range(0, new_height, block_size):
    for j in range(0, new_width, block_size):
        # Extract the block from the image
        block = image_cropped[i:i+block_size, j:j+block_size]
        
        # Extract the corresponding block from the mask
        block_mask = tissue_mask_cropped[i:i+block_size, j:j+block_size]
        
        # If the majority of pixels in the block are not tissue, skip the processing
        if np.mean(block_mask) < (160): # 255/2 would be 1/2 of block as tissue
            myelin_fractions[i // block_size, j // block_size] = 0
            continue

        # Process the block to get myelin area fraction
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        low_color = np.percentile(hsv, 5, axis=(0,1))
        high_color = np.percentile(hsv, 95, axis=(0,1))
        lower_color = np.array([low_color[0], low_color[1], low_color[2]])
        upper_color = np.array([high_color[0], high_color[1], high_color[2]])
        mask_stained = cv2.inRange(hsv, lower_color, upper_color)
        image_stained = cv2.bitwise_and(block, block, mask=mask_stained)
        image_gray = cv2.cvtColor(image_stained, cv2.COLOR_BGR2GRAY)
        image_gr_blr = cv2.GaussianBlur(image_gray, (5, 5), 0)
        threshold_value = filters.threshold_otsu(image_gr_blr)
        binary_myelin = image_gr_blr < threshold_value
        stained_area_fraction = np.mean(binary_myelin)  # myelin = 1 are summed up and divided by all (1 and 0)


        # Store the myelin area fraction in the corresponding position in the array
        myelin_fractions[i // block_size, j // block_size] = stained_area_fraction


# count the tissue pixels (where there is a myelin area fraction)
tissue_pixels = np.sum(myelin_fractions > 0)
# sum up all (tissue) pixel values (outside tissue, it is 0)
sum_myelin_frac_vals = np.sum(myelin_fractions)
# calculate overall myelin area fraction
overall_myelin_AF = sum_myelin_frac_vals/tissue_pixels
print('Overall Myelin Area Fraction: ', overall_myelin_AF)

std_dev = np.std(myelin_fractions)
print('Standard Deviation of Myelin Area Fractions: ', std_dev)




# %% Create stained area fraction map

# Set size of figure
plt.figure(figsize=(6, 4))  # width, height

# Display the myelin area fractions
plt.imshow(myelin_fractions, cmap='viridis', vmin=0, vmax=1)

# Add a colourbar
cbar = plt.colorbar()

# Set label
cbar.set_label('Stained Area Fraction [a.u.]', fontsize=16)

# Remove x and y ticks
plt.xticks([])
plt.yticks([])

# Increase the font size of the ticks of the color bar
tick_font_size = 14 
cbar.ax.tick_params(labelsize=tick_font_size)

# Adjust the layout
plt.tight_layout()


# Save the figure as an SVG
plt.savefig(directory + filename + '.svg', format='svg')

# Save the figure as a PDF
plt.savefig(directory + filename + '.pdf', format='pdf')


# Show the plot
plt.show()

