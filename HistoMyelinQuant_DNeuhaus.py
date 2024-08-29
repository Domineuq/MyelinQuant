# -*- coding: utf-8 -*-
"""
------------
MyelinQuant
------------

Created on Tue Nov  7 15:29:40 2023
@author: Dominique Neuhaus


Quantification of myelin density based on histology scans

# Description of main steps
1. Grayscale and Otsu's threshold: binary mask -> tissue: 1, background: 0
2. Splitting original image and binary mask into blocks (e.g., 35x35 pixels)
3. If the block contains enough tissue (more than half of the pixels), it is processed further:
    - The block is converted to the HSV colour space
    - Colour thresholds are calculated to set myelin and background to 0
    - A mask is created containing the remaining tissue
    - Conversion to grayscale and bluring
    - Otsu's thresholding method: dark areas set to 1: myelin and background
    - Multiplication with tissue mask to remove background
    - ==> image with myelin = 1, background and non-myelin tissue = 0
4. The fraction of the block area covered by myelin is calculated
5. Calculation of overall myelin area fraction / myelin density

"""


# %% Packages
import cv2
from skimage import filters, morphology
import numpy as np
from matplotlib import pyplot as plt


# %% Definitions
# Specify the output directory for the images
directory = r'path/to/output/directory/'

# Set file name for myelin area fraction map
filename = 'filename'

# Load the input image
image_loc = r"path/to/input_file.png"
image = cv2.imread(image_loc)


# Create a tissue mask ----------------------------------
# Convert the image to grayscale
image_gray_4mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding method to create a binary mask
ret, binary_mask = cv2.threshold(image_gray_4mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# --> Values are either 0 or 255 (not 0 and 1)


# Invert the mask (-> to set background = 0)
binary_mask_inverted = cv2.bitwise_not(binary_mask)

#--------------------------------------------------------

# Define the block size (can be adapted, but 35 works well)
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


# -------------------------------------------------------------------------------
# Initialize empty arrays to store the myelin area fractions and intermediate images
myelin_fractions = np.zeros((new_height // block_size, new_width // block_size))

# Activate below for intermediate images
block_image = np.zeros((new_height // block_size, new_width // block_size))
block_mask_image = np.zeros((new_height // block_size, new_width // block_size))
hsv_image = np.zeros((new_height // block_size, new_width // block_size, 3), dtype=np.uint8)
mask_stained_image = np.zeros((new_height // block_size, new_width // block_size))
image_stained_image = np.zeros((new_height // block_size, new_width // block_size))
image_gray_image = np.zeros((new_height // block_size, new_width // block_size))
image_grblr_image = np.zeros((new_height // block_size, new_width // block_size))
binary_myelin_image = np.zeros((new_height // block_size, new_width // block_size))


# %% Process each block of the image
for i in range(0, new_height, block_size):
    for j in range(0, new_width, block_size):
        # Extract the block from the image
        block = image_cropped[i:i+block_size, j:j+block_size]
        
        # Extract the corresponding block from the mask
        block_mask = tissue_mask_cropped[i:i+block_size, j:j+block_size]
        
        # If the majority of pixels in the block are not tissue, skip the processing
        if np.mean(block_mask) < (160): # 255/2 would be 1/2 of block as tissue. 160 works well
            myelin_fractions[i // block_size, j // block_size] = 0
            continue

        # Process the block to get myelin area fraction--------------------------
        # Convert to HSV colour space and set thresholds
        # Set pixels out of HSV range to 0 
        # Very dark and very dark pixels, respectively 
        # (background: initially white, and myelin: initially dark blue)
        hsv = cv2.cvtColor(block, cv2.COLOR_BGR2HSV)
        low_color = np.percentile(hsv, 2, axis=(0,1))
        high_color = np.percentile(hsv, 95, axis=(0,1))
        lower_color = np.array([low_color[0], low_color[1], low_color[2]])
        upper_color = np.array([high_color[0], high_color[1], high_color[2]])
        mask_stained = cv2.inRange(hsv, lower_color, upper_color)
        image_stained = cv2.bitwise_and(block, block, mask=mask_stained)
        
        # Convert to grayscale image and add blur for Otsu's thresholding method
        image_gray = cv2.cvtColor(image_stained, cv2.COLOR_BGR2GRAY)
        image_gr_blr = cv2.GaussianBlur(image_gray, (5, 5), 0)
        threshold_value = filters.threshold_otsu(image_gr_blr)
        
        # Set dark pixels to 1 (myelin)
        # Privious myelin and background are also set to 1 (were before set to 0)
        binary_myelin = image_gr_blr < threshold_value
        
        # binary_myelin has now myelin and background set as 1 and tissue w/o myelin set to 0
        # multiply with tissue mask to keep myelin and set background to 0
        tissue_mask=block_mask/255  # is in range [0,255], needs to be [0,1]
        binary_myelin = binary_myelin*tissue_mask
        binary_myelin_uint8 = (binary_myelin.astype(np.uint8)) * 255    # Only needed for visualisation
        stained_area_fraction = np.mean(binary_myelin)  # myelin = 1 is added up and devided by nr. of all pixels
        
       

        # Store the resulting block mean value at the corresponding position in the array
        myelin_fractions[i // block_size, j // block_size] = stained_area_fraction
        
        # Intermediate images, activate for debugging, etc.
        block_image[i // block_size, j // block_size] = np.mean(block)
        block_mask_image[i // block_size, j // block_size] = np.mean(block_mask)
        hsv_image[i // block_size, j // block_size] = np.mean(hsv, axis=(0,1))
        mask_stained_image[i // block_size, j // block_size] =np.mean(mask_stained)
        image_stained_image[i // block_size, j // block_size] = np.mean(image_stained)
        image_gray_image[i // block_size, j // block_size] = np.mean(image_gray)
        image_grblr_image[i // block_size, j // block_size] = np.mean(image_gr_blr)
        binary_myelin_image[i // block_size, j // block_size] = np.mean(binary_myelin_uint8)

# Save the myelin area fractions as an image
path = directory + filename + 'myelin_fractions.png'
cv2.imwrite(path, myelin_fractions * 255)


# Activate to save intermediate images
path = directory + 'block_image.png'
cv2.imwrite(path, block_image)

path = directory + 'block_mask.png'
cv2.imwrite(path, block_mask_image)

path = directory + 'hsv.png'
cv2.imwrite(path, hsv_image)

path = directory + 'mask_stained.png'
cv2.imwrite(path, mask_stained_image)

path = directory + 'image_stained.png'
cv2.imwrite(path, image_stained_image)

path = directory + 'image_gray.png'
cv2.imwrite(path, image_gray_image)

path = directory + 'image_grblr.png'
cv2.imwrite(path, image_grblr_image)

path = directory + 'binary_myelin.png'
cv2.imwrite(path, binary_myelin_image)





# %% Calculate overall myelin fraction
# count the pixels containing tissue
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
plt.figure(figsize=(6, 4))  # width, hight

# Display the myelin area fractions
plt.imshow(myelin_fractions, cmap='viridis', vmin=0, vmax=1)

# Add a colorbar
cbar = plt.colorbar()

# Set label
cbar.set_label('Stained Area Fraction [a.u.]', fontsize=16)

# Remove x and y ticks
plt.xticks([])
plt.yticks([])

# Increase the font size of the ticks of the color bar
tick_font_size = 14 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=tick_font_size)

# Adjust the layout
plt.tight_layout()


# Save the figure as an SVG
plt.savefig(directory + filename + '.svg', format='svg')

# Save the figure as a PDF
plt.savefig(directory + filename + '.pdf', format='pdf')


# Show the plot
plt.show()



