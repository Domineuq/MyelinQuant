# MyelinQuant
![MyelinQuant_Logo_Final](https://github.com/user-attachments/assets/676509fb-74ac-44b7-944e-7b1deedf57a1)




### The Myelin Quantification Tool

MyelinQuant is a tool for the quantification of myelin density in a digitised histological sample. By automatically adapting the segmentation threshold, this tool is robust to varying staining intensities and tears. No manual interaction is required. Tested with Holmes-Luxol stained histochemical samples.

Created by the [Forensic Medicine and Imaging Research Group](https://dbe.unibas.ch/en/research/imaging-modelling-diagnosis/forensic-medicine-imaging-research-group/).
If you use it, please cite our publication: 
tbd

# Requirements
+ cv2
+ skimage (filters, morphology)
+ numpy
+ matplotlib (pyplot as plt)



# Pipeline
Preparatory steps:
+ A digitised high-resolution image of the histological sample is required

Script:
+ Import the image
+ Grayscale and Otsu's threshold: binary mask -> tissue: 1, background: 0
+ Splitting original image and binary mask into blocks (e.g., 35x35 pixels)
+ If the block contains enough tissue (more than half of the pixels), it is processed further:
  + The block is converted to the HSV colour space
  + Colour thresholds are calculated to identify stained areas
  + A mask is created to isolate these areas
  + The masked area is converted to grayscale and blurred
  + Otsu's thresholding method is applied to create a binary image where myelin areas are identified
+ The fraction of the block area covered by myelin is calculated
  

# Usage
+ Download the python script
+ Define the path where you have located your digitised histological image 
+ Define the path where you want to save the area fraction maps

# MIT License
