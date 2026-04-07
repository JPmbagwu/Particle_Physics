#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os #This line uses and imports the modules of OS.
import re #This line uses and imports the modules of RE.
import numpy as np #This line uses and imports the modules of Numpy.
import cv2 #This line uses and imports the modules of CV2.
from PIL import Image #This line uses PIL and imports its Image module.
from sklearn.cluster import KMeans #This line uses sklearn.cluster and imports KMeans. 
from scipy.spatial.distance import hamming #This line uses scipy.spatial.distance and imports hamming for hamming distance.
import matplotlib.pyplot as plt #This line uses and imports the modules of Matplotlib
import pandas as pd #This line uses and imports the modules of Pandas.

"""
Creation Date: April 25, 2025
@author of Section 1: Chetan Hiremath and Nirupam Dasika

Part I - System_A Implementation (Final Correct Version)
"""

"""
These 6 lines that are below this multi-line comment define the numericalfile() function that numerically sorts the files of the folders.
"""
def numericalfile(filename):
    filenumbermatch = re.search(r'(\d+)', filename) #This line defines the filenumbermatch variable that uses the re.search() function to handle regular expressions and use the pattern for matching digits.
    #This if-statement with 2 lines checks if the match is valid.
    if filenumbermatch:
        return int(filenumbermatch.group(1)) #This line converts the string of digits to an integer.
    return -1 #This line returns -1 if there is no match.

"""
These 32 lines that are below this multi-line comment define the binarization() function that performs binarization and uses GallerySet and ProbeSet's face images to perform the comparison of gallery and probe face images and generate the 100 X 100 score matrix and other calculations that are rounded to 2 decimal places by the built-in round() function.
"""  
def binarization(inputfolder1, inputfolder2):
    os.makedirs(inputfolder1, exist_ok = True) #This line checks if the first folder exists.
    os.makedirs(inputfolder2, exist_ok = True) #This line checks if the second folder exists.
    pgm_files1 = sorted([f for f in os.listdir(inputfolder1) if f.endswith('.pgm')], key = numericalfile)[:100] #This line defines the pgm_files1 variable that gets the first 100 files of the first folder, filters them if the filenames end with '.pgm', and sorts them numerically by using the numericalfile() function.
    pgm_files2 = sorted([f for f in os.listdir(inputfolder2) if f.endswith('.pgm')], key = numericalfile)[:100] #This line defines the pgm_files2 variable that gets the first 100 files of the second folder, filters them if the filenames end with '.pgm', and sorts them numerically by using the numericalfile() function.
    threshold = 128 #This line defines the threshold variable that is used for thresholding.
    gallery_images = [(np.array(Image.open(os.path.join(inputfolder1, f)).convert('L')) > threshold).astype(np.uint8).flatten() for f in pgm_files1] #This line defines the gallery_images variable that uses thresholding and converts the files of the first folder to binary images, which are converted to 1D arrays. If the grayscale pixel is greater than the threshold, then the output binary pixel is 1. If not, then the output binary pixel is 0.
    probe_images = [(np.array(Image.open(os.path.join(inputfolder2, f)).convert('L')) > threshold).astype(np.uint8).flatten() for f in pgm_files2] #This line defines the probe_images variable that uses thresholding and converts the files of the second folder to binary images, which are converted to 1D arrays. If the grayscale pixel is greater than the threshold, then the output binary pixel is 1. If not, then the output binary pixel is 0.
    scorematrix = np.zeros((len(gallery_images), len(probe_images))) #This line defines the scorematrix variable that creates a 2D matrix of the given shape with zeros.
    pixelcount = gallery_images[0].size #This line defines the pixelcount variable that finds the number of pixels of the gallery face image.
    #The nested for-loop with 4 lines enumerates or iterates to generate gallery and probe face images' respective comparisons' scores that are stored in the score matrix.
    for i, probe_image in enumerate(probe_images):
        for j, gallery_image in enumerate(gallery_images):
            hammingdistancescore = (np.sum(gallery_image != probe_image) / pixelcount) * 100 #This line defines the hammingdistancescore variable that finds the number of positions of different pixels of gallery and probe face images.
            scorematrix[i, j] = hammingdistancescore  #This line uses the hammingdistancescore variable to find the respective scores of the hamming distances of gallery and probe face images. 
    df = pd.DataFrame(np.round(scorematrix, 2), index = pgm_files2, columns = pgm_files1) #This line defines the df variable that creates an organized score matrix with labels.
    print("Score Matrix:") #This line prints the "Score Matrix:" message.
    print(df) #This line prints the organized 2D score matrix.
    genuinescores = np.diag(df.values) #This line defines the genuinescores variable that determines the genuine scores of the gallery and probe face images of the same subjects by finding the diagonal values of the score matrix.
    imposterscores = df.values[~np.eye(df.values.shape[0], dtype = bool)] #This line defines the imposterscores variable that determines the imposter scores of the gallery and probe face images of the different subjects by finding the non-diagonal values of the score matrix.
    decidabilityindexvalue = (np.sqrt(2) * (np.abs(np.mean(genuinescores) - np.mean(imposterscores)))) / (np.sqrt((np.std(genuinescores) ** 2) + (np.std(imposterscores) ** 2))) #This line defines the decidabilityindexvalue variable that uses the decidability index value formula to calculate the decidability index value.
    print("Genuine scores:", genuinescores) #This line prints the genuine scores of the score matrix.
    print("Imposter scores:", imposterscores) #This line prints the imposter scores of the score matrix.
    print("Decidability Index Value of System_A =", round(decidabilityindexvalue, 2)) #This line prints the decidability index value of the score matrix. 
    facerecdecidabilityindexvalue = 3.5 #This line defines the facerecdecidabilityindexvalue variable that uses 3.5 as the decidability index value of FaceRec.
    #The if-statement with 4 lines checks if the decidability index value of this system is greater than the decidability index value of FaceRec.
    if decidabilityindexvalue > facerecdecidabilityindexvalue:
        print("System_A's performance is better than FaceRec's performance since", round(decidabilityindexvalue, 2), "or System_A's decidability index value is greater than", facerecdecidabilityindexvalue, "or FaceRec's decidability index value.") #This line prints this message if the decidability index value of this system is greater than the decidability index value of FaceRec.
    else:
        print("System_A's performance is worse than FaceRec's performance since", round(decidabilityindexvalue, 2), "or System_A's decidability index value is less than", facerecdecidabilityindexvalue, "or FaceRec's decidability index value.") #This line prints this message if the decidability index value of this system is less than the decidability index value of FaceRec.
    df.to_csv("scorematrix.csv") #This line writes the score matrix that is saved in the scorematrix.csv file.
    return round(decidabilityindexvalue, 2) #This line returns the decidability index value of this system.
    
inputfolder1 = "./GallerySet" #This line defines the inputfolder1 variable that uses the GallerySet folder.
inputfolder2 = "./ProbeSet" #This line defines the inputfolder2 variable that uses the ProbeSet folder.
decidabilityindexvalueA = binarization(inputfolder1, inputfolder2) #This line defines the decidabilityindexvalueA variable that uses the binarization() function to perform the comparison of gallery and probe face images by using the GallerySet folder and the ProbeSet folder for thresholding and hamming distance and stores the decidability index value of this system for future calculations.
print() #This line creates a new line to organize the results.

"""
Created on Apr 28, 2025
@author of Section 2: johnpaulmbagwu and Michael Chukwuka

Part II - System_B Implementation (Final Correct Version)
"""
   
# Define gallery and probe set paths
gallery_path = "./GallerySet"
probe_path = "./ProbeSet"
 
# Step 1: Load and preprocess images
def load_image(path):
  """Load grayscale image and normalize."""
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  img = cv2.equalizeHist(img)  # Normalize brightness/contrast
  return img

def kmeans_binarize(img):
  """Apply KMeans clustering (k=2) to binarize image consistently."""
  pixels = img.reshape(-1, 1)
  kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(pixels)
  labels = kmeans.labels_.reshape(img.shape)

  centers = kmeans.cluster_centers_.flatten()
  high_intensity_label = np.argmax(centers)  # Always map high intensity to 1
  binary_img = (labels == high_intensity_label).astype(np.uint8)
  return binary_img

def preprocess_dataset(folder_path):
  """Load and binarize all images from a folder."""
  files = sorted(os.listdir(folder_path))
  binarized_images = []
  for filename in files:
      img = load_image(os.path.join(folder_path, filename))
      bin_img = kmeans_binarize(img)
      binarized_images.append(bin_img)
  return binarized_images

# Step 2: Preprocess gallery and probe sets
gallery_images = preprocess_dataset(gallery_path)
probe_images = preprocess_dataset(probe_path)

# Step 3: Build score matrix B
def build_score_matrix(probe_images, gallery_images):
  n_probe = len(probe_images)
  n_gallery = len(gallery_images)

  B = np.zeros((n_probe, n_gallery))
  for i in range(n_probe):
      probe_bin = probe_images[i]
      for j in range(n_gallery):
          gallery_bin = gallery_images[j]
          B[i, j] = hamming(probe_bin.flatten(), gallery_bin.flatten())
  return B

B = build_score_matrix(probe_images, gallery_images)
np.save("score_matrix_B.npy", B)

# Display part of the score matrix
print("Score Matrix B[0:9, 0:9]:\n", B[0:10, 0:10])

# Step 4: Calculate Decidability Index d'
genuine_scores_B = np.diag(B)
impostor_scores_B = B[~np.eye(100, dtype=bool)]

mu_g = np.mean(genuine_scores_B)
mu_i = np.mean(impostor_scores_B)
std_g = np.std(genuine_scores_B)
std_i = np.std(impostor_scores_B)

d_prime_B = abs(mu_g - mu_i) / np.sqrt(0.5 * (std_g**2 + std_i**2))

print(f"\nDecidability Index (d') for System_B: {d_prime_B:.3f}")

# Step 5: Calculate Improvement Factor
# Assume d_prime_A from Part I is known
d_prime_A = decidabilityindexvalueA  # <-- Replace with your actual System_A d'
delta_d_prime = d_prime_B - d_prime_A
improvement_factor = round(delta_d_prime, 2)

print(f"Improvement Factor (IF): {improvement_factor}")

# Step 6: Plot genuine vs impostor score distributions
plt.figure(figsize=(8, 5))
plt.hist(impostor_scores_B, bins=50, alpha=0.6, label='Impostor Scores', color='red', density=True)
plt.hist(genuine_scores_B, bins=50, alpha=0.6, label='Genuine Scores', color='blue', density=True)
plt.title("System_B: Score Distributions")
plt.xlabel("Hamming Distance")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
