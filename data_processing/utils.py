import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image
import cv2

def load_images(base_dir) : 

    images = {}

    for patient_dir in os.listdir(base_dir):

        patient_path = os.path.join(base_dir, patient_dir)
        
        if os.path.isdir(patient_path):
            # Traverse the directory by image type
            for image_type_dir in os.listdir(patient_path):
                image_type_path = os.path.join(patient_path, image_type_dir)
                
                if os.path.isdir(image_type_path):
                    # Check for 'row.tif'
                    for file in glob.glob(os.path.join(image_type_path, 'raw.tif')):
                        images[patient_dir + "_" + image_type_dir] = {"raw" : cv2.imread(file,  flags=cv2.IMREAD_GRAYSCALE), "labelled" : []}
                    
                    for coronary_label_dir in os.listdir(image_type_path):
                        coronary_label_path = os.path.join(image_type_path, coronary_label_dir)
                        for file in glob.glob(os.path.join(coronary_label_path, 'labelled.tif')):
                            images[patient_dir + "_" + image_type_dir]["labelled"].append(cv2.imread(file, flags=cv2.IMREAD_GRAYSCALE))
                            
    return images

def distribution_per_center(images, center_means = True, scaled=False) : 

    hospital_data = {}

    # Group raw arrays by hospital number
    for patient_id, patient_data in images.items():
        hospital_number = patient_id.split("-")[0]  # Extract hospital number (e.g., "01")
        
        # Initialize list if hospital number is not in the dictionary
        if hospital_number not in hospital_data:
            hospital_data[hospital_number] = []
        
        # Append the raw array for the current patient
        if scaled :
            hospital_data[hospital_number].append(cv2.calcHist([np.array(patient_data['raw'])],[0],None,[256],[-4,4]).flatten())
        else :
            hospital_data[hospital_number].append(cv2.calcHist([np.array(patient_data['raw'])],[0],None,[256],[0,256]).flatten())

    if center_means : 
        # Compute the mean pixel intensity for each hospital
        hospital_means = {}
        for hospital_number, arrays in hospital_data.items():
            mean_array = np.mean(arrays, axis=0)
            hospital_means[hospital_number] = mean_array
            
        hospital_means = dict(sorted(hospital_means.items()))
        return hospital_data, hospital_means
    else :
        return hospital_data
    
import cv2
import itertools  # For generating pairs
def calculate_similarity(intensity_data, hospital_name):
    # Calculate Bhattacharyya distance between all pairs of histograms
    similarity_data = []

    # Iterate over all unique pairs of histograms (no repeats)
    for (i, hist1), (j, hist2) in itertools.combinations(intensity_data.items(), 2):

        # Calculate Bhattacharyya distance
        bhattacharyya_dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        
        # Store the result in a dictionary
        similarity_data.append({
            'Hospital': hospital_name,
            'Image 1': i,
            'Image 2': j,
            'Bhattacharyya Distance': bhattacharyya_dist
        })
    return similarity_data

def ridgeline(data, overlap=0, fill=True, n_points=255, scaled = True, plot=plt):
    
    """
    Creates a standard ridgeline plot.

    data, dict of array.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(0,255, n_points) if not scaled else np.linspace(-4,4, n_points)

    column_names = list(data.keys())
    pal = sns.color_palette("flare", len(column_names))
    
    ys = []
    for i, d in enumerate(column_names):
        pdf = data[d][1:]
        y = i*(3000-overlap)
        ys.append(y)

        if fill:
            plot.fill_between(xx, np.ones(n_points)*y, 
                             pdf+y, zorder=len(data)-i+1, color = pal[i])
        plot.plot(xx, pdf+y, c="k", zorder=len(data)-i+1)
    if plot==plt :
        plot.yticks(ys, column_names)
    else :
        plot.set_yticks(ys, column_names)