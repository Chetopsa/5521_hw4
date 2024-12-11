import os
import numpy as np
import pandas as pd
from PIL import Image
# Change if we want to process cloud or noncloud photos
CLOUD = False
image_dir = './cloud/cloud' if CLOUD else './noncloud/noncloud'

# Initialize an empty list to hold the feature vectors
all_features = []

# Iterate through all files in the directory
counter = 0
for filename in os.listdir(image_dir):
    if counter == 400:  # limit processing to 400 images
        break
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        # Open the image file
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)

        grayscale_image = image.convert("L")  # "L" mode converts to grayscale

        #convert to numpy array
        grayscale_array = np.array(grayscale_image)

        # flatten array
        flattened_features = grayscale_array.flatten()

        # add flattened array to list
        all_features.append(flattened_features)
    counter += 1

# make the first column of each row the class
if CLOUD:
    all_features = np.insert(all_features, 0, 1, axis=1)
else:
    all_features = np.insert(all_features, 0, 0, axis=1)
# convert list of features to dataframe
df = pd.DataFrame(all_features)


# Ssave the features to a CSV file
output_csv = './cloud_features.csv' if CLOUD else './noncloud_features.csv'

# write to CSV file
df.to_csv(output_csv, index=False)

print(f"Successful proccessing of {'cloud' if CLOUD else 'noncloud'} to {output_csv}")
