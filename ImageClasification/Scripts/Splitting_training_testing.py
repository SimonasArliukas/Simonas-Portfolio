import os
from os import getcwd
from sklearn.model_selection import train_test_split

categories = ["cat", "dog", "giraffe", "goat", "tortoise"] ##Take the classes
base_path = os.path.join(getcwd(),"OID/train") ##take my directory and go into train data

all_pairs = []

for cat in categories:
    # Where my img_folder is located
    img_folder = os.path.join(base_path, cat)
    # Where my label folder is located
    label_folder = os.path.join(img_folder, "labels")


    images = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg')]) #Select all the images

    for img_name in images:
        print(f"Scrapping Image {img_name}")
        # Get image path
        img_path = os.path.join(img_folder, img_name)

        #Get label path
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(label_folder, label_name)

        ##Append image path label path and class name
        all_pairs.append((img_path, label_path, cat))




# Split the paired list (80% Train, 20% Test)
train_data, test_data = train_test_split(
    all_pairs,
    test_size=0.2,
    random_state=1,
    stratify=[p[2] for p in all_pairs]  # Use stratified sampling
)

