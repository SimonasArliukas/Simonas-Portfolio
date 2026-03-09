from Splitting_training_testing import train_data, test_data
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors


class OIDDataset(Dataset):

    def __init__(self, data_tuples, transform=None):
        self.data = data_tuples ##Directory tuples
        self.transform = transform ##Transformations to apply
        self.label_map = { #Label mapping
            "cat": 0,
            "dog": 1,
            "giraffe": 2,
            "goat": 3,
            "tortoise": 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path, label_path, animal_name = self.data[idx]

        # Load image
        img = Image.open(img_path).convert("RGB") ##ADD RGB channels
        orig_w, orig_h = img.size ##Image size



        with open(label_path, "r") as f:
            parts = f.readline().split()
            ##Taking the rectangle box.
            xmin = float(parts[1])
            ymin = float(parts[2])
            xmax = float(parts[3])
            ymax = float(parts[4])

        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32) ##Putting the boxes into torch tensor

        # Adding metadata to boxes how it is proportional to the image
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="xyxy",
            canvas_size=(orig_h, orig_w)
        )

        ##Adding labels instead of names
        label = torch.tensor(self.label_map[animal_name], dtype=torch.long)

        # Apply transforms
        if self.transform:
            img, boxes = self.transform(img, boxes)

        boxes=boxes/224

        return img, label, boxes.squeeze(0)


#Transformation for training data
train_transforms = v2.Compose([
    v2.Resize((224, 224)), ##Resize image to 224 width and 224 height
    v2.ColorJitter(brightness=0.3, contrast=0.3),#Adjusts the brightness and contrast so that image doesn't memorize lightning
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), #Converts RGB pixels to [0,1]
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225] #Standardizes the data using Image net specifications
    )
])


test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#Initialize objects
train_dataset = OIDDataset(train_data, transform=train_transforms)
test_dataset = OIDDataset(test_data, transform=test_transforms)

#This repeatedly calls the __get__ magic method to grab all the data and puts it in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

##Looping through dataloader for test images
test_targets = []
test_images_list = []

for images_test, class_labels_test, boxes_test in test_loader:

    labels_column_test = class_labels_test.view(-1, 1).float()
    y_test_batch = torch.cat((labels_column_test, boxes_test), dim=1)
    test_targets.append(y_test_batch)


    test_images_list.append(images_test)


y_test_table = torch.cat(test_targets, dim=0)
images_test = torch.cat(test_images_list, dim=0)

y_test_table


width = y_test_table[:,3] - y_test_table[:,1]
height = y_test_table[:,4] - y_test_table[:,2]

print(width.mean(), height.mean())