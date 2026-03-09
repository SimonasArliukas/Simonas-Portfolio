from Data_Preprocessing import train_loader
import torch
import torch.nn as nn


class DetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DetectionModel, self).__init__() #Connects to pytorch infrastructure

        # ======Encoder-------

        #Start 3x224x224. 3 color channels and 224x224 image. Low level features
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),  #32x224x224. 32 features maps that are 224x224. #Kernel is 32x3x3x3 (padding keeps original dimensions)
        nn.BatchNorm2d(32), #Normalizing the shape
        nn.ReLU(), #Activation function
        nn.MaxPool2d(2, 2), #32x112x112

        #32x112x112 with low-medium level features
        nn.Conv2d(32, 64, 3, padding=1),#Kernel size 64x32x3x3 #64x112x112
        nn.BatchNorm2d(64), #Normalizing
        nn.ReLU(), #Activation
        nn.MaxPool2d(2, 2), #64x56x56

        #Medium level features #64x56x56
        nn.Conv2d(64, 128, 3, padding=1),#Kernel size 128x64x3x3 #128x56x56
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), #128x28x28

        nn.Conv2d(128, 256, 3, padding=1), #Kernel size 256x128x3x3 #256x28x28
        nn.BatchNorm2d(256), #Normalizing
        nn.ReLU(), #Activation
        nn.MaxPool2d(2,2),

        nn.Conv2d(256,512,3,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()


    )


        self.classifier = nn.Sequential(
            nn.Linear(512, 256), #Fully connected layers 256 -> 256
            nn.ReLU(), #Activation
            nn.Dropout(0.3), #We deactivate 30% of the neurons to not overfit and memorize.
            nn.Linear(256, 128),
            nn.ReLU(),  #activation function
            nn.Linear(128, num_classes) #Linear map to classes
        )


        self.box_regressor = nn.Sequential(
            nn.Linear(4608, 256), #Fully connected layer 4096 -> 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),  #Fully connected 128 -> 64
            nn.ReLU(),
            nn.Linear(64, 4), #Maps to the final box function
            nn.Sigmoid()
        )

    def forward(self, x):
        #Combining the heads

        features = self.encoder(x) #Takes the high level features

        class_features = torch.mean(features, dim=[2, 3])
        class_logits = self.classifier(class_features) #Classifies

        #box_features = nn.AdaptiveAvgPool2d((3, 3))(features) #256x32x28x28 -> 256x32
        #box_features = box_features.view(box_features.size(0), -1)
        #box_coords = self.box_regressor(box_features)


        return class_logits


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = torch.max(boxA[:, 0], boxB[:, 0])
    yA = torch.max(boxA[:, 1], boxB[:, 1])
    xB = torch.min(boxA[:, 2], boxB[:, 2])
    yB = torch.min(boxA[:, 3], boxB[:, 3])

    # Compute the area of intersection
    interWidth = torch.clamp(xB - xA, min=0)
    interHeight = torch.clamp(yB - yA, min=0)
    interArea = interWidth * interHeight

    # Compute the area of both bounding boxes
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    # Compute IoU (Union = Area A + Area B - Intersection)
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou.mean()




if __name__ == "__main__":

    #Puts dataset into Tensore object


    num_classes = 5
    model = DetectionModel(num_classes=num_classes) #Initiating the class



    def train_detection_model(model, train_loader, epochs=40):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #Optimizer Adam
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5) #Lower step size by 50% every 7 epoch for convergence
        criterion_cls = nn.CrossEntropyLoss()
        #criterion_box = nn.SmoothL1Loss()
        for epoch in range(epochs):
            for i,(imgs, labels, boxes) in enumerate(train_loader):
                optimizer.zero_grad() #Initialize gradient
                p_cls = model(imgs) #Model Prediction
                loss_cls = criterion_cls(p_cls, labels)

                #smooth_l1 = criterion_box(p_box, boxes)
                #iou = calculate_iou(p_box, boxes)
                #loss_box = 0.5 * smooth_l1 + 0.5 * (1 - iou)
                total_loss = (1 * loss_cls)  #Evaluate total loss
                total_loss.backward() #Backpropagation to see where to go

                optimizer.step() #Take a step with the optimizer
                if i % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{epochs}], Batch [{i}/{len(train_loader)}], Total: {total_loss.item():.4f} (Cls: {loss_cls.item():.2f}")
            scheduler.step() #After 7 epochs lower the step
        torch.save(model.state_dict(), "classification_model.pth") #Save the model
        print("Model saved successfully!")

        return model


    train_detection_model(model, train_loader) #Start training