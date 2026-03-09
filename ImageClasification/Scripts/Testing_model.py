from Data_Preprocessing import images_test,y_test_table
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Modelling import DetectionModel,calculate_iou
import torch
import torch.nn as nn


model=DetectionModel(num_classes=5)

test_labels = y_test_table[:, 0].long()
test_boxes = y_test_table[:, 1:].float()

test_boxes

model.load_state_dict(torch.load("classification_model.pth"))
model.eval()


with torch.no_grad():

    p_cls, p_box = model(images_test)

    pred_labels = torch.argmax(p_cls, dim=1)



y_true = test_labels.cpu().numpy()
y_pred = pred_labels.cpu().numpy()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

print(f"--- Classification Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")


# Usage in your code:
iou_score = calculate_iou(p_box, test_boxes).item()

print(f"\n--- Detection Metrics ---")
print(f"Mean IoU: {iou_score:.4f}")