import cv2
import numpy as np
from ultralytics import YOLO
import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import torchinfo
import torch
from torchvision import transforms
from plotting_utils import plot_detections_vs_conf, plot_detections_vs_iou

COCO_DIR = "coco_val_30_with_gt"
COCO_IMAGES_PATH = f"{COCO_DIR}/images"
TEST_IMAGE_PATH = f"{COCO_DIR}/test_image_assignment_2.JPEG"

def print_task(idx):
    print(f"###### TASK {idx} ######")

### TASK 1 ###
print_task(1)
# Load model (downloads weights on first run)
model = YOLO('yolov8n.pt')

# Run torchinfo summary
pt_model = model.model

print(torchinfo.summary(pt_model))

### TASK 2 ###
print_task(2)
def preprocess_image(img, tensor_height, tensor_width):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((tensor_height, tensor_width)),
        transforms.ToTensor(),
    ])
    
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

#Load images from the MS COCO folder
# Raw predictions from YOLO model. Make sure the img_tensor is formatted accordingly
img_path = f"{COCO_IMAGES_PATH}/000000023272.jpg"

img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Warning: Could not read image {img_path}. Skipping.")

img_tensor = preprocess_image(img, 640, 640)
print(img_tensor.shape)

raw = model.model(img_tensor)

if isinstance(raw, (list, tuple)):
    for i, t in enumerate(raw):
        print(i, type(t), getattr(t, "shape", None))
else:
    print(type(raw), raw.shape)
    
def run_inference(model, img, conf=0.25, iou=0.45, verbose=False):
    # Run inference
    results = model.predict(source=img, conf=conf, iou=iou, verbose=verbose)

    r = results[0]
    names = model.names  # class id -> name
    # print(names)
    # print(r.boxes)

    # Extract detections
    # r.boxes.xyxy: (N,4), r.boxes.conf: (N,), r.boxes.cls: (N,)
    xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
    # print(scores)
    # print(cls_ids)
    
    return names, scores, cls_ids


def load_image_files():
    all_files = os.listdir(COCO_IMAGES_PATH)

    image_files = []
    image_extensions = ('.jpg', '.jpeg', '.png')

    for file_name in all_files:
        if file_name.lower().endswith(image_extensions):
            image_files.append(os.path.join(COCO_IMAGES_PATH, file_name))

    print(f"Found {len(image_files)} image files.")
    print("30 image files:")
    for i, img_file in enumerate(image_files[:30]):
        print(f"  {i+1}. {img_file}")
        
    return image_files

def print_bounding_box(img, img_path, results, output_dir):
    r = results[0]

    # Extract detections
    xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
    
    # Draw boxes
    vis = img.copy() # Make a copy to draw on, keep original img if needed
    for (x1, y1, x2, y2), s, c in zip(xyxy, scores, cls_ids):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names[c]} {s:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image with detections
    plt.figure(figsize=(12, 9))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Detections for {os.path.basename(img_path)}")
    plt.axis("off")

    plt.savefig(f"{output_dir}/detections_{os.path.basename(img_path)}.png")
    plt.close()

    return len(scores)

def predict_bounding_boxes(image_files_sel, conf=0.25, iou=0.45, verbose=False):
    num_detections = {}
    output_dir = f"{COCO_IMAGES_PATH}/detections_conf_{conf}_iou_{iou}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in image_files_sel:
        # print(f"Processing image: {img_path}")

        # Load image (BGR)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # Run inference
        results = model.predict(source=img, conf=conf, iou=iou, verbose=verbose)

        num_detections[img_path] = print_bounding_box(img, img_path, results, output_dir)

    print("Number of bounding boxes detected:")
    for img_path, count in num_detections.items():
        print(f"  {os.path.basename(img_path)}: {count}")

    return num_detections

confs = [0.05,0.10,0.25,0.50,0.75]
ious = [0.30,0.45,0.60,0.80]

names = model.names
image_files = load_image_files()
image_files_sel = image_files[:10] # First 10 images

### TASK 3 ###
print_task(3)

iou_fixed_for_conf = 0.45
num_detections_conf = {}
for conf in confs:
    print(f"Evaluating images with a confidence threshold of {conf}")
    num_detections_conf[conf] = predict_bounding_boxes(image_files_sel, conf=conf, iou=iou_fixed_for_conf)

### TASK 4 ###
print_task(4)
conf_fixed_for_iou = 0.25
num_detections_iou = {}
for iou in ious:
    print(f"Evaluating images with an IoU threshold of {iou}")
    num_detections_iou[iou] = predict_bounding_boxes(image_files_sel, conf=conf_fixed_for_iou, iou=iou)

plot_dir = "object_detection_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

plot_detections_vs_conf(num_detections_conf, fixed_iou=iou_fixed_for_conf, save_dir=plot_dir)
plot_detections_vs_iou(num_detections_iou, fixed_conf=conf_fixed_for_iou, save_dir=plot_dir)

### TASK 5 ###
print_task(5)
gt_yolo_dir = os.path.join(COCO_DIR, "labels_yolo")

image_extensions = (".jpg", ".jpeg", ".png")
image_files = sorted([
    os.path.join(COCO_IMAGES_PATH, f)
    for f in os.listdir(COCO_IMAGES_PATH)
    if f.lower().endswith(image_extensions)
])

image_files_sel = image_files[:30]
CONF_THRES = 0.001
IOU_NMS = 0.2

C = len(names)
N = len(image_files_sel)

def read_yolo_classes(txt_path):
    if not os.path.isfile(txt_path):
        return set()
    s = set()
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                s.add(int(float(parts[0])))
    return s

# Convert the class labels for an image into C-element binary vector
def create_class_id_bit_vector(true_class_label, C):
    binary_class_vector = np.zeros(C, dtype=int)
    for c in range(C):
        binary_class_vector[c] = 1 if c in true_class_label else 0

    return binary_class_vector

Y_true = {}
Y_scores = {}
# Build true and score class vectors
for img in image_files_sel:
    _, scores, cls_ids = run_inference(model, img, conf=CONF_THRES, iou=IOU_NMS, verbose=False)
    image_id = os.path.splitext(os.path.basename(img))[0]
    
    pred_vec = np.zeros(C, dtype=float)
    for score, cls_id in zip(scores, cls_ids):
        pred_vec[cls_id] = max(pred_vec[cls_id], score)
    Y_scores[image_id] = pred_vec

    true_class_label = read_yolo_classes(os.path.join(gt_yolo_dir, image_id + ".txt"))
    Y_true[image_id] = create_class_id_bit_vector(true_class_label, C)

# Construct a matrix out of dictionaries with the same image order
Y_true_matrix = []
Y_pred_matrix = []
for img in image_files_sel:
    img_id = os.path.splitext(os.path.basename(img))[0]
    Y_true_matrix.append(Y_true[img_id])
    Y_pred_matrix.append(Y_scores[img_id])

Y_true_matrix = np.array(Y_true_matrix)
Y_pred_matrix = np.array(Y_pred_matrix)

# Find class IDs that exist in ground-truth images
gt_present_classes = np.where(Y_true_matrix.sum(axis=0) > 0)[0]
print("Number of classes present in the ground-truth images:", len(gt_present_classes))
aps = []

#Compute mAP using average_precision_score function
for cls_id in gt_present_classes:
    y_true_cls = Y_true_matrix[:, cls_id]
    y_pred_cls = Y_pred_matrix[:, cls_id]
    ap = average_precision_score(y_true_cls, y_pred_cls)
    aps.append(ap)
    print("Class", cls_id, "AP:", ap)

mAP = np.mean(aps)
print(f"Multi-label mAP (over GT-present classes) = {mAP:.4f}")

### TASK 6 ###
print_task(6)
test_image = cv2.imread(TEST_IMAGE_PATH)
if test_image is None:
    print(f"Warning: Could not read image {TEST_IMAGE_PATH}. Skipping.")
    raise ValueError(f"Warning: Could not read image {TEST_IMAGE_PATH}.")

results = model.predict(source=test_image, conf=0.1, iou=0.45, verbose=False)
num_detections = print_bounding_box(test_image, TEST_IMAGE_PATH, results, plot_dir)
print("Number of detections: ", num_detections)

### TASK 7 ###
print_task(7)
model = YOLO("yolo11n.pt")
pt_model = model.model
print(torchinfo.summary(pt_model))

results = model.predict(source=test_image, conf=0.1, iou=0.45, verbose=False)
num_detections = print_bounding_box(test_image, TEST_IMAGE_PATH, results, plot_dir)
print("Number of detections: ", num_detections)