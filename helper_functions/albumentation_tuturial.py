import random
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import os
from albumentations.pytorch.transforms import ToTensorV2

def visualize_only_image(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()



def only_image_albumentation(image):
    transform = A.Compose([
        A.CLAHE(),
        # A.RandomRotate90(),
        # A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.50, rotate_limit=30, p=.75),
        A.Blur(blur_limit=3),
        # A.OpticalDistortion(),
        # A.GridDistortion(),
        A.HueSaturationValue(),
    ])

    return transform(image=image)['image']

def image_bbox_albumentation(data:dict):
    transform = A.Compose([
        # A.RandomCrop(width=450, height=450),
        # A.resize(1280,1280),
        # A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),

        A.Resize(1280, 1280),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
                    always_apply=False, p=1.0),
        A.pytorch.transforms.ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


    return transform(image=data["image"], bboxes=data["bboxes"], class_labels=data["labels"])




if __name__ == '__main__':
    image_dir = "../sample_data/images"
    label_dir = "../sample_data/labels"
    # -------------------------------------------------
    # Read sample image
    # -------------------------------------------------
    images = [item for item in os.listdir(image_dir)]
    image = cv2.imread(os.path.join(image_dir, images[0]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_w = image.shape[0]
    img_H = image.shape[1]

    # -------------------------------------------------
    # Visualize only image
    # -------------------------------------------------
    visualize_only_image(image)

    # -------------------------------------------------
    # augment and visualize only image
    # -------------------------------------------------
    augmented_image = only_image_albumentation(image)
    visualize_only_image(augmented_image)

    # -------------------------------------------------
    # Read detections labeled in yolo format [x_center, y_center, width, height] normalized and between 0-1
    # -------------------------------------------------
    files = [line.strip().split() for line in open(os.path.join(label_dir, images[0][:-3]+"txt"))]

    # -------------------------------------------------
    # Transform them to Pascal format [x_min, y_min, x_max, y_max]
    # This format is needed for Faster RCNN and Mask-RCNN
    # -------------------------------------------------
    bboxes = [[int((float(item[1]) - (float(item[3])/2)) * img_w),  # x_min
               int((float(item[2]) - (float(item[4])/2)) * img_H),  # y_min
               int((float(item[1]) + (float(item[3]) / 2)) * img_w),  # x_max
               int((float(item[2]) + (float(item[4]) / 2)) * img_H)]  # y_max
              for item in files]

    labels = [int(item[0]) for item in files]

    # print(bboxes)
    # print(labels)
    # -------------------------------------------------
    # Visualize image and labels before transform
    # -------------------------------------------------
    category_id_to_name = {0: 'car', 1: 'person'}
    img_to_show = visualize(image, bboxes, labels, category_id_to_name)
    # cv2.imwrite("test.png", img_to_show)


    transformed = image_bbox_albumentation({"image": image, "bboxes": bboxes, "labels": labels})
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']

    # -------------------------------------------------
    # Visualize image and labels after transform
    # -------------------------------------------------
    my_new_img = transformed_image.permute(1, 2, 0).numpy() # transform image from tensor to numpy
    # print(transformed_bboxes)
    # print(transformed_class_labels)
    img_to_show = visualize(my_new_img, transformed_bboxes, transformed_class_labels, category_id_to_name)
