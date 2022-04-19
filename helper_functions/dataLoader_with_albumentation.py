import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tqdm


class TrainDataset(Dataset):
    def __init__(self, label_dir, image_dir, ImageSize):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.labels = [item for item in os.listdir(label_dir) if item.endswith(".txt")]
        self.images = [item for item in os.listdir(image_dir) if item.endswith(".jpg")]
        self.ImageSize = ImageSize
        self.test_image_labels()

    def test_image_labels(self):
        valid_label_files = []
        for label_file in tqdm.tqdm(self.labels, desc="check validity of labels"):
            # -------------------------------------------------
            # Read detections labeled in yolo format [x_center, y_center, width, height] normalized and between 0-1
            # -------------------------------------------------
            # read file in to list of lines if it is not empty
            files = [line.strip().split() for line in open(os.path.join(self.label_dir, label_file)) if len(line) > 0]
            # split each line in the "list of lines" that is in "list of string" format
            # to "list of num values of boxes" format
            bboxes = np.array([[float(item[1]) - (float(item[3]) / 2),  # x_min
                                float(item[2]) - (float(item[4]) / 2),  # y_min
                                float(item[1]) + (float(item[3]) / 2),  # x_max
                                float(item[2]) + (float(item[4]) / 2),  # y_max
                                float(item[3]),  # width
                                float(item[4])]  # height
                               for item in files if len(item) > 3])

            # -------------------------------------------------
            # Skip invalid labels and images if the label is invalid
            # 1- box coordinate should be between 0 and 1
            # 2- y_max and x_max should be greater than y_min and x_min
            # -------------------------------------------------
            if bboxes.shape[0] > 0:  # there should be at least one box in the image
                if not np.isnan(bboxes).any(): # only valid if there is no nan value inside
                    array_a = np.where((bboxes[:, 0] < 0) | (bboxes[:, 0] > 1))[0]
                    array_b = np.where((bboxes[:, 1] < 0) | (bboxes[:, 1] > 1))[0]
                    array_c = np.where((bboxes[:, 2] < 0) | (bboxes[:, 2] > 1))[0]
                    array_d = np.where((bboxes[:, 3] < 0) | (bboxes[:, 3] > 1))[0]
                    # width and height should be greate than 0.0007
                    array_e = np.where((bboxes[:, 4] < 0.0007) | (bboxes[:, 5] < 0.0007))[0]
                    if len(array_a) == 0 and len(array_b) == 0 and len(array_c) == 0 \
                            and len(array_d) == 0 and len(array_e) == 0:
                        valid_label_files.append(label_file)

        # -------------------------------------------------
        # Skip images without labels files based on the filtered labels
        # -------------------------------------------------
        valid_images = [item for item in self.images if item[:-3]+"txt" in valid_label_files]

        # -------------------------------------------------
        # Skip label files that do not have any image belonging to it
        # -------------------------------------------------
        valid_label_files = [item for item in valid_label_files if item[:-3]+"jpg" in valid_images]
        print("-" * 50)
        print(f"Total number of available images and text files: {len(self.images)} / {len(self.labels)}")
        print("-"*50)
        print(f"Total number of valid images and text files: {len(valid_images)} / {len(valid_label_files)}")
        print("-" * 50)
        self.images = valid_images
        self.labels = valid_label_files


    def __getitem__(self, index: int):
        image_name = self.images[index]
        # print(image_name)
        try:
            image = cv2.imread(os.path.join(self.image_dir, image_name))
        except:
            print(os.path.join(self.image_dir, image_name))
            exit()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_w = image.shape[0]
        img_H = image.shape[1]

        label_file = os.path.splitext(image_name)[0] + ".txt"
        if label_file in self.labels:
            # -------------------------------------------------
            # Read detections labeled in yolo format [x_center, y_center, width, height] normalized and between 0-1
            # -------------------------------------------------
            files = [line.strip().split() for line in open(os.path.join(self.label_dir, label_file))]

            # -------------------------------------------------
            # Transform them to Pascal format [x_min, y_min, x_max, y_max]
            # This format is needed for Faster RCNN and Mask-RCNN
            # -------------------------------------------------
            bboxes = [[int((float(item[1]) - (float(item[3]) / 2)) * img_w),  # x_min
                       int((float(item[2]) - (float(item[4]) / 2)) * img_H),  # y_min
                       int((float(item[1]) + (float(item[3]) / 2)) * img_w),  # x_max
                       int((float(item[2]) + (float(item[4]) / 2)) * img_H)]  # y_max
                      for item in files]

            current_labels = [int(item[0])+1 for item in files]  # reserve label 0 for background

            area = [(item[2]-item[0])*(item[3]-item[1]) for item in bboxes]

        else:
            bboxes = []
            current_labels = []
            area = []

        image /= 255.0  # for more stabilty of trainings

        transformed = self.image_bbox_albumentation({"image": image, "bboxes": bboxes, "labels": current_labels})

        target = {"boxes": torch.tensor(transformed['bboxes'], dtype=torch.float32),
                  "labels": torch.as_tensor(transformed['class_labels'], dtype=torch.int64),
                  'image_id': torch.tensor([index]),
                  'area': torch.as_tensor(area, dtype=torch.float32),
                  'iscrowd': torch.zeros((len(current_labels),), dtype=torch.int64)
        }

        return transformed['image'], target, image_name

    def __len__(self) -> int:
        return len(self.images)

    def image_bbox_albumentation(self, data: dict):
        transform = A.Compose([
            # A.RandomCrop(width=450, height=450),
            # A.resize(1280,1280),
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.Resize(self.ImageSize, self.ImageSize),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0,
            #             always_apply=False, p=1.0), # do image normalization so pixel values willl be around 0,1 by mean 0.5
            A.pytorch.transforms.ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        return transform(image=data["image"], bboxes=data["bboxes"], class_labels=data["labels"])

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':

    DIR_INPUT = "/media/vahid/Elements/Data/face_plate_copy/dataLoader_problem"
    IMG_SIZE = 1280

    # -------------------------------------------------
    # Initialize our custom data loader
    # -------------------------------------------------
    train_dataset = TrainDataset(os.path.join(DIR_INPUT, 'labels'),
                                 os.path.join(DIR_INPUT, 'images'),
                                 IMG_SIZE)

    print(f'length of train dataset {len(train_dataset)}')
    print("-"*50)

    BATCH_NUM = 1
    # Load dataset to the torch using its DataLoader function
    data_loader = DataLoader(train_dataset, batch_size=BATCH_NUM, shuffle=False, num_workers=1,
                             collate_fn=TrainDataset.collate_fn)

    # Select the device to train on
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #/////////////////// Show one sample of training data
    images, targets, _ = next(iter(data_loader))
    images = list(image.to(device) for image in images)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))

    #  plot 4 images -> minimum batch number should be 4
    for idx in range(1):
        ax = plt.subplot(2, 2, idx + 1)  # set ax for plotting
        boxes = targets[idx]['boxes'].cpu().numpy()
        labels = targets[idx]['labels'].cpu().numpy()
        sample = images[idx].permute(1, 2, 0).cpu().numpy()
        for box, label in zip(boxes, labels):
            cv2.rectangle(sample,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (220, 0, 0), 3)
            sample = cv2.putText(img=sample, text=str(label), org=(int(box[0]), int(box[1])),
                                 fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
                                 color=(125,246,55), thickness=2, lineType=cv2.LINE_AA)

        ax.set_axis_off()
        ax.imshow(sample)

    plt.tight_layout()  # reduce margin of subplots
    plt.show()

    # -------------------------------------------------
    # Test data ( iterate through dataloader and check for errors)
    # -------------------------------------------------
    for i, (image, targets, image_name) in enumerate(data_loader):
        if i%100 == 0:
            print(i)
            # images = list(image.to(device) for image in images)
            # targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    print(i)

