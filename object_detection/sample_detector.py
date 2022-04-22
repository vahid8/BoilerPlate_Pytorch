import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import tqdm
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class DetectDataset(Dataset):
    def __init__(self, image_dir, ImageSize):
        super().__init__()
        self.image_dir = image_dir
        self.images = [item for item in os.listdir(image_dir) if item.endswith(".jpg")]
        self.ImageSize = ImageSize

    def __getitem__(self, index: int):
        image_name = self.images[index]
        # print(image_name)
        try:
            PIL_image = Image.open(os.path.join(self.image_dir, image_name))
            # image = cv2.imread(os.path.join(self.image_dir, image_name))
        except:
            print(os.path.join(self.image_dir, image_name))
            exit()

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # # image /= 255.0  # for more stabilty of trainings
        # PIL_image = Image.fromarray(image.astype('uint8'), 'RGB')
        transformed = self.data_transforms(PIL_image)

        return transformed, image_name

    def __len__(self) -> int:
        return len(self.images)

    def data_transforms(self, img):
        transform = transforms.Compose([
            transforms.Resize(self.ImageSize),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    state_dict_path = "/home/tower/Codes/faster_out_test/temp.pt"
    IMG_SIZE = 1280
    num_classes = 3
    BATCH_NUM = 5
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    DIR_INPUT = "/home/tower/Codes/testing_rcnn_data/test/img5"
    OUT_DIR = DIR_INPUT
    # load data
    detect_dataset = DetectDataset(os.path.join(DIR_INPUT), IMG_SIZE)
    # using built in torch data loader to load images in batches
    # Load dataset to the torch using its DataLoader function
    detect_data_loader = DataLoader(detect_dataset, batch_size=BATCH_NUM, shuffle=False, num_workers=1,
                                    collate_fn=detect_dataset.collate_fn)

    # load the base Architecture
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    model = FasterRCNN(backbone, num_classes=num_classes)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()  # activate interface mode of the model (batchnorm or dropout layers will work in eval mode instead of training mode.)

    with torch.no_grad():  # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop
        for images, image_name in tqdm.tqdm(detect_data_loader):
            images = list(image.to(device) for image in images)

            detections = model(images)
            # print(detections)
            for item, img_name in zip(detections, image_name):
                # get the image name,bbox,conf,label
                boxes = item['boxes'].cpu().detach().tolist()
                labels = item['labels'].cpu().detach().tolist()
                conf = item['scores'].cpu().detach().tolist()

                # print(img_name)
                with open(os.path.join(OUT_DIR, img_name[:-3] + "txt"), "w") as file:
                    for idx in range(len(labels)):
                        # boxes have the x1,y1, x2, y2 format => converting to yolo format
                        # FASTER-RCNN  keeps one label for background so we need to subtract one from labels
                        width = int(boxes[idx][2] - boxes[idx][0])
                        height = int(boxes[idx][3] - boxes[idx][1])
                        x_center = int((boxes[idx][2] + boxes[idx][0]) / 2)
                        y_center = int((boxes[idx][3] + boxes[idx][1]) / 2)

                        line = list(map(lambda x: str(x), [labels[idx] - 1, x_center, y_center, width, height,
                                                           round(conf[idx], 2), "\n"]))
                        line = (" ").join(line)

                        file.write(line)

            ### Show sample output
            # fig, ax = plt.subplots(2, 2, figsize=(16, 8))
            #
            # #  plot 4 images -> minimum batch number should be 4
            # for idx in range(len(images)):
            #     ax = plt.subplot(2, 2, idx + 1)  # set ax for plotting
            #
            #     # get the image name,bbox,conf,label
            #     boxes = detections[idx]['boxes'].cpu().numpy()
            #     labels = detections[idx]['labels'].cpu().numpy()
            #     conf = detections[idx]['scores'].cpu().numpy()
            #
            #     sample = images[idx].permute(1, 2, 0).cpu().numpy()
            #     sample *= 255
            #     sample = sample.astype(np.uint8)
            #     # print(sample)
            #     sample = Image.fromarray(sample, 'RGB')
            #     for box, label in zip(boxes, labels):
            #         shape = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]
            #         img1 = ImageDraw.Draw(sample)
            #         img1.rectangle(shape, outline="red")
            #
            #     ax.set_axis_off()
            #     ax.imshow(sample)
            #
            # plt.tight_layout()  # reduce margin of subplots
            # plt.show()
