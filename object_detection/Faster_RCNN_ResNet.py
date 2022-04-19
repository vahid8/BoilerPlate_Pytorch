import os
import time
import tqdm
import torch
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN

from helper_functions.ObjectDetectionVal import bboxEvaluation
from helper_functions.dataLoader_with_albumentation import TrainDataset



if __name__ == '__main__':
    ##########################
    ### SETTINGS
    ##########################
    BATCH_SIZE = 8
    NUM_EPOCHS = 7
    NUM_CLASSES = 3
    IMG_SIZE = 1280
    pretraide_model = False

    DIR_INPUT = '/home/tower/Codes/face_plate_data'
    DIR_OUT = "/home/tower/Codes/faster_out_test"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------
    # Initialize our custom data loader
    # -------------------------------------------------
    train_dataset = TrainDataset(os.path.join(DIR_INPUT, 'train', 'labels'),
                                 os.path.join(DIR_INPUT, 'train', 'images'),
                                 IMG_SIZE)

    val_dataset = TrainDataset(os.path.join(DIR_INPUT, 'val', 'labels'),
                               os.path.join(DIR_INPUT, 'val', 'images'),
                               IMG_SIZE)

    print(f'length of train dataset {len(train_dataset)}')
    print(f'length of val dataset {len(val_dataset)}')
    print("-" * 50)

    # Load dataset to the torch using its DataLoader function
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              collate_fn=TrainDataset.collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                            collate_fn=TrainDataset.collate_fn)

    # /////////////////// Show one sample of training data
    images, targets, _ = next(iter(train_loader))
    images = list(image.to(device) for image in images)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    fig, ax = plt.subplots(2, 2, figsize=(16, 8))

    #  plot 4 images -> minimum batch number should be 4
    for idx in range(4):
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
                                 color=(125, 246, 55), thickness=2, lineType=cv2.LINE_AA)

        ax.set_axis_off()
        ax.imshow(sample)

    plt.tight_layout()  # reduce margin of subplots
    plt.show()

    ##########################
    ### DEFINE MODEL
    ##########################
    # load a mcodel; pre-trained on COCO
    backbone = resnet_fpn_backbone('resnet152', pretrained=True)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    for param_group in optimizer.param_groups:
        print(f"current learning rate {param_group['lr']}")

    # Intialize evaluator
    evaluator = bboxEvaluation(os.path.join(DIR_INPUT, 'val', 'labels'),
                               os.path.join(DIR_INPUT, 'val', 'images'))
    # # check trained model:
    # ### loading the model
    # model.load_state_dict(torch.load('test_model/my_ResNet_model.pt'))
    # model.eval()
    if pretraide_model:
        # checkpoint_path = config["DIR_OUT"] + pretrained_name
        # checkpoint = torch.load(checkpoint_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        model.load_state_dict(torch.load('test_model/my_ResNet_model.pt'))
        optimizer.load_state_dict(torch.load('test_model/my_ResNet_optimizer.pt'))
        lr_scheduler.load_state_dict(torch.load('test_model/my_ResNet_scheduler.pt'))

    ##########################
    ### TRAIN THE MODEL
    ##########################
    logging_interval = 50
    train_losses = []
    validation_losses = []
    itr = 1
    checkpoint_num = 0
    step_num = 0
    minibatch_loss_list = []
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()  # So it uses training mode
        epoch_loss = 0

        for batch_idx, (images, targets, image_names) in enumerate(train_loader):
            step_num += 1
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Forward Pass and calculate loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # Clear the gradients
            optimizer.zero_grad()
            # Calculate gradients
            losses.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            # save epoch loss for final plot
            epoch_loss += losses.item()

            # ## LOGGING for mini batches at intervals
            minibatch_loss_list.append(losses.item())
            if not batch_idx % logging_interval:
                itr += 1
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {losses:.4f}')



        # Saving the model
        torch.save(model.state_dict(), 'test_model/my_ResNet_model.pt')
        torch.save(optimizer.state_dict(), 'test_model/my_ResNet_optimizer.pt')
        torch.save(lr_scheduler.state_dict(), 'test_model/my_ResNet_scheduler.pt')

        # END OF TRAINING ONE EPOCH
        if lr_scheduler is not None:
            lr_scheduler.step()

        # EVALUATE THE MODEL FOR THE CUREENT EPOCH
        # print(f'step num {step_num}, len(train_dataset) {len(train_dataset)} ')
        train_losses.append(epoch_loss / len(train_dataset))

        epoch_loss = 0.0
        step_num = 0
        data = []
        model.eval()  # activate interface mode of the model (batchnorm or dropout layers will work in eval mode instead of training mode.)

        with torch.no_grad():  # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop
            for images, targets, image_name in tqdm.tqdm(val_loader,
                                                    desc="validation " + str(epoch) + "/" + str(NUM_EPOCHS)):
                step_num += 1
                images = list(image.to(device) for image in images)
                # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # Forward Pass and calculate loss
                # val_loss_dict = model(images, targets)
                # val_losses = sum(loss for loss in val_loss_dict.values())
                # epoch_loss += val_losses.item()
                detections = model(images)
                for item, img in zip(detections, image_name):
                    # get the image name,bbox,conf,label
                    boxes = item['boxes'].cpu().detach().tolist()
                    labels = item['labels'].cpu().detach().tolist()
                    conf = item['scores'].cpu().detach().tolist()
                    for idx in range(len(labels)):
                        # boxes have the x1,y1, x2, y2 format
                        # FASTER-RCNN  keeps one label for background so we need to subtract one from labels
                        data.append([img[:-3] + "txt", labels[idx] - 1, [b / IMG_SIZE for b in boxes[idx]], conf[idx]])

        predicted_df = pd.DataFrame(data, columns=["image_name", "label", "bbox", "conf"])
        evaluator.calc_mAP(predicted_df, print_details=True)
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch training time {elapsed}")

        # print(f'step num {step_num}, len(val_dataset) {len(val_dataset)}')
        # validation_losses.append(epoch_loss/ len(val_dataset))

        # print(f"Epoch #{epoch}, train loss: {train_losses[-1]}, validation loss: {validation_losses[-1]}")

        if epoch > 0 and epoch % 3 == 0:  # Save checkpoint each --- iteration
            print("saving checkpoint")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, DIR_OUT + "/checkpoint_" + str(checkpoint_num) + ".pt")
            checkpoint_num += 1


    # save last checkpoint
    print("saving checkpoint")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, DIR_OUT + "/checkpoint_last.pt")

    # Save the last model
    torch.save(model.state_dict(), DIR_OUT + '/last.pth')


