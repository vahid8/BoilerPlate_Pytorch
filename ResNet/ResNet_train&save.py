import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from helper_functions.evaluation_helper import compute_confusion_matrix, compute_accuracy
from helper_functions.plotting_helper import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_functions.dataloader_helper import get_dataloaders_mnist
from helper_functions.train_helper import train_model


if __name__ == '__main__':
    ##########################
    ### SETTINGS
    ##########################
    BATCH_SIZE = 8
    NUM_EPOCHS = 15
    NUM_CLASSES = 8
    data_dir = 'data/street_types2'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##########################
    ### DEFINE DATA TRANSFORM
    ##########################
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # TO do: test data loaders
    # for i, (features, targets) in enumerate(valid_loader):
        # features = features
        # targets = targets.float()

    # train_loader, valid_loader, test_loader = get_dataloaders_mnist(
    #     batch_size=BATCH_SIZE,
    #     validation_fraction=0.1,
    #     train_transforms=train_transform,
    #     test_transforms=train_transform)

    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        print('Class labels of 10 examples:', labels[:10])
        break

    ##########################
    ### DEFINE MODEL
    ##########################
    model = torchvision.models.resnet18(pretrained=True)
    # replace the last layer output with our number of classes
    _in_features_num = model.fc.in_features
    model.fc = torch.nn.Linear(_in_features_num, NUM_CLASSES)

    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    ##########################
    ### TRAIN MODEL
    ##########################
    minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
        model=model,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=None,
        optimizer=optimizer,
        device=DEVICE,
        logging_interval=100)

    ##########################
    ### VISUALIZE RESULTS
    ##########################
    plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                       num_epochs=NUM_EPOCHS,
                       iter_per_epoch=len(train_loader),
                       results_dir=None,
                       averaging_iterations=100)
    plt.show()

    plot_accuracy(train_acc_list=train_acc_list,
                  valid_acc_list=valid_acc_list,
                  results_dir=None)
    plt.ylim([80, 100])
    plt.show()

    # Saving the model
    torch.save(model.state_dict(), 'my_trained_model/my_ResNet_model.pt')
    torch.save(optimizer.state_dict(), 'my_trained_model/my_ResNet_optimizer.pt')
    torch.save(scheduler.state_dict(), 'my_trained_model/my_ResNet_scheduler.pt')
    #
    ### loading the model
    # model = torchvision.models.resnet18(pretrained=True)
    # model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    # model.load_state_dict(torch.load('my_trained_model/my_ResNet_model.pt'))
    # model.eval()
    # model = model.to('cpu')

    model.cpu()

    class_dict = {0: '0',
                  1: '1',
                  2: '2',
                  3: '3',
                  4: '4',
                  5: '5',
                  6: '6',
                  7: '7'}

    show_examples(model, valid_loader, unnormalizer=True, class_dict=class_dict)


    mat = compute_confusion_matrix(model=model, data_loader=valid_loader, device=torch.device('cpu'))
    plot_confusion_matrix(mat, class_names=class_dict.values())
    plt.show()

