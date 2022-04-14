import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time

from helper_functions.evaluation_helper import compute_confusion_matrix, compute_accuracy
from helper_functions.plotting_helper import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_functions.dataloader_helper import get_dataloaders_cifar10, UnNormalize
from helper_functions.train_helper import train_model

from AlexNet_architechture import AlexNet


if __name__ == '__main__':
    ##########################
    ### SETTINGS
    ##########################
    BATCH_SIZE = 256
    NUM_EPOCHS = 200
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##########################
    ### DEFINE DATA TRANSFORM
    ##########################
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),  # cifar images are 32*32 we make it larger otherwise we get problem with the dimensions at final layers
        torchvision.transforms.RandomCrop((64, 64)),  # to make model more robust against the over fitting
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                          ])  # normalizing the color channels

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((70, 70)),
        torchvision.transforms.CenterCrop((64, 64)), # we dont want random crop on test or validation data
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ##########################
    ### LOAD DATA
    ##########################
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=BATCH_SIZE,
        validation_fraction=0.1,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        num_workers=2)

    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        print('Class labels of 10 examples:', labels[:10])
        break

    ##########################
    ### DEFINE MODEL
    ##########################
    model = AlexNet(num_classes=10)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.1,
                                                           mode='max',
                                                           verbose=True)

    ##########################
    ### TRAIN MODEL
    ##########################
    minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
        model=model,
        num_epochs=NUM_EPOCHS,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=DEVICE,
        scheduler=scheduler,
        scheduler_on='valid_acc',
        logging_interval=100)

    ##########################
    ### VISUALIZE RESULTS
    ##########################
    plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                       num_epochs=NUM_EPOCHS,
                       iter_per_epoch=len(train_loader),
                       results_dir=None,
                       averaging_iterations=200)
    plt.show()

    plot_accuracy(train_acc_list=train_acc_list,
                  valid_acc_list=valid_acc_list,
                  results_dir=None)
    plt.ylim([60, 100])
    plt.show()

    model.cpu()
    unnormalizer = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # un normalize images to plot them
    class_dict = {0: 'airplane',
                  1: 'automobile',
                  2: 'bird',
                  3: 'cat',
                  4: 'deer',
                  5: 'dog',
                  6: 'frog',
                  7: 'horse',
                  8: 'ship',
                  9: 'truck'}

    show_examples(model=model, data_loader=test_loader, unnormalizer=unnormalizer, class_dict=class_dict)

    mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
    plot_confusion_matrix(mat, class_names=class_dict.values())
    plt.show()

    # Saving the model
    torch.save(model.state_dict(), 'my_trained_model/my_AlexNet_model.pt')
    torch.save(optimizer.state_dict(), 'my_trained_model/my_AlexNet_optimizer.pt')
    torch.save(model.state_dict(), 'my_trained_model/my_AlexNet_scheduler.pt')

