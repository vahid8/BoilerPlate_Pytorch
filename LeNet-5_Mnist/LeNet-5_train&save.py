import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time

from helper_functions.evaluation_helper import compute_confusion_matrix, compute_accuracy
from helper_functions.plotting_helper import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_functions.dataloader_helper import get_dataloaders_mnist
from helper_functions.train_helper import train_model

from LeNet_5_architechture import LeNet5

if __name__ == '__main__':


    ##########################
    ### SETTINGS
    ##########################
    BATCH_SIZE = 256
    NUM_EPOCHS = 1
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##########################
    ### DEFINE DATA TRANSFORM
    ##########################
    train_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((32, 32)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])  # normalize to center at zero and between 1 & -1


    train_loader, valid_loader, test_loader = get_dataloaders_mnist(
        batch_size=BATCH_SIZE,
        validation_fraction=0.1,
        train_transforms=train_transform,
        test_transforms=train_transform)

    # Checking the dataset
    for images, labels in train_loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        print('Class labels of 10 examples:', labels[:10])
        break

    ##########################
    ### DEFINE MODEL
    ##########################
    model = LeNet5(grayscale=True, num_classes=10)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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


    model.cpu()
    show_examples(model=model, data_loader=test_loader)

    class_dict = {0: '0',
                  1: '1',
                  2: '2',
                  3: '3',
                  4: '4',
                  5: '5',
                  6: '6',
                  7: '7',
                  8: '8',
                  9: '9'}

    mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
    plot_confusion_matrix(mat, class_names=class_dict.values())
    plt.show()

    # Saving the model
    torch.save(model.state_dict(), 'my_trained_model/my_LeNet5_model.pt')
    torch.save(optimizer.state_dict(), 'my_trained_model/my_LeNet5_optimizer.pt')
    torch.save(model.state_dict(), 'my_trained_model/my_LeNet5_scheduler.pt')

