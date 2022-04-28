import cv2
import numpy as np
from tqdm import tqdm
from dataloader import A2D2_Dataset
import torch
import wandb
import multiprocessing
from torchinfo import summary
from alive_progress import alive_bar

from models.UNet import UNet

if __name__ == "__main__":

    wandb.init(project="road-segmentation-pytorch", config={})

    # Setting up the system for cude if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    # Constant
    BATCH_SIZE = 8
    IMG_SIZE = (512, 512)  # 424 ?
    LEARNING_RATE = 0.001
    WORKERS = multiprocessing.cpu_count()

    # Open datasets
    train_dataset = A2D2_Dataset("training", size=IMG_SIZE)
    test_dataset = A2D2_Dataset("testing", size=IMG_SIZE)
    val_dataset = A2D2_Dataset("validation", size=IMG_SIZE)

    # Define data loader
    data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=WORKERS)
    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Create model
    model = UNet(3, train_dataset.classes())
    model.to(device)

    wandb.watch(model, log_freq=100)

    # Summary
    summary(model, input_size=(BATCH_SIZE, 3) + IMG_SIZE)

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loop over the dataset multiple times
    for epoch in range(50):

        ########################
        #       TRAINING       #
        ########################

        with alive_bar(len(data_loader_train), title="Epoch : " + str(epoch) + " - Training") as bar:

            # Set the model for training
            model.train()
            training_loss = 0.0

            # Iterate trough the batches of the dataloader
            for i, data in enumerate(data_loader_train, 1):

                # get the inputs; data is a list of [inputs, targets] to the correct device
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                # Clear the gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward Pass
                outputs = model(inputs)

                # Find the Loss
                loss = criterion(outputs, targets)

                # Calculate gradients
                loss.backward()

                # Update Weights
                optimizer.step()

                # Calculate Loss
                training_loss += loss.item()

                # Update the progress bar and Weight & Biases
                bar.text(f'Loss: {training_loss / i:.3f}')
                bar()

        ########################
        #      VALIDATION      #
        ########################

        with alive_bar(len(data_loader_val), title="Epoch : " + str(epoch) + " - Validation") as bar:

            # Set the model for infering
            model.eval()
            validation_loss = 0.0

            # Disabling th gradient
            with torch.no_grad():
                # Iterate trough the batches of the dataloader
                for i, data in enumerate(data_loader_val, 1):

                    # get the inputs; data is a list of [inputs, targets] to the correct device
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward Pass
                    outputs = model(inputs)

                    # Find the Loss
                    val_loss = criterion(outputs, targets)

                    # Calculate Loss
                    validation_loss += val_loss.item()

                    # Update the progress bar and Weight & Biases
                    bar.text(f'Loss: {validation_loss / i:.3f}')
                    bar()

        ########################
        #        LOGGING       #
        ########################

        # Average of loss
        training_loss = training_loss / len(data_loader_train)
        validation_loss = validation_loss / len(data_loader_val)

        # Logging
        print(f"Epoch : {epoch} - Training loss {training_loss:.3f} - Validation loss {validation_loss:.3f}\n")
        wandb.log({
            "epoch": epoch,
            "validation_loss": validation_loss,
            "training_loss": training_loss
        })

        # Save the model
        torch.save(model.state_dict(), f'./unet_epoch-{epoch}.pth')


