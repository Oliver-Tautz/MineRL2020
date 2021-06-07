# Pytorch
import torch
import numpy as np
# Progressbar
from tqdm import tqdm

IMG_SCALE = 1. / 255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


def prepare_img(img):
    new_img = np.zeros((len(img), 3, 64, 64), dtype=np.double)
    for index in range(len(img)):
        #print(np.shape(img[index].numpy()))
        calc = (img[index].numpy() * IMG_SCALE - IMG_MEAN) / IMG_STD
        calct = calc.transpose(2, 0, 1)
        new_img[index] = calct

    return new_img


def prepare_target(img):
    new_img = np.zeros((8, 8, 64, 64), dtype=np.double)
    for index in range(len(img)):
        for row in range(len(img[index])):
            for col in range(len(img[index][row])):
                # print(img[index][row][col])
                new_img[index][img[index][row][col]][row][col] = 1

    # print(np.shape(((img[index] * IMG_SCALE - IMG_MEAN) / IMG_STD)))
    # new_img[index][0] = ((img[index] * IMG_SCALE - IMG_MEAN) / IMG_STD).transpose(2, 0)

    # print(np.shape(new_img))

    return new_img


# Training
def train(model, dataloader, criterion, optimizer, num_epochs=20,
          device=torch.device('cpu'), verbose=False, progress=True):
    # Logging of training progress
    log = {
        'loss': []
    }

    # Main Training loop for num_epochs
    for epoch in range(num_epochs):
        # Data (with optional progressbar)
        data = dataloader

        # Progress update if requested
        if progress:
            # Enumerate epochs
            print(f'Epoch {epoch}/{num_epochs}')
            # Wrap dataloader with progressbar
            data = tqdm(dataloader)

        # Iterate dataset
        for (inputs, targets) in data:
            # Move data to device used for training
            inputs = torch.tensor(prepare_img(inputs), dtype=torch.float)
            inputs = inputs.to(device)

            # print("targets", np.shape(targets))
            # print("targets", np.shape(prepare_target(targets)))

            # print(torch.max(targets), torch.min(targets))

            targets = targets.long().to(device)

            # print(np.shape(targets))

            # Clear optimizer gradients
            optimizer.zero_grad()

            # Enable gradients computation (needed for backward step)
            with torch.set_grad_enabled(True):
                # Forward pass
                outputs = model(inputs)

                # Apply sofmax over output to match targets (maps from channel
                # per class to one channel with maximum prediction)
                outputs = torch.nn.LogSoftmax(dim=1)(outputs['out'])

                # print(np.shape(outputs), np.shape(targets))
                # Loss computation
                loss = criterion(outputs, targets)

                # Print loss to track training progress
                if verbose:
                    print(loss.item())

                # Training step
                loss.backward()
                optimizer.step()

                # Logging
                log['loss'].append(loss.item())

    # Return trained model and loss history
    return model, log 
