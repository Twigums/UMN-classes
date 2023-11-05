import numpy as np

import torch
import torch.nn as nn

# for saving images
# from PIL import Image

# similar comments with MyMLP.py
class MyCNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride_size, max_pool_size, learning_rate, max_epochs):
        super(MyCNN, self).__init__()
        n = int((np.sqrt(input_size) - kernel_size) / stride_size + 1)
        k = n * n
        hidden_size = 128

        self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = 20,
                    kernel_size = [kernel_size, kernel_size],
                    stride = stride_size
                    # default: dilation = 1, padding = 0, bias = True
                    ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size = [max_pool_size, max_pool_size],
                    # default: stride = kernel_size
                    ),
                nn.Dropout(), # default p = 0.5
                nn.Flatten(),iamges
                nn.Linear(
                    k * 5, # case when kernel for maxpool is 2
                    hidden_size,
                    ),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(
                    hidden_size,
                    output_size,
                    ),
                )

        self.eta = learning_rate
        self.max_epochs = max_epochs

    def fit(self, train_loader, criterion, optimizer):
        for i in range(self.max_epochs):
            epoch_loss = 0
            total_data = 0
            correct_data = 0

            for j, (images, labels) in enumerate(train_loader):

                output = self.forward(images)

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                items = len(train_loader)
                epoch_loss += loss.item() / items

                _, pred = torch.max(output.data, 1)
                total_data += labels.size(0)
                correct_data += (pred == labels).sum().item()



            error_rate = 1 - (correct_data / total_data)

            print("EPOCH " + str(i + 1) + ", (loss, err_rate): (" + str(epoch_loss) + ", " + str(error_rate) + ").")

            # convergence criteria
            if i == 1: # second epoch, let's calculate initial slope over the first 2 epochs
                slope_init = epoch_loss - last_loss

            elif i > 1:
                slope_current = epoch_loss - last_loss

                if abs(slope_init) * 0.1 > abs(slope_current): # defined as 0.1 * initial slope (negligible improvement relative to initial step)
                    print("Convergence criteria met, exiting fit.")
                    return None

            last_loss = epoch_loss

    def predict(self, test_loader, criterion):
        with torch.no_grad(): # no backprop step so turn off gradients
            pred_loss = 0
            total_data = 0
            correct_data = 0

            for j,(images,labels) in enumerate(test_loader):

                # Compute prediction output and loss
                output = self.forward(images)
                loss = criterion(output, labels)

                # Measure loss and error rate and record
                items = len(test_loader)
                pred_loss += loss.item() / items

                # the highest predicted digit
                _, pred = torch.max(output.data, 1)
                total_data += labels.size(0)
                correct_data += (pred == labels).sum().item()

                # for saving miss-classified images
                """
                incorrect = (pred != labels)
                false_ind = np.where(incorrect == False)[0]
                for index in false_ind:
                    image = images[index].cpu().numpy()
                    image_array = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    image_to_save = Image.fromarray(image_array[0], mode = "L")
                    image_to_save.save(f"mnist{j}_{index}.jpg")
                """

        # Print/return test loss and error rate
        error_rate = 1 - (correct_data / total_data)
        print("PREDICTION (loss, err_rate): (" + str(pred_loss) + ", " + str(error_rate) + ").")

    def forward(self, X):
        return self.layers(X)


