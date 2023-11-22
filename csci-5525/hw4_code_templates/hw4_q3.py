################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyGenerator import MyGenerator
from MyDiscriminator import MyDiscriminator

from hw4_utils import load_MNIST

np.random.seed(2023)

batch_size = 128

normalize_vals = (0.5, 0.5)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

# plot helper function to plot 5 images
def plot_five(i, images):
    fig, axes = plt.subplots(1, 5, figsize = (15, 3))

    for j in range(5):
        plot_image = plot_images[j].squeeze().detach().cpu().numpy()
        plot_image = plot_image.reshape(28, 28)
        axes[j].imshow(plot_image, cmap = "gray")
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(str(i) + ".jpg")
    plt.clf()

torch.manual_seed(2023)

learning_rate = 0.0002
k_steps = 1
generator_input_dims = 128
max_epochs = 40

generator = MyGenerator()
discriminator = MyDiscriminator()

criterion = nn.BCELoss()
optimizer_gen = torch.optim.Adam(generator.parameters(), lr = learning_rate)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr = learning_rate)

epoch_loss_gen = []
epoch_loss_dis = []

converged_gen = False
converged_dis = False

for i in range(max_epochs):
    loss_gen = 0
    loss_dis = 0

    for j, (images, labels) in enumerate(train_loader):
        items = len(train_loader)
        num_labels = len(labels)

        gt_real = torch.ones((num_labels, )).unsqueeze(1)
        gt_fake = torch.zeros((num_labels, )).unsqueeze(1)

        optimizer_dis.zero_grad()
        optimizer_gen.zero_grad()

        # discriminator step: feed real, then feed fake
        loss_dis_real = discriminator.get_losses(images, gt_real, criterion)

        input_gen = torch.randn(num_labels, generator_input_dims)
        output_gen = generator.forward(input_gen).detach()

        loss_dis_fake = discriminator.get_losses(output_gen, gt_fake, criterion)

        optimizer_dis.step()

        # generator step: feed discriminator output
        input_gen_update = torch.randn(num_labels, generator_input_dims)
        output_gen_update = generator.forward(input_gen_update)
        output_dis_update = discriminator.forward(output_gen_update)

        loss_gen_update = criterion(output_dis_update, gt_real)

        loss_gen_update.backward()
        optimizer_gen.step()

        # respective losses
        loss_gen += loss_gen_update.item() / items
        loss_dis += (loss_dis_real.item() + loss_dis_fake.item()) / items

    epoch_loss_gen.append(loss_gen)
    epoch_loss_dis.append(loss_dis)

    plot_input = torch.randn(5, generator_input_dims)
    plot_images = generator.forward(plot_input)

    plot_five(i, plot_images)

    print("EPOCH " + str(i + 1) + ", (loss_gen, loss_dis): (" + str(loss_gen) + ", " + str(loss_dis) + ").")

    # convergence criteria
    if i > 10:
        slope_gen = loss_gen - last_loss_gen
        slope_dis = loss_dis - last_loss_dis

        if slope_gen > 0:
            converged_gen = True

        if slope_dis > 0:
            converged_dis = True

        if converged_gen == True and converged_dis == True:
            print("Convergence criteria met, exiting fit.")
            break

    last_loss_gen = loss_gen
    last_loss_dis = loss_dis

final_input = torch.randn(num_labels, generator_input_dims)
final_output = generator.forward(input_gen_update)

plot_five("final", final_output)
