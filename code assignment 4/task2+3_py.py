# %%
# a basic CNN classifies CIFAR-10
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu' #toggle on/off for seeing difference between cpu and gpu performance
print(device)

# %%
# loading the data
batch_size = 4

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
# functions to show an image
def imshow(img, title = None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# %%
# defining the CNN
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 is since input is 3 dimensinal - RGB image
        self.pool = nn.MaxPool2d(2, 2, return_indices=True) # indices used for unpooling
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes)) # 10 since we have 10 classes

        # define deconvoloution
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5) # taking the 16 channels from conv2 as input and output 6 
        self.deconv1 = nn.ConvTranspose2d(6, 3, 5)  # take the 6 channels from deconv2 and output 3 as an image - RGB
        self.unpool = nn.MaxUnpool2d(2, 2) # same unpool

    def forward(self, x):
        x, indices1  = self.pool(F.relu(self.conv1(x))) # conv1 -> relu -> maxpool -> z1
        z1 =[ x.clone(), indices1 ] # get first latent features
        x, indices2  = self.pool(F.relu(self.conv2(x))) # conv2 -> relu -> maxpool -> z2
        z2 = [ x.clone(), indices2 ] # get second latent features

        # deconvolution branch - recconstruct
        x_wave = x.clone()
        x_wave = self.deconv2(F.relu(self.unpool(x_wave, indices2)))
        x_wave = self.deconv1(F.relu(self.unpool(x_wave, indices1)))
        
        # regular NN branch
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x, x_wave, z1, z2

net = Net().to(device)

# %%
import torch.optim as optim
# defining loss and optimizer
criterion = nn.CrossEntropyLoss()

def new_criterion(outputs, labels, deconv_outputs, inputs):
    ce_criterion = nn.CrossEntropyLoss()
    ce_loss = ce_criterion(outputs, labels)
    rec_loss = 0
    for i in range(3):
        rec_loss += torch.mean(torch.pow(inputs[:, i, :, :] - deconv_outputs[:, i, :, :], 2)) # |x_wave_i-x_i|^2 and then MSE between them, on i's channel

    rec_loss /= 3

    lambda_sc = 1

    return ce_loss + lambda_sc*rec_loss

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
# training the model
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs, deconv_output, _,_ = net(inputs)
        # using the new criterion created
        loss = new_criterion(outputs, labels, deconv_output, inputs)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# %%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %%
dataiter = iter(testloader)
images, labels = next(dataiter)

_, deconv_output,_,_ = net(images.to(device))

# print images
imshow(torchvision.utils.make_grid(images), "Original Images")
imshow(torchvision.utils.make_grid(deconv_output.cpu()), "Deconv Images")

# %%
outputs,_ ,_,_= net(images.to(device))
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

# %%
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs ,_,_,_ = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# %%
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs ,_ ,_,_= net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# %% [markdown]
# *Task 3*

# %%
# Task 3
def reconstruct_image(z,indices, unpool, deconv):
    # Reconstruct the image
    z_reconstructed = z.clone()
    with torch.no_grad():
        z_reconstructed = deconv(F.relu(unpool(z_reconstructed, indices)))

    return z_reconstructed

# %%
def turn_on_only_feature_i(z, i):
    # Clone the input tensor to avoid modifying the original tensor
    z_zeroed = z.clone()
    
    # Create a zero tensor with the same shape as the input tensor
    z_zeroed = torch.zeros_like(z_zeroed)
    
    # Copy the specified channel to the zero tensor
    z_zeroed[i, :, :] = z[i, :, :]
    
    return z_zeroed

# %%
dataiter_test = iter(testloader)
dataiter_train = iter(trainloader)

# %%
def get_latent_features(net, device, num_of_features, images, title, is_z2=False):
    # Pass the test image through the model
    with torch.no_grad():
        _, _, z1, z2 = net(images.to(device))  # Get the model's output for the test image

    if (not is_z2):
        imshow(torchvision.utils.make_grid(images[0]), title)

    z1, z1_indices = z1
    z2, z2_indices = z2

    # only one image
    z1 = z1[0]
    z2 = z2[0]

    if (is_z2):
        z = z2
    else:
        z = z1

    reconstructed_images = []
    for i in range(num_of_features):
        z_zeroed = turn_on_only_feature_i(z,i)
        if (is_z2):
          z_zeroed = reconstruct_image(z_zeroed, z2_indices[0,:,:,:], net.unpool, net.deconv2)
        z_reconstructed = reconstruct_image(z_zeroed, z1_indices[0,:,:,:], net.unpool, net.deconv1)
        reconstructed_images.append(z_reconstructed.detach().cpu())
    
    if is_z2:
        title = "z2 reconstructed"
    else:
        title = "z1 reconstructed"

    imshow(torchvision.utils.make_grid(reconstructed_images),title)


# %%
# Select one image from the test set
images_test, _ = next(dataiter_test)
images_train, _ = next(dataiter_train)

# for train
get_latent_features(net, device, 6, images_train,"Original Train Image")
get_latent_features(net, device, 3, images_train, "Original Train Image", is_z2=True)

# for test
get_latent_features(net, device, 6, images_test, "Original Test Image")
get_latent_features(net, device, 3, images_test, "Original Test Image", is_z2=True)


