import torch
import torch.nn as nn           # torch and torch.nn - indispensable packages for neural networks
import torch.optim as optim     # for efficient gradient descents
import torch.nn.functional as F

from PIL import Image   # to load the image

import torchvision.transforms as transforms     # to transform PIL images into tensors
import torchvision.models as models             # to train or load pre-trained models : vgg19
from torchvision.utils import save_image        # to store the generated image

# to provide us with the conv layers
# model = models.vgg19(pretrained=True).features
# print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
image_size = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(image_size),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def load_image(image_name):
    # using the PIL library
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    # unsqueeze 0 to add additional dimension for the batch size (1)
    image = loader(image).unsqueeze(0)
    return image.to(device)

content_image = load_image("grace.png")
style_image = load_image("georgia.jpg")

assert style_image.size() == content_image.size(), \
    "style and content images of the same size (width and height) need to be imported"

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # from print(model), we take each Conv2d layer (numbered) that comes after MaxPool2d
        # in notation, they should be (conv 1-1, 2-1, 3-1, 4-1, 5-1)
        # these layers are inserted into the loss function
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]    # up to 29, inclusive to 28

    def forward(self, x):
        features = []
        
        for layer_num, layer in enumerate(self.model):
            # after the output from the conv layer
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

model = VGG().to(device).eval() 

# a copy of the original image works faster/better
generated_image = content_image.clone().requires_grad_(True)
# requires_grad_(True) is essential because it freezes the network 
# so that the only thing that can change is the generated image

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # 'detach' the target content from the tree used to dynamically compute the gradient: 
        # this is a stated value,not a variable. Otherwise the forward method of the criterion
        # will throw an error
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class gram_matrix(nn.Module):
    def forward(self, input):
        b, c, w, h = input.size()
        F = input.view(b, c, h * w)
        # gram matrix is computed by multiplying the input but its transpose
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h*w)
        return G

class gram_mse_loss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(gram_matrix()(input), target)
        return out

# hyperparameters
total_steps = 1000
learning_rate = 0.001
content_weight = 0.5   # parameters that will be multiplied by the respective content and style losses
style_weight = 0.1     # how much style we want in the image
optimizer = optim.Adam([generated_image], lr=learning_rate)

content_loss = 0
style_loss = 0

for step in range(total_steps):     # how many times the image will be modified
    # send each of the 3 images through the vgg network
    generated_features = model(generated_image)
    content_image_features = model(content_image)
    style_image_features = model(style_image)
    # will receive a list containing the output from 5 different conv layers

    content_loss = 0
    style_loss = 0

    # iterate through all of the features for the chosen layers
    for generated_feature, content_feature, style_feature in zip(
        # everything from the 5 conv layers
        # taking conv 1-1, 2-1, 3-1, 4-1, 5-1 from below features
        generated_features, content_image_features, style_image_features
    ):
        # changes for each block layer we're looking at, so it is important to take it for every block
        batch_size, channel, height, width = generated_feature.shape
        content_loss += torch.mean((generated_feature - content_feature) ** 2)

        """compute gram matrix"""
        # the shape is batch size, channel, height and width, but the batch size is just 1 (sending 1 image)

        # multiplying every pixel value from each channel with every other channel for the generated features
        # and will end up having shaped channel by channel 

        # can be viewed as the gram matrix calculating some sort of correlation matrix 
        # if the pixel colours are similar across the channels of the generated image and the style image,
        # then that results in the two pictures having similar style
        G = generated_feature.view(channel, height * width).mm(   # mm = matrix multiply
            generated_feature.view(channel, height * width).t()   # t = transpose
        )

        A = style_feature.view(channel, height * width).mm(     
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated_image, "generated_image.jpg")
