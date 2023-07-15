import torch 
from torchvision.datasets import ImageFolder 
from torchvision import transforms 
from torch.utils.data import DataLoader
from PIL import Image 
import os 
import numpy as np 
from datetime import datetime 

from loss import LossFn 
from model import ResNet
from tqdm import tqdm
import matplotlib.pyplot as plt 

import warnings 
warnings.filterwarnings("ignore")

class Generate():

    def __init__(self, dev):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = dev


    def init_dataset(self, dataset_path, style_path, batch_size=1, image_size=256):
        print('Loading dataset...')
        self.batch_size = batch_size
        self.dataset_transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x : x.mul(255))
        ])
        folder = ImageFolder(dataset_path, transform=self.dataset_transform)
        self.loader = DataLoader(folder, batch_size=batch_size, shuffle=True)

        self.style_transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.mul(255))
                ])
        self.style_img = self.style_transform(Image.open(style_path)).to(self.device)

        self.test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.mul(255))
                ])


    def init_weights(self, m):
        if type(m) == torch.nn.Conv2d :
            torch.nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)


    def init_model(self, weight_path=None, num_residual_layers=5):
        self.num_residual_layers = num_residual_layers
        print('Loading model...')
        self.model = ResNet(num_residual_layers)
        self.model = self.model.to(self.device)
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        else : 
            self.model.apply(self.init_weights)


    def deprocess(self, data):
        data = data.to('cpu')
        img = data.clone().clamp(0,255).detach().numpy()
        img = img.transpose(1,2,0).astype('uint8')
        img = Image.fromarray(img)
        return img


    def generate_samples(self,samples_path):
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15,7))
        ax = ax.flatten()

        self.model.eval()
        sample_paths = [os.path.join(samples_path,i) for i in os.listdir(samples_path)]
        sample_images = [self.test_transform(Image.open(i)) for i in sample_paths]
        sample_images = torch.stack(sample_images, dim=0).to(self.device)
        with torch.no_grad(): 
            output_images = self.model(sample_images)
            for axis, img in zip(ax, output_images):
                axis.imshow(self.deprocess(img))
        return fig
                
    
    def train(self, lr, epochs, style_weight, content_weight, save_path, samples_path, save_every=10, generate_every=10, content_loss_layer=0):
        print('Starting training...')
        crit = LossFn(self.style_img, content_weight, style_weight, self.batch_size, self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        global_step = 0
        images, _ = next(iter(self.loader))

        for e in range(1,epochs+1):
            loop = tqdm(enumerate(self.loader), total=len(self.loader), leave=False, position=0)
            
            for batch_id, (input_imgs, _) in loop : 
                if input_imgs.shape[0] == self.batch_size :
                    self.model.train()
                    input_imgs = input_imgs.to(self.device)
                    opt.zero_grad()
                    output_imgs = self.model(input_imgs)
                    loss, c_loss, s_loss = crit.calc_loss(output_imgs, input_imgs, content_loss_layer=content_loss_layer)
                    loss.backward(retain_graph=True)
                    opt.step()

                loop.set_description(f'Epoch : [{e}/{epochs}]')
                loop.set_postfix(loss=loss.item())

                if batch_id % generate_every == 0 : 
                    fig = self.generate_samples(samples_path)
                    fig.savefig('sample_output.png')
                    plt.close(fig)
                
                if batch_id % save_every == 0 : 
                    torch.save(self.model.state_dict(), os.path.join(save_path, f'{e}.pt'))

