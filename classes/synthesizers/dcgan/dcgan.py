import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from IPython.display import HTML
from classes.base.namespaces import ExperimentDict

from classes.helper.utils import Utils
from classes.helper.datasets import Datasets
from classes.base.enums import DatasetsNames, OptimizerNames

from .generator import Generator
from .discriminator import Discriminator

import torch
torch.manual_seed(42)
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils

class DCGan():
    def __init__(self,
                 exp_dict: ExperimentDict,
                 g_optimizer_type: OptimizerNames,
                 d_optimizer_type: OptimizerNames,
                 lr: float,
                 paths_dict: dict):
        
        self.g_optimizer_type = g_optimizer_type
        self.d_optimizer_type = d_optimizer_type
        self.lr = lr

        self.data_type = exp_dict.dataset.type
        self.specialist_class = exp_dict.dataset.class_number

        self.datapath = paths_dict["data"]

        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.loss_func = None

        self.num_epochs = 100
        self.batch_size = 32
        self.ngpu = 1
        self.nchannel = 3
        self.nlatent_space = 100
        self.data_dim = 1024 #ref: tamanho da img 32x32

        self.device = Utils.get_device_available()
        print(f"O processamento ocorrerá utilizando: {self.device}.")

    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def setup_network(self):
        # Create the Gerador
        self.generator = Generator(nz = self.nlatent_space, nc = self.nchannel).to(self.device)
        self.generator.apply(self.weights_init)

        # Create the Discriminator
        self.discriminator = Discriminator(nc = self.nchannel).to(self.device)
        self.discriminator.apply(self.weights_init)

    def setup_optimizer_loss(self):
        # Initialize the loss function
        self.loss_func = nn.BCELoss()     

        # Setup Adam optimizers for both G and D
        if self.g_optimizer_type == OptimizerNames.ADAM:
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        if self.g_optimizer_type == OptimizerNames.SGD:
            self.g_optimizer == optim.SGD(self.generator.parameters(), lr = self.lr, momentum = 0.9)

        if self.d_optimizer_type == OptimizerNames.ADAM:
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        if self.d_optimizer_type == OptimizerNames.SGD:
            self.d_optimizer == optim.SGD(self.discriminator.parameters(), lr = self.lr, momentum = 0.9)

    def vector_to_images(self, vectors, dsize: tuple = (32, 32)):
        return vectors.view(vectors.size(0), self.nchannel, dsize[0], dsize[1])
    
    def save(self, model, filename: str = 'model_gan.pt'):
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(f'./dcgan/models/{filename}')

    def load(self, model_path: str):
        self.generator = torch.jit.load(model_path)

    def fit(self):
        if self.data_type == DatasetsNames.CIFAR10:
            dataloader = Datasets.get_cifar_as_dataloaders(datapath = self.datapath, 
                                                           specialist_class = self.specialist_class)
        else:
            raise Exception("data_type não implementado ainda.")

        self.nchannel = next(iter(dataloader))[0].size(1)
        self.setup_network()
        self.setup_optimizer_loss()

        # Create batch of latent vectors that we will use to visualize the progression of the Gerador
        fixed_noise = torch.randn(self.batch_size, self.nlatent_space, 1, 1, device = self.device)
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype = torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.loss_func(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nlatent_space, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.loss_func(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.d_optimizer.step()

                ############################
                # (2) Update G network: minimize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                label.fill_(real_label)  # fake labels are real for Gerador cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.loss_func(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.g_optimizer.step()

                ## Output training stats
                # if i % 50 == 0:
                #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                #         % (epoch, nepochs, i, len(dataloader),
                #             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                ## Check how the Gerador is doing by saving G's output on fixed_noise
                # if (iters % 500 == 0) or ((epoch == nepochs-1) and (i == len(dataloader)-1)):
                #     with torch.no_grad():
                #         fake = self.generator(fixed_noise).detach().cpu()
                #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                # iters += 1
        
        # self.save(model = self.generator, filename = f'model_dcgan_c_{specialist_class}.pt')
        return G_losses[-1]

    def generate_sinthetic_images(self, specialist_class_number: int, epoch: int, dirname: str):
        ## Verificando se há um modelo instanciado ##
        if self.generator is None:
            return "Necessário carregar um modelo previamente. Utilize a função load()."
        
        base_dir = './dcgan/results'
        dirpath = Utils.criar_pasta(path = base_dir, name = dirname)
        imgpath = Utils.criar_pasta(path = dirpath, name = str(specialist_class_number))

        ## Gerando imagens sintéticas ##
        # Carregando ruido aleatório   
        latent_space_samples = torch.randn(self.batch_size, self.nlatent_space, 1, 1).to(self.device)
        # Sintetizando
        self.generator.eval()
        generated_samples = self.generator(latent_space_samples)
        # Convertendo vetores para imagens
        generated_imgs = self.vector_to_images(vectors = generated_samples)
        # Desnormalizando
        desnorm_func = Utils.inverse_normalize()
        desnorm_imgs = desnorm_func(generated_imgs)
        # Alterando device para cpu
        desnorm_imgs = desnorm_imgs.cpu().detach()

        # Salvando imagens
        for i in range(latent_space_samples.size(0)):
            # Ajuste a escala para 0-255 e converta para tipo uint8
            data = (desnorm_imgs[i].permute(1, 2, 0)).numpy()
            # data = data.reshape((data.shape[0], data.shape[1]))
            data = (data * 255).astype(np.uint8)
            # Crie um objeto de imagem usando a matriz de dados
            image = Image.fromarray(data, mode='RGB')
            # Especifique o caminho do arquivo onde deseja salvar a imagem
            filename = f'{imgpath}/img_{i}.jpg' 
            # Salve a imagem
            image.save(filename)