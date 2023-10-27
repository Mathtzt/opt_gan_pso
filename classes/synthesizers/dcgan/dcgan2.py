import numpy as np

import torch
torch.manual_seed(42)
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from classes.base.namespaces import ExperimentDict

from classes.helper.utils import Utils
from classes.helper.datasets import Datasets
from classes.helper.vis import Vis
from classes.base.enums import DatasetNames, OptimizerNames

from .generator import Generator
from .discriminator import Discriminator

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
        self.specialist_class = exp_dict.dataset.class_name

        self.paths_dict = paths_dict

        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.loss_func = None

        self.num_epochs = exp_dict.synthesizer.num_epochs
        self.batch_size = exp_dict.synthesizer.batch_size
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

    def setup_optimizer(self):
        # Initialize the loss function
        # Setup Adam optimizers for both G and D
        if self.g_optimizer_type == OptimizerNames.ADAM:
            self.g_optimizer = optim.Adam(self.generator.parameters(), lr = self.lr, betas=(0.5, 0.999))
        if self.g_optimizer_type == OptimizerNames.SGD:
            self.g_optimizer = optim.SGD(self.generator.parameters(), lr = self.lr, momentum = 0.9)

        if self.d_optimizer_type == OptimizerNames.ADAM:
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr = self.lr, betas=(0.5, 0.999))
        if self.d_optimizer_type == OptimizerNames.SGD:
            self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr = self.lr, momentum = 0.9)

    def vector_to_images(self, vectors, dsize: tuple = (32, 32)):
        return vectors.view(vectors.size(0), self.nchannel, dsize[0], dsize[1])
    
    def save(self, model, filename: str = 'model_gan.pt'):
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(f'{self.paths_dict["model"]}/{filename}')

    def load(self, model_path: str):
        self.generator = torch.jit.load(model_path)

    def fit(self):
        if self.data_type == DatasetNames.CIFAR10:
            dataloader = Datasets.get_cifar_as_dataloaders(datapath = self.paths_dict["data"], 
                                                           specialist_class = self.specialist_class)
        else:
            raise Exception("data_type não implementado ainda.")

        self.nchannel = next(iter(dataloader))[0].size(1)
        self.setup_network()
        self.setup_optimizer()

        epoch_g_loss, epoch_d_loss = [], []
        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.
        d_train_steps = 1
        print("Starting Training Loop...")
        # For each epoch
        for epoch in tqdm(range(self.num_epochs)):
            # For each batch in the dataloader
            G_losses, D_losses = [], []
            
            for i, data in enumerate(dataloader, 0):
                
                D_step_loss = []
                for _ in range(d_train_steps):

                    real_img = data[0].to(self.device)
                    b_size = real_img.size(0)
                    label = torch.full((b_size,), real_label, dtype = torch.float, device = self.device)

                    self.d_optimizer.zero_grad()

                    d_loss = self.train_discriminator(real_img)

                    d_loss.backward()
                    self.d_optimizer.step()

                    D_step_loss.append(d_loss.item())

                D_losses.append(np.mean(D_step_loss))

                self.g_optimizer.zero_grad()

                g_loss = self.train_generator(real_img)
                
                G_losses.append(g_loss.item())

                g_loss.backward()
                self.g_optimizer.step()

            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                   %(epoch, self.num_epochs, np.mean(G_losses), np.mean(D_losses)))

            epoch_g_loss.append(np.mean(G_losses))
            epoch_d_loss.append(np.mean(D_losses))
        
        self.save(model = self.generator, filename = f'model_{self.specialist_class}.pt')

        Utils.save_list_as_txt(epoch_g_loss, self.paths_dict["eval_metrics"], name = "generator_losses")
        Utils.save_list_as_txt(epoch_d_loss, self.paths_dict["eval_metrics"], name = "discriminator_losses")

        Vis.plot_gan_train_evolution(epoch_g_loss, epoch_d_loss, self.paths_dict["eval_imgs"], "gan_loss_evolution")

        return G_losses[-1]

    def train_discriminator(self, real_cpu):
        noise = torch.randn(real_cpu.size(0), self.nlatent_space, 1, 1, device = self.device)

        g_out = self.generator(noise)

        d_fake = self.discriminator(g_out.detach()).view(-1)
        d_real = self.discriminator(real_cpu)#.view(-1)

        d_loss = torch.sum(-torch.mean(torch.log(d_real + 1e-8)
                                       + torch.log(1 - d_fake + 1e-8)))

        return d_loss
    
    def train_generator(self, real_cpu):
        noise = torch.randn(real_cpu.size(0), self.nlatent_space, 1, 1, device = self.device)

        g_out = self.generator(noise)
        d_out = self.discriminator(g_out)#.view(-1)

        g_loss = -torch.mean(torch.log(d_out + 1e-8))

        return g_loss
    
    def generate_synthetic_images(self, exp_dirname: str):
        ## Verificando se há um modelo instanciado ##
        if self.generator is None:
            return "Necessário carregar um modelo previamente. Utilize a função load()."
        
        base_dir = './'
        dirpath = Utils.criar_pasta(path = base_dir, name = "gen_imgs")
        exp_path = Utils.criar_pasta(path = dirpath, name = exp_dirname)
        img_path = Utils.criar_pasta(path = exp_path, name = self.specialist_class)

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
            filename = f'{img_path}/img_{i}.jpg' 
            # Salve a imagem
            image.save(filename)