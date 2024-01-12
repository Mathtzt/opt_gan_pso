import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.metrics import FID, InceptionScore, RunningAverage, SSIM
from ignite.contrib.handlers import ProgressBar 

from .generator import Generator
from .discriminator import Discriminator
from optgan.classes.base.namespaces import ExperimentDict
from optgan.classes.base.enums import OptimizerNames
from optgan.classes.helper.vis import Vis

ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)

from pytorch_fid.inception import InceptionV3

from cleanfid import fid

### Reproductibility and logging details ###  
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# ignite.utils.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

class DCGanIgnite():
    def __init__(self,
                 train_loader,
                 test_loader,
                 exp_dict: ExperimentDict,
                 paths_dict: dict,
                 g_optimizer_type: OptimizerNames,
                 d_optimizer_type: OptimizerNames,
                 lr: float,
                 g_n_conv_blocks: int,
                 d_n_conv_blocks: int,
                 batch_size: int = None,
                 latent_size: int = 100):

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.g_optimizer_type = g_optimizer_type
        self.d_optimizer_type = d_optimizer_type
        self.lr = lr
        self.g_n_conv_blocks = g_n_conv_blocks
        self.d_n_conv_blocks = d_n_conv_blocks
        
        self.data_type = exp_dict.dataset.type
        self.specialist_class = exp_dict.dataset.class_name

        self.paths_dict = paths_dict

        self.num_epochs = exp_dict.synthesizer.num_epochs
        self.batch_size = batch_size if batch_size else exp_dict.synthesizer.batch_size
  
        self.nlatent_space = latent_size
        self.nchannel = next(iter(train_loader))[0].size(1)
        self.d_train_steps = 2
        self.netG = None
        self.netD = None
        self.trainer = None
        self.evaluator = None

        self.G_losses = []
        self.D_losses = []
        self.img_list = []
        self.fid_values = []
        self.kid_values = []
        self.is_values = []

    def setup_network(self):
        # Create the Gerador
        self.netG = idist.auto_model(Generator(n_conv_blocks = self.g_n_conv_blocks, nc = self.nchannel, nz = self.nlatent_space))
        self.netD = idist.auto_model(Discriminator(n_conv_blocks = self.d_n_conv_blocks, nc = self.nchannel))

    def setup_optimizer(self):
        # Initialize the loss function
        # Setup Adam optimizers for both G and D
        self.criterion = nn.BCELoss()

        if self.g_optimizer_type == OptimizerNames.ADAM:
            self.g_optimizer = idist.auto_optim(optim.Adam(self.netG.parameters(), lr = self.lr, betas=(0.5, 0.999)))
        if self.g_optimizer_type == OptimizerNames.SGD:
            self.g_optimizer = idist.auto_optim(optim.SGD(self.netG.parameters(), lr = self.lr, momentum = 0.9))
        if self.g_optimizer_type == OptimizerNames.NADAM:
            self.g_optimizer = idist.auto_optim(optim.NAdam(self.netG.parameters(), lr = self.lr))

        if self.d_optimizer_type == OptimizerNames.ADAM:
            self.d_optimizer = idist.auto_optim(optim.Adam(self.netD.parameters(), lr = self.lr, betas=(0.5, 0.999)))
        if self.d_optimizer_type == OptimizerNames.SGD:
            self.d_optimizer = idist.auto_optim(optim.SGD(self.netD.parameters(), lr = self.lr, momentum = 0.9))
        if self.d_optimizer_type == OptimizerNames.NADAM:
            self.d_optimizer = idist.auto_optim(optim.NAdam(self.netD.parameters(), lr = self.lr))

    def initialize_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def get_metrics(self):
        # pytorch_fid model
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to('cuda')

        # wrapper model to pytorch_fid model
        wrapper_model = WrapperInceptionV3(model)
        wrapper_model.eval();

        fid_metric = FID(device=idist.device(), num_features=dims, feature_extractor=wrapper_model)
        # ssim = SSIM(data_range=1.0, device=idist.device())
        is_metric = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])

        return fid_metric, is_metric

    # @trainer.on(Events.STARTED)
    def init_weights(self):
        self.netD.apply(self.initialize_fn)
        self.netG.apply(self.initialize_fn)

    # @trainer.on(Events.ITERATION_COMPLETED)
    def store_losses(self, engine):
        o = engine.state.output
        self.G_losses.append(o["Loss_G"])
        self.D_losses.append(o["Loss_D"])

    # @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(self, engine):        
        print(f"Epoch [{engine.state.epoch}/{self.num_epochs}]")

    def log_similarity_result(self, engine):
        self.evaluator.run(self.test_loader, max_epochs=1)
        metrics = self.evaluator.state.metrics
        fid_score = metrics['fid']
        # ssim_score = metrics['ssim']
        is_score = metrics['is']
        self.fid_values.append(fid_score)
        self.is_values.append(is_score)
        
        print("\nMetric Scores:")
        print(f"*    FID : {fid_score:4f}")
        print(f"*    IS : {is_score:4f}")

    def generate_synthetic_imgs(self, engine):
        Vis.synthesize(generator = self.netG,
                       nsamples = 1000,
                       nlatent_space = self.nlatent_space,
                       img_path = self.paths_dict["fake_imgs"],
                       device = idist.device())
    
    def compute_kid(self, engine):
        test_dir = f'cifar10_test/{self.specialist_class}'
        test_imgs_path = os.path.join(self.paths_dict["data"], test_dir)

        fexp_dir = self.paths_dict["fake_imgs"]
        kid_score = fid.compute_kid(test_imgs_path, fexp_dir)

        print("\nMetric Scores:")
        print(f"*    KID : {kid_score:4f}")

        self.kid_values.append(kid_score)

    # @trainer.on(Events.ITERATION_COMPLETED(every=50))
    def store_images(self, engine):
        fixed_noise = torch.randn(128, self.nlatent_space, 1, 1, device=idist.device())

        with torch.no_grad():
            fake = self.netG(fixed_noise).cpu()
        self.img_list.append(fake)

    def training_step(self, engine, data):
        real_label_value = 0.9
        fake_label_value = 0.1
        # Set the models for training
        self.netG.train()
        self.netD.train()

        real = data[0].to(idist.device())
        b_size = real.size(0)

        true_label = torch.full((b_size,), real_label_value, dtype=torch.float, device=idist.device())
        fake_label = torch.full((b_size,), fake_label_value, dtype=torch.float, device=idist.device())

        
        ############################
        # (1) Update G network: maximize log(D(G(z)))
        ###########################
        noise = torch.randn(b_size, self.nlatent_space, 1, 1, device=idist.device())

        self.g_optimizer.zero_grad()
        fake_img = self.netG(noise)
        output3 = self.netD(fake_img).view(-1)
        # Calculate G's loss based on this output
        errG = -torch.mean(torch.log(output3 + 1e-8))#self.criterion(output3, true_label)
        # Calculate gradients for G
        errG.backward()
        # Update G
        self.g_optimizer.step()

        # for _ in range(self.d_train_steps):
        ############################
        # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        self.d_optimizer.zero_grad()
        # Forward pass real batch through D
        output1 = self.netD(real).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output1, true_label)
        # Classify all fake batch with D
        output2 = self.netD(fake_img.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = torch.sum(-torch.mean(torch.log(output1 + 1e-8) + torch.log(1 - output2 + 1e-8))) #self.criterion(output2, fake_label)
        # Compute error of D as sum over the fake and the real batches
        errD = (errD_real + errD_fake) * 0.5
        # Calculate gradients for D
        errD.backward()
        # Update D
        self.d_optimizer.step()

        return {
            "Loss_G" : errG.item(),
            "Loss_D" : errD.item(),
            "D_x": output1.mean().item(),
            "D_G_z1": output2.mean().item(),
            "D_G_z2": output3.mean().item(),
        }
    
    def evaluation_step(self, engine, batch):
        with torch.no_grad():
            noise = torch.randn(self.batch_size, self.nlatent_space, 1, 1, device=idist.device())
            
            self.netG.eval()
            fake_batch = self.netG(noise)
            fake = Vis.interpolate(fake_batch)
            real = Vis.interpolate(batch[0])
            
            return fake, real
        
    def training(self, *args):
        self.trainer.run(self.train_loader, max_epochs = self.num_epochs)
        
    def fit(self, save_files: bool = True):
        self.setup_network()
        self.setup_optimizer()

        self.trainer = Engine(self.training_step)
        # self.evaluator = Engine(self.evaluation_step)

        self.trainer.add_event_handler(Events.STARTED, self.init_weights)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.store_losses)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every = 50), self.log_training_results)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), self.store_images)
        # self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every = self.num_epochs), self.log_similarity_result)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every = self.num_epochs), self.generate_synthetic_imgs)
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every = self.num_epochs), self.compute_kid)
        
        # fid_metric, is_metric = self.get_metrics()
        # fid_metric.attach(self.evaluator, "fid")
        # is_metric.attach(self.evaluator, "is")

        RunningAverage(output_transform=lambda x: x["Loss_G"]).attach(self.trainer, 'Loss_G')
        RunningAverage(output_transform=lambda x: x["Loss_D"]).attach(self.trainer, 'Loss_D')

        # ProgressBar().attach(self.trainer, metric_names=['Loss_G','Loss_D'])
        # ProgressBar().attach(self.evaluator)

        with idist.Parallel(backend='xla-tpu', nproc_per_node = 4, start_method = 'fork') as parallel: #'nccl'
            parallel.run(self.training)

        if save_files:
            Vis.plot_gan_train_evolution(self.G_losses, self.D_losses, self.paths_dict["eval_imgs"], "gan_loss_evolution")
            # Vis.plot_fid_is_evolution(self.is_values, self.fid_values, self.paths_dict["eval_imgs"], "measures_evolution")
            Vis.plot_real_fake_images(self.train_loader, self.img_list, self.paths_dict["eval_imgs"], "real_fake_samples")

        return self.kid_values[-1]
    
# wrapper class as feature_extractor
class WrapperInceptionV3(nn.Module):

    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]

        return y

            




