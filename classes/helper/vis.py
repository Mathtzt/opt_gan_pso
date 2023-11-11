import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

from tqdm import tqdm
from scipy.special import kl_div
from classes.helper.utils import Utils

class Vis:

    @staticmethod
    def plot_gan_train_evolution(g_losses, d_losses, path, figname):
        plt.figure(figsize=(5, 3))
        plt.plot(g_losses, color = "c", label = "generator")
        plt.plot(d_losses, color = "k", label = "discriminator")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend(fontsize = 10)
        plt.tight_layout()

        plt.savefig(f"{path}/{figname}.png")
        plt.close()

    @staticmethod
    def plot_fid_is_evolution(is_values, fid_values, path, figname):
        fig, ax1 = plt.subplots()

        plt.title("Evaluation Metric During Training")

        color = 'tab:red'
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('IS', color=color)
        ax1.plot(is_values, color=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('FID', color=color)
        ax2.plot(fid_values, color=color)

        fig.tight_layout()
        
        plt.savefig(f"{path}/{figname}.png")
        plt.close()

    @staticmethod
    def plot_real_fake_images(train_loader, img_list, path, figname):
        # Grab a batch of real images from the dataloader
        real_batch = next(iter(train_loader))

        # Plot the real images
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:32], padding=2, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(vutils.make_grid(img_list[-1][:32], padding=2, normalize=True).cpu(),(1,2,0)))

        plt.tight_layout()
        plt.savefig(f"{path}/{figname}.png")
        plt.close()

    @staticmethod
    def plot_pdf(pdf, bins, class_label, ax = None, show_xlabel = False, show_ylabel = False, kl_value = None):
        if ax is None:
            _, ax = plt.subplots()

        # Plote a PDF
        ax.bar(bins[:-1], pdf, width=1.0, color='gray')
        title = f"{class_label} - KL: {kl_value}" if kl_value else f"{class_label}"
        ax.set_title(title)

        if show_xlabel: ax.set_xlabel("Intensidade de Pixel")
        if show_ylabel: ax.set_ylabel("Probabilidade")

        ax.set_ylim(0, 0.017)

        return ax
    
    @staticmethod
    def plot_grid_pdfs(dataset, classes_list, original_pdfs_for_kl = None):
        # Crie uma figura com subplots 2x5
        rows = 2
        columns = 5

        _, axs = plt.subplots(rows, columns, figsize=(18, 5))

        # Flatten os eixos para que possamos iterar sobre eles facilmente
        axs = axs.flatten()

        # Itere sobre as imagens e os eixos, e plote as imagens
        for i, ax in tqdm(enumerate(axs.flatten())):
            sxlb = True if i >= columns else False
            sylb = True if i in [0, columns] else False

            pdf, bins, class_label = Utils.calculate_pdf(dataset, i)  # Certifique-se de ter a função calculate_pdf definida

            if original_pdfs_for_kl:
                o_pdf = original_pdfs_for_kl[i]
                kl_value = np.round(np.sum(kl_div(o_pdf, pdf)), 5)

            else: kl_value = None

            Vis.plot_pdf(pdf, bins, classes_list[class_label], ax, show_xlabel = sxlb, show_ylabel = sylb, kl_value = kl_value)

        # Ajuste o layout para evitar sobreposições
        plt.tight_layout()
        plt.suptitle("Funções de densidade de probabilidade das classes", y = 1.05)
        # Mostre a figura
        plt.show()

        

    @staticmethod
    def interpolate(batch):
        arr = []
        for img in batch:
            pil_img = transforms.ToPILImage()(img)
            resized_img = pil_img.resize((299,299), Image.BILINEAR)
            arr.append(transforms.ToTensor()(resized_img))
        
        return torch.stack(arr)
