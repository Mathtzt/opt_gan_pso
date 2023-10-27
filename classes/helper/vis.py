import matplotlib.pyplot as plt

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