import torchvision.utils as vutils
import matplotlib.pyplot as plt

def show_imgs(x, new_fig=True):
    grid = vutils.make_grid(x.detach().cpu(), nrow=8, normalize=True, pad_value=0.3)
    grid = grid.transpose(0,2).transpose(0,1)  # channels last
    if new_fig:
        plt.figure()
    plt.imshow(grid.numpy())
    plt.axis('off')
    plt.show()