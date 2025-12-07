import argparse
import os
import logging
import random
import sys

from matplotlib import pyplot as plt
import numpy as np
import skimage.io as skio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms as T
from torchvision.utils import make_grid

img_size = 224
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# hyper-parameters, tune them for better results
num_epochs = 2000
sigma = 1.0
langevin_num_steps = 10
langevin_step_size = 1.0
# learning rates for (weight, bias) of up to 3 conv layers
lrs = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]


# hydra-style logging
def get_logger(exp_dir):
    logger = logging.getLogger(__name__)
    logger.handlers = []
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(exp_dir, 'output.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    return logger


def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_cudnn():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def load_img_tensor(img_path, transform, device):
    img = Image.open(img_path)
    img = transform(img)
    return img.to(device)


def visualize_img_tensor(img_tensor, img_path, mu=0, show=None):
    if img_tensor.ndim == 4:
        img_tensor = img_tensor[0]
    img_array = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    if isinstance(mu, torch.Tensor):
        mu = mu.detach().cpu().numpy()
    img_array += mu
    if img_array.mean() > 1:
        img_array /= 255
    img_array = img_array.clip(0, 1)

    if show:
        plt.figure()
        plt.imshow(img_array)
        plt.title(show)
        plt.axis('off')

    img_array = (img_array * 255).astype(np.uint8)
    skio.imsave(fname=img_path, arr=img_array)


class Descriptor(nn.Module):
    """
    Hierarchical descriptor used by the FRAME model.
    num_layers = 1, 2, or 3 controls how many conv layers are stacked.
    """
    def __init__(self, num_layers):
        super(Descriptor, self).__init__()

        # You can change the channel numbers / kernel sizes if you want
        # to design your own 1-layer and 2-layer structures.
        self.num_layers = num_layers

        # First conv layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=15,
            stride=1,
            padding=7,
            bias=True
        )

        # Second conv layer (used when num_layers > 1)
        if num_layers > 1:
            self.conv2 = nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True
            )

        # Third conv layer (used when num_layers > 2)
        if num_layers > 2:
            self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )

        self.get_feat_sizes()

    def get_feat_sizes(self):
        """
        Compute feature-map spatial sizes for each layer, only used to
        normalize gradients of weights and biases (heuristic).
        """
        device = next(self.parameters()).device
        x = torch.zeros(1, 3, img_size, img_size).to(device)
        self.feat_size_list = []
        for c in self.children():
            x = c(x)
            feat_size = x.shape[2] * x.shape[3]
            # one entry for weight, one for bias of this conv layer
            self.feat_size_list.extend([feat_size, feat_size])

    def forward(self, x):
        # Layer 1
        x = F.relu(self.conv1(x))

        # Layer 2
        if hasattr(self, 'conv2'):
            x = F.relu(self.conv2(x))

        # Layer 3
        if hasattr(self, 'conv3'):
            x = F.relu(self.conv3(x))

        # The FRAME scoring function f(I; w) will later be taken as
        # the sum over this final feature map.
        return x


def langevin(descriptor, x, num_steps, eps, sigma):
    """
    Langevin dynamics for sampling from p(I; w).

    Update rule (discrete-time):
        I_{t+1} = I_t + eps^2/2 * ( d/dI f(I_t; w) - I_t / sigma^2 ) + eps * N(0, 1)

    Here f(I; w) is implemented as the sum of the top-layer feature map
    of the descriptor network.
    """
    for _ in range(num_steps):
        # Re-wrap x so that autograd tracks it
        x = Variable(x.data, requires_grad=True)
        
        # TODO non-scalar tensor backward (Use Equation 5 from proj4.pdf)

        # Forward pass through descriptor
        descriptor.zero_grad()
        feat = descriptor(x)

        # f(I; w) = sum_k,x [F_k^{(L)} * I](x)
        f = feat.sum()

        # Compute gradient w.r.t. image x: d f(I; w) / dI
        f.backward()
        grad_x = x.grad

        # Gaussian noise
        noise = torch.randn_like(x).to(device)

        # Langevin update: gradient term of log p(I; w) plus noise
        x.data += (eps * eps / 2.0) * (grad_x - x / (sigma * sigma)) + eps * noise

    # The caller usually detaches again, but returning a detached tensor
    # is safe and avoids accidental graph retention.
    return x.detach()


def run(exp_dir: str, num_layers: int, img_path: str, logger: logging.Logger):
    transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    img_tgt = load_img_tensor(img_path=img_path, transform=transform, device=device)
    visualize_img_tensor(img_tensor=img_tgt, img_path=os.path.join(exp_dir, 'target.png'))
    # img_tgt: [3, 224, 224]

    # empirically, scale at [0, 255] is more tractable than scale at [0, 1]
    img_tgt *= 255
    mu_tgt = img_tgt.mean((1, 2))
    img_tgt = img_tgt - mu_tgt.reshape(3, 1, 1)
    img_tgt = img_tgt[None, ...]        # [1, 3, H, W]
    img_syn = torch.zeros_like(img_tgt, device=device)

    model = Descriptor(num_layers=num_layers)
    model.to(device)

    log_interval = max(1, num_epochs // 100)
    vis_interval = max(1, num_epochs // 10)

    for epoch in range(num_epochs):
        # --------------------------------------------------------------
        # 1. Update synthesized image via Langevin dynamics
        # --------------------------------------------------------------
        img_syn = langevin(
            descriptor=model,
            x=img_syn,
            num_steps=langevin_num_steps,
            eps=langevin_step_size,
            sigma=sigma
        )

        # --------------------------------------------------------------
        # 2. Compute energies and objective f(Itgt) - f(Isyn)
        #    f(I; w) is the sum of descriptor top-layer features.
        # --------------------------------------------------------------
        # target image energy
        feat_tgt = model(img_tgt)
        f_tgt = feat_tgt.sum()

        # synthesized image energy (img_syn is treated as constant)
        feat_syn = model(img_syn.detach())
        f_syn = feat_syn.sum()

        # objective: maximize f(Itgt) - f(Isyn)
        f_diff = f_tgt - f_syn

        # --------------------------------------------------------------
        # 3. Backprop and update descriptor parameters (gradient ascent)
        # --------------------------------------------------------------
        model.zero_grad()
        f_diff.backward()

        # manual gradient-ascent update with per-layer normalization
        for p, feat_size, lr in zip(model.parameters(), model.feat_size_list, lrs):
            if p.grad is None:
                continue
            grad = p.grad.data / float(feat_size)
            p.data += lr * grad

        # --------------------------------------------------------------
        # 4. Logging & visualization
        # --------------------------------------------------------------
        if (epoch + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch + 1:<4d}: "
                f"f_diff = {f_diff.item():<10.2f}, "
                f"f_tgt = {f_tgt.item():<10.2f}, "
                f"f_syn = {f_syn.item():<10.2f}"
            )

        if (epoch + 1) % vis_interval == 0:
            visualize_img_tensor(
                img_tensor=img_syn,
                img_path=os.path.join(exp_dir, f'{epoch + 1}.png'),
                mu=mu_tgt
            )

    # --------------------------------------------------------------
    # 5. Visualize conv1 filters after training
    # --------------------------------------------------------------
    weights = list(model.parameters())[0].data  # conv1 weights
    grid = make_grid(weights, normalize=True)
    visualize_img_tensor(
        img_tensor=grid,
        img_path=os.path.join(exp_dir, 'conv1.png')
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, help="Number of layers (1, 2, or 3)")
    parser.add_argument('--tag', type=str, help="Image tag, must match file name in images/ (without extension)")
    args = parser.parse_args()

    img_files = os.listdir('images')

    for img_file in img_files:
        if args.tag == os.path.splitext(img_file)[0]:
            set_seed(1)
            set_cudnn()
            exp_dir = f'{args.tag}_{args.layer}layer'
            os.makedirs(exp_dir, exist_ok=True)
            logger = get_logger(exp_dir=exp_dir)
            run(
                exp_dir=exp_dir,
                num_layers=args.layer,
                img_path=os.path.join('images', img_file),
                logger=logger
            )
            return

    raise ValueError(f"The specified image tag should be included in {img_files}")


if __name__ == '__main__':
    main()
