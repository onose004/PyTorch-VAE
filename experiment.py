import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import numpy as np
import os

import matplotlib.pyplot as plt

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict
                 ) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        track_steps = False

        if track_steps:
            self.logger.experiment.add_histogram(f"training_epoch_{self.current_epoch:02}/0_input", real_img, batch_idx)
            encoding = self.model.encode(real_img)[0]
            self.logger.experiment.add_histogram(f"training_epoch_{self.current_epoch:02}/1_encoding", encoding, batch_idx)
            quantized_inputs, vq_loss = self.model.vq_layer(encoding)
            self.logger.experiment.add_histogram(f"training_epoch_{self.current_epoch:02}/2_quantized_inputs", quantized_inputs, batch_idx)


        results = self.forward(real_img, labels = labels)

        if track_steps:
            self.logger.experiment.add_histogram(f"training_epoch_{self.current_epoch:02}/3_output", results[0], batch_idx)

        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({f"Loss/{key}": val.item() for key, val in train_loss.items()})

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    @staticmethod
    def matplotlib_imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    @staticmethod
    def data_grid(data, max_show_batch=4):
        if data.ndim == 4:
            nb, nd, nw, nh = data.shape
        if data.ndim == 5:
            nb, _, nd, nw, nh = data.shape
        else:
            raise ValueError
            
        data = np.array(data[0:min(nb, max_show_batch)])
        
        if nd > 1:
            ncols = nd
            nrows = max_show_batch
            grid = data.reshape(nrows, ncols, nh, nw).swapaxes(1,2).reshape(1, nh*nrows, nw*ncols)
            
        else:
            ncols = 1
            nrows = max_show_batch 
            grid = data.reshape(nrows, ncols, nh, nw).swapaxes(1,2).reshape(1, nh*nrows, nw*ncols)
            
        grid = grid / 2 + 0.5     # unnormalize
        return torch.from_numpy(grid)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)

        grid_real = self.data_grid(test_input.data.cpu())
        self.logger.experiment.add_image("Image/Real", grid_real)

        grid_recons = self.data_grid(recons.data.cpu())
        self.logger.experiment.add_image("Image/Reconstruction", grid_recons)

        self.logger.experiment.add_histogram("model/0_input", test_input, self.current_epoch)
        self.logger.experiment.add_graph(self.model.encoder, test_input)
        encoding = self.model.encode(test_input)[0]
        self.logger.experiment.add_histogram("model/1_encoding", encoding, self.current_epoch)
        quantized_inputs, vq_loss = self.model.vq_layer(encoding)
        self.logger.experiment.add_histogram("model/2_quantized_inputs", quantized_inputs, self.current_epoch)
        self.logger.experiment.add_graph(self.model.decoder, quantized_inputs)
        self.logger.experiment.add_histogram("model/3_output", recons, self.current_epoch)


        if False:
            torch.save(test_label, f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                      f"{self.logger.name}_{self.current_epoch}_label.pt")            
            torch.save(recons.data, f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                    f"recons_{self.logger.name}_{self.current_epoch}_data.pt")      
            torch.save(                                                         
                    test_input.data, f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                    f"real_img_{self.logger.name}_{self.current_epoch}_data.pt")
        # vutils.save_image(recons.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"recons_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        # try:
        #     samples = self.model.sample(144,
        #                                 self.curr_device,
        #                                 labels = test_label)
        #     vutils.save_image(samples.cpu().data,
        #                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                       f"{self.logger.name}_{self.current_epoch}.png",
        #                       normalize=True,
        #                       nrow=12)
        # except:
        #     pass


        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        return optimizer
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            dataset = CelebA(root = self.params['data_path'],
                             split = "train",
                             transform=transform,
                             download=True)
        elif self.params['dataset'] == 'CIFAR10':
            dataset = CIFAR10(root = self.params['data_path'],
                             train=True,
                             transform=transform,
                             download=False)
        elif self.params['dataset'] == 'MNIST':
            dataset = MNIST(root = self.params['data_path'],
                             train=True,
                             transform=transform,
                             download=False)
            # idx = (dataset.targets == 3) | (dataset.targets == 5) | (dataset.targets == 8)
            # dataset.targets = dataset.targets[idx]
            # dataset.data = dataset.data[idx]
        elif self.params['dataset'] == 'VOL':
            vol_np = np.load(self.params['data_path'])
            data = torch.from_numpy(vol_np).float()
            targets = torch.ones(len(data))
            dataset = torch.utils.data.TensorDataset(data, targets)
        elif self.params['dataset'] == 'MNISTB':
            vol_np = np.load(os.path.join(self.params['data_path']))
            data = torch.from_numpy(vol_np).float()
            targets = torch.ones(len(data))
            dataset = torch.utils.data.TensorDataset(data, targets)
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        dl = DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          shuffle = True,
                          drop_last=True)
        return dl

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'CIFAR10':
            self.sample_dataloader =  DataLoader(CIFAR10(root = self.params['data_path'],
                                                        train=False,
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'MNIST':
            dataset = MNIST(root = self.params['data_path'],
                    train=False,
                    transform=transform,
                    download=False)
            # idx = (dataset.targets == 3) | (dataset.targets == 5 ) | (dataset.targets == 8) | (dataset.targets == 1)
            # dataset.targets = dataset.targets[idx]
            # dataset.data = dataset.data[idx]
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'VOL':
            vol_np = np.load(self.params['data_path'])
            data = torch.from_numpy(vol_np).float()
            targets = torch.ones(len(data))
            dataset = torch.utils.data.TensorDataset(data, targets)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'MNISTB':
            vol_np = np.load(os.path.join(self.params['data_path']))
            data = torch.from_numpy(vol_np).float()
            targets = torch.ones(len(data))
            dataset = torch.utils.data.TensorDataset(data, targets)
            self.sample_dataloader =  DataLoader(dataset,
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'celeba':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'CIFAR10':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'MNIST':
            transform = transforms.Compose([transforms.Resize(self.params['img_size']),
                                            transforms.ToTensor(),
                                            SetRange])
        elif self.params['dataset'] == 'VOL':
            transform = transforms.Compose([SetRange])
        elif self.params['dataset'] == 'MNISTB':
            transform = transforms.Compose([SetRange])
        else:
            raise ValueError('Undefined dataset type')
        return transform

