import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
import torchvision.models as pretrained
import torchvision.transforms as transforms
from utils import datautils
import models
from utils import utils
import numpy as np
import PIL
from tqdm import tqdm
import sklearn
import scipy
import higher
import copy


class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super().__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        sigmoid = nn.Sigmoid()
        return sigmoid(out)


class BaseModel(nn.Module):
    """
    Inspired by the PYTORCH LIGHTNING https://pytorch-lightning.readthedocs.io/en/latest/
    Similar but lighter and customized version.
    """
    DATA_ROOT = os.environ.get('DATA_ROOT', '../data/')
    CLOTHING_PATH = os.environ.get(
        'CLOTHING_PATH', '../data/clothing1m/')

    def __init__(self, params, device):
        super().__init__()
        self.hparams = copy.deepcopy(params)
        self.device = device
        if params.data == 'cifar':
            n_classes = 10
        elif params.data == 'cifar100':
            n_classes = 100
        else:
            n_classes = 14
        if self.hparams.encoder_type == 'simclr':
            encoder = models.encoder.EncodeProject(params)
            encoder = encoder.to(device)
            if params.encoder_ckpt:  # load encoder
                ckpt = torch.load(params.encoder_ckpt, map_location=device)
                missing = encoder.load_state_dict(
                    ckpt['state_dict'], strict=False)
                #print(missing)
            if params.data == 'cifar':
                hdim = encoder(torch.ones(10, 3, 32, 32).to(
                    device)).shape[1]
            elif params.data == 'cifar100':
                hdim = encoder(torch.ones(10, 3, 32, 32).to(
                    device)).shape[1]
            elif params.data == 'clothing':
                hdim = encoder(torch.ones(10, 3, 224, 224).to(
                    device)).shape[1]
            self.n_classes = n_classes
            print('projection dimension: ', hdim)
            linear_model = nn.Linear(hdim, n_classes).to(device)
            linear_model.weight.data.zero_()
            linear_model.bias.data.zero_()
            model = models.encoder.Model(encoder, linear_model)
            self.model = model

        elif self.hparams.encoder_type == 'imagenet' and self.hparams.data == 'clothing':
            model = pretrained.resnet50(pretrained=True)
            model = model.to(device)
            model.fc = nn.Linear(2048, n_classes).to(device)
            self.model = model

        elif self.hparams.encoder_type == 'imagenet' and 'cifar' in self.hparams.data:
            encoder = models.encoder.EncodeProject(params)
            encoder = encoder.to(device)
            linear_model = nn.Linear(2048, 1000).to(device)
            model = models.encoder.Model(encoder, linear_model)
            model = model.to(device)
            ckpt = torch.load(params.imagenet_ckpt, map_location=device)
            missing = model.load_state_dict(ckpt['state_dict'], strict=False)
            #print(missing)
            model.linear_layer = nn.Linear(2048, n_classes).to(device)
            model.linear_layer.weight.data.zero_()
            model.linear_layer.bias.data.zero_()
            self.model = model

    def prepare_data(self):
        train_transform, test_transform = self.transforms()
        if self.hparams.data == 'cifar':
            self.trainset = datasets.CIFAR10(
                root=self.DATA_ROOT, train=True, download=True, transform=train_transform)
            self.testset = datasets.CIFAR10(
                root=self.DATA_ROOT, train=False, download=True, transform=test_transform)

            self.trainset, self.valset = torch.utils.data.random_split(self.trainset, [len(
                self.trainset)-1000, 1000], generator=torch.Generator().manual_seed(self.hparams.seed))
        elif self.hparams.data == 'cifar100':
            self.trainset = datasets.CIFAR100(
                root=self.DATA_ROOT, train=True, download=True, transform=train_transform)
            self.testset = datasets.CIFAR100(
                root=self.DATA_ROOT, train=False, download=True, transform=test_transform)
            self.trainset, self.valset = torch.utils.data.random_split(self.trainset, [len(
                self.trainset)-1000, 1000], generator=torch.Generator().manual_seed(self.hparams.seed))
        elif self.hparams.data == 'clothing':
            self.trainset = utils.ClothingDataset(
                transform=train_transform, mode='train', path=self.CLOTHING_PATH)
            self.testset = utils.ClothingDataset(
                transform=test_transform, mode='test', path=self.CLOTHING_PATH)
            self.valset = utils.ClothingDataset(
                transform=train_transform, mode='val', path=self.CLOTHING_PATH)
        else:
            raise NotImplementedError

        if self.hparams.data != 'clothing':  # add artificial noise
            np.random.seed(self.hparams.seed)
            num_classes = len(self.trainset.dataset.classes)
            class_to_idx = self.trainset.dataset.class_to_idx
            C = utils.noise_matrix(
                self.hparams.corruption_prob, num_classes, self.hparams.corruption_type, class_to_idx)
            for i in self.trainset.indices:
                self.trainset.dataset.targets[i] = np.random.choice(
                    num_classes, p=C[self.trainset.dataset.targets[i]])

    def dataloaders(self, iters=None):
        if self.hparams.data == 'clothing':
            self.prepare_data()  # again
            test_loader = torch.utils.data.DataLoader(
                dataset=self.testset,
                batch_size=self.hparams.test_bs,
                shuffle=False,
                num_workers=self.hparams.workers)
            valid_loader = torch.utils.data.DataLoader(
                self.valset,
                num_workers=self.hparams.workers,
                shuffle=False,
                batch_size=self.hparams.batch_size, drop_last=False,
            )
            meta_loader = torch.utils.data.DataLoader(
                self.valset,
                num_workers=self.hparams.workers,
                shuffle=True,
                batch_size=self.hparams.batch_size, drop_last=True,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.workers)
            return train_loader, test_loader, valid_loader, meta_loader

        trainsampler = torch.utils.data.sampler.RandomSampler(
            self.trainset)
        testsampler = torch.utils.data.sampler.RandomSampler(self.testset)
        validsampler = torch.utils.data.sampler.RandomSampler(self.valset)

        self.object_trainsampler = trainsampler
        trainsampler = torch.utils.data.BatchSampler(
            self.object_trainsampler,
            batch_size=self.hparams.batch_size, drop_last=False,
        )

        if iters is not None:
            trainsampler = datautils.ContinousSampler(trainsampler, iters)

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=trainsampler,
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            sampler=testsampler,
            batch_size=self.hparams.test_bs,
        )

        valid_loader = torch.utils.data.DataLoader(
            self.valset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            sampler=validsampler,
            batch_size=self.hparams.test_bs,
        )

        metasampler = torch.utils.data.sampler.RandomSampler(self.valset)
        metasampler = torch.utils.data.BatchSampler(
            metasampler,
            batch_size=self.hparams.batch_size, drop_last=False,
        )
        metasampler = datautils.ContinousSampler(metasampler, iters)
        meta_loader = torch.utils.data.DataLoader(
            self.valset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=metasampler,
        )
        return train_loader, test_loader, valid_loader, meta_loader

    def transforms(self):
        if self.hparams.data == 'cifar' or self.hparams.data == 'cifar100':
            if self.hparams.encoder_type in {'simclr'}:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    datautils.Clip(),
                ])
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            elif self.hparams.encoder_type == 'imagenet':
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        elif self.hparams.data == 'clothing':
            if self.hparams.encoder_type == 'simclr':
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        224,
                        scale=(self.hparams.scale_lower, 1.0),
                        interpolation=PIL.Image.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
                test_transform = transforms.Compose([
                    datautils.CenterCropAndResize(proportion=0.875, size=224),
                    transforms.ToTensor()
                ])
            elif self.hparams.encoder_type == 'imagenet':
                train_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                test_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])

        return train_transform, test_transform


class FineTune(BaseModel):
    def __init__(self, params, device):
        super().__init__(params, device)
        self.optimizer, self.scheduler = models.noisy_models.configure_optimizers(
            self.hparams, self.model, - 1)
        self.hparams.num_val_samples = -1  # default no validation set
        self.hparams.noisy_val_samples = False  # default no noise
        self.hparams.loss = getattr(self.hparams, 'loss', 'ce')
        self.q = getattr(self.hparams, 'q', 0.66)

    def step(self, batch):
        h, y = batch
        p = self.model(h)
        if self.hparams.loss == 'ce':
            loss = F.cross_entropy(p, y)
        elif self.hparams.loss == 'mae':  # mae
            qry_prob = F.softmax(p, dim=-1)
            qry_prob = qry_prob[torch.arange(qry_prob.shape[0]), y]
            loss = 2*torch.mean(1. - qry_prob)
        elif self.hparams.loss == 'qloss':
            qry_prob = F.softmax(p, dim=-1)
            qry_prob = qry_prob[torch.arange(qry_prob.shape[0]), y]
            loss = ((1-(qry_prob**self.q))/self.q)
            loss = torch.mean(loss)
        acc = (p.argmax(1) == y).float()
        return {
            'loss': loss,
            'acc': acc,
        }

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch, it=None):
        logs = self.step(batch)
        if it is not None:
            iters_per_epoch = len(self.trainset) / self.hparams.batch_size
            iters_per_epoch = max(1, int(np.around(iters_per_epoch)))
            logs['epoch'] = (it / iters_per_epoch)
        loss = logs['loss']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return logs

    def test_step(self, batch):
        with torch.no_grad():
            logs = self.step(batch)
        return logs


class MWNetModel(BaseModel):
    def __init__(self, params, device):
        super().__init__(params, device)
        self.hparams.meta_loss = getattr(self.hparams, 'meta_loss', 'ce')
        print(self.hparams)
        self.vnet = VNet(1, 100, 1).to(device)
        self.meta_optimizer = torch.optim.SGD(self.vnet.parameters(), 1e-3,
                                              momentum=0.9, nesterov=True, weight_decay=1e-4)
        self.optimizer, self.scheduler = models.noisy_models.configure_optimizers(
            self.hparams, self.model, - 1)

    def test_step(self, batch):
        xs, ys = batch
        with torch.no_grad():
            final_yhat = self.model(xs)
            loss = F.cross_entropy(final_yhat, ys)
        acc = (final_yhat.argmax(1) == ys).float()
        logs = {
            'loss': loss,
            'acc': acc,
        }
        return logs

    def train_step(self, batch, meta_batch, it=None):
        xs, ys = batch
        meta_xs, meta_ys = meta_batch
        inner_opt = torch.optim.SGD(
            self.model.parameters(), lr=self.hparams.inner_lr)
        with higher.innerloop_ctx(self.model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):
            yhat = fmodel(xs)
            cost = F.cross_entropy(yhat, ys, reduction='none')
            cost_v = torch.reshape(cost, (len(cost), 1))
            v_lambda = self.vnet(cost_v.data)
            meta_loss = torch.mean(v_lambda * cost_v)
            diffopt.step(meta_loss)
            #
            qry_logits = fmodel(meta_xs)
            # TODO check MAE implementation
            if self.hparams.meta_loss == 'ce':
                qry_loss = F.cross_entropy(qry_logits, meta_ys)
            else:
                qry_prob = F.softmax(qry_logits, dim=-1)
                qry_prob = qry_prob[torch.arange(qry_prob.shape[0]), meta_ys]
                qry_loss = 2*torch.mean(1. - qry_prob)
            self.meta_optimizer.zero_grad()
            qry_loss.backward()
            self.meta_optimizer.step()
        final_yhat = self.model(xs)
        cost_w = F.cross_entropy(final_yhat, ys, reduction='none')
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        with torch.no_grad():
            w_new = self.vnet(cost_v)
        loss = torch.mean(cost_v * w_new)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        acc = (final_yhat.argmax(1) == ys).float()
        logs = {
            'loss': loss,
            'acc': acc,
            'avg_weight': torch.mean(w_new).cpu().float(),
        }
        if it is not None:
            iters_per_epoch = len(self.trainset) / self.hparams.batch_size
            iters_per_epoch = max(1, int(np.around(iters_per_epoch)))
            logs['epoch'] = (it / iters_per_epoch)
        return logs


def configure_optimizers(args, model, cur_iter=-1):
    iters = args.iters

    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if args.opt == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]

    LR = args.lr

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
        )
    else:
        raise NotImplementedError

    if args.lr_schedule == 'warmup-anneal':
        scheduler = utils.LinearWarmupAndCosineAnneal(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'linear':
        scheduler = utils.LinearLR(optimizer, iters, last_epoch=cur_iter)
    elif args.lr_schedule == 'const':
        scheduler = None
    elif args.lr_schedule == 'step':
        step_size = args.step_size
        gamma = args.gamma
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, last_epoch=cur_iter)
    else:
        raise NotImplementedError

    return optimizer, scheduler
