import numpy as np
import torch
import warnings
import time
from PIL import Image
import random


def noise_matrix(mixing_ratio, num_classes, noise_type, class_to_idx):
    if noise_type == 'unif':
        return uniform_mix_C(mixing_ratio, num_classes)
    elif noise_type == 'flip':
        return flip_labels_C(mixing_ratio, num_classes)
    elif noise_type == 'flip2':
        return flip_labels_C_two(mixing_ratio, num_classes)
    elif noise_type == 'asym':
        return asym_noise(mixing_ratio, num_classes, class_to_idx)
    else:
        raise NotImplementedError


def asym_noise(mixing_ratio, num_classes, class_to_idx=None):
    if num_classes == 10:
        P = np.eye(10)
        # automobile <- truck
        P[9, 9], P[9, 1] = 1. - mixing_ratio, mixing_ratio
        # bird -> airplane
        P[2, 2], P[2, 0] = 1. - mixing_ratio, mixing_ratio

        # cat <-> dog
        P[3, 3], P[3, 5] = 1. - mixing_ratio, mixing_ratio
        P[5, 5], P[5, 3] = 1. - mixing_ratio, mixing_ratio
        # deer -> horse
        P[4, 4], P[4, 7] = 1. - mixing_ratio, mixing_ratio
        return P
    else:
        super_class = {}
        super_class['aquatic mammals'] = [
            'beaver', 'dolphin', 'otter', 'seal', 'whale']
        super_class['fish'] = ['aquarium_fish',
                               'flatfish', 'ray', 'shark', 'trout']
        super_class['flowers'] = [
            'orchid', 'poppy', 'rose', 'sunflower', 'tulip']
        super_class['food containers'] = [
            'bottle', 'bowl', 'can', 'cup', 'plate']
        super_class['fruit and vegetables'] = [
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
        super_class['household electrical devices'] = [
            'clock', 'keyboard', 'lamp', 'telephone', 'television']
        super_class['household furniture'] = [
            'bed', 'chair', 'couch', 'table', 'wardrobe']
        super_class['insects'] = ['bee', 'beetle',
                                  'butterfly', 'caterpillar', 'cockroach']
        super_class['large carnivores'] = [
            'bear', 'leopard', 'lion', 'tiger', 'wolf']
        super_class['large man-made outdoor things'] = ['bridge',
                                                        'castle', 'house', 'road', 'skyscraper']
        super_class['large natural outdoor scenes'] = [
            'cloud', 'forest', 'mountain', 'plain', 'sea']
        super_class['large omnivores and herbivores'] = [
            'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo']
        super_class['medium mammals'] = [
            'fox', 'porcupine', 'possum', 'raccoon', 'skunk']
        super_class['non-insect invertebrates'] = ['crab',
                                                   'lobster', 'snail', 'spider', 'worm']
        super_class['people'] = ['baby', 'boy', 'girl', 'man', 'woman']
        super_class['reptiles'] = ['crocodile',
                                   'dinosaur', 'lizard', 'snake', 'turtle']
        super_class['small mammals'] = ['hamster',
                                        'mouse', 'rabbit', 'shrew', 'squirrel']
        super_class['trees'] = ['maple_tree', 'oak_tree',
                                'palm_tree', 'pine_tree', 'willow_tree']
        super_class['vehicles 1'] = ['bicycle', 'bus',
                                     'motorcycle', 'pickup_truck', 'train']
        super_class['vehicles 2'] = ['lawn_mower',
                                     'rocket', 'streetcar', 'tank', 'tractor']
        pass
        P = np.eye(100)
        for k, v in super_class.items():
            for idx in range(5):
                src_class, tgt_class = class_to_idx[v[idx]], class_to_idx[v[(
                    idx+1) % 5]]
                P[src_class, src_class], P[src_class, tgt_class] = 1. - \
                    mixing_ratio, mixing_ratio
        return P


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)


def flip_labels_C(corruption_prob, num_classes):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    # np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C


def flip_labels_C_two(corruption_prob, num_classes):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    # np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i],
                              2, replace=False)] = corruption_prob / 2
    return C


class ClothingDataset(torch.utils.data.Dataset):
    def __init__(self, transform, mode, path, noisy_val_samples=False):
        self.train_imgs = []
        self.test_imgs = []
        self.val_imgs = []
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.transform = transform
        self.mode = mode
        self.noisy_val_samples = noisy_val_samples
        if self.mode == 'train':
            train_imgs = []
            with open(path+'noisy_train_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = path+l[7:]
                train_imgs.append(img_path)
        elif self.mode == 'test':
            with open(path+'clean_test_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = path+l[7:]
                self.test_imgs.append(img_path)
        elif self.mode == 'val':
            with open(path+'clean_val_key_list.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                img_path = path+l[7:]
                self.val_imgs.append(img_path)
        else:
            raise NotImplementedError

        if (self.mode == 'train') or (self.mode == 'val' and self.noisy_val_samples):
            with open(path+'noisy_label_kv.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = path+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        else:
            with open(path+'clean_label_kv.txt', 'r') as f:
                lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = path+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])
        if self.mode == 'train':
            num_samples = 64*2000
            random.shuffle(train_imgs)
            class_num = torch.zeros(14)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples/14) and len(self.train_imgs) < num_samples:
                    self.train_imgs.append(impath)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)
        #

    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            if self.noisy_val_samples:
                target = self.train_labels[img_path]
            else:
                target = self.test_labels[img_path]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)

        return img, target

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_imgs)
        elif self.mode == 'test':
            return len(self.test_imgs)
        elif self.mode == 'val':
            return len(self.val_imgs)


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def agg_all_metrics(outputs):
    if len(outputs) == 0:
        return outputs
    res = {}
    keys = [k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict)]
    for k in keys:
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        if k != 'epoch':
            res[k] = np.mean(all_logs)
        else:
            res[k] = all_logs[-1]
    return res


def viz_array_grid(array, rows, cols, padding=0, channels_last=False, normalize=False, **kwargs):
    # normalization
    '''
    Args:
        array: (N_images, N_channels, H, W) or (N_images, H, W, N_channels)
        rows, cols: rows and columns of the plot. rows * cols == array.shape[0]
        padding: padding between cells of plot
        channels_last: for Tensorflow = True, for PyTorch = False
        normalize: `False`, `mean_std`, or `min_max`
    Kwargs:
        if normalize == 'mean_std':
            mean: mean of the distribution. Default 0.5
            std: std of the distribution. Default 0.5
        if normalize == 'min_max':
            min: min of the distribution. Default array.min()
            max: max if the distribution. Default array.max()
    '''
    array = tonp(array)
    if not channels_last:
        array = np.transpose(array, (0, 2, 3, 1))

    array = array.astype('float32')

    if normalize:
        if normalize == 'mean_std':
            mean = kwargs.get('mean', 0.5)
            mean = np.array(mean).reshape((1, 1, 1, -1))
            std = kwargs.get('std', 0.5)
            std = np.array(std).reshape((1, 1, 1, -1))
            array = array * std + mean
        elif normalize == 'min_max':
            min_ = kwargs.get('min', array.min())
            min_ = np.array(min_).reshape((1, 1, 1, -1))
            max_ = kwargs.get('max', array.max())
            max_ = np.array(max_).reshape((1, 1, 1, -1))
            array -= min_
            array /= max_ + 1e-9

    batch_size, H, W, channels = array.shape
    assert rows * cols == batch_size

    if channels == 1:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1)))
        array = array[:, :, :, 0]
    elif channels == 3:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1),
                          3))
    else:
        raise TypeError('number of channels is either 1 of 3')

    for i in range(rows):
        for j in range(cols):
            img = array[i * cols + j]
            start_h = i * padding + i * H
            start_w = j * padding + j * W
            canvas[start_h: start_h + H, start_w: start_w + W] = img

    canvas = np.clip(canvas, 0, 1)
    canvas *= 255.0
    canvas = canvas.astype('uint8')
    return canvas


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    return x.detach().cpu().numpy()


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(
                lr * np.minimum(-self.last_epoch * 1. / self.num_epochs + 1., 1.), 0.))
        return res


class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            # ref: https://github.com/pytorch/pytorch/blob/2de4f245c6b1e1c294a8b2a9d7f916d43380af4b/torch/optim/lr_scheduler.py#L493
            le = self.last_epoch - self.warm_up
            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max)) *
                    group['lr']
                    for group in self.optimizer.param_groups]


class BaseLR(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
