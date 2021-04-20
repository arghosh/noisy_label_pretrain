import numpy as np
import random
import pickle
import os
from utils import utils
import torch
import torch.backends.cudnn as cudnn
import argparse
from torch.nn import functional as F
import torchvision.transforms as transforms
import models
from PIL import Image
    

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x.transpose((0, 2, 3, 1))
        self.y= y
        self.transform= transform
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        target = self.y[idx]
        img = Image.fromarray(self.x[idx])
        img = self.transform(img)
        return img,target

def data_loader(params):
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
    test_x, test_y = load_test()
    train_x,train_y = load_train()
    trainset = ImageNetDataset(train_x, train_y,transform=train_transform)
    valset = ImageNetDataset(test_x, test_y,transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.workers)
    valid_loader = torch.utils.data.DataLoader(
        valset,
        num_workers=params.workers,
        shuffle=False,
        batch_size=params.test_bs, drop_last=False,
    )
    return train_loader, valid_loader


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_train(data_folder='/mnt/nfs/scratch1/arighosh/data/imagenet/train/', img_size=32):
    def f(idx):
        data_file = os.path.join(data_folder, 'train_data_batch_')
        d = unpickle(data_file + str(idx))
        x = d['data']
        y = d['labels']
        y = [i-1 for i in y]
        data_size = x.shape[0]
        img_size2 = img_size * img_size
        x = np.dstack(
            (x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
        return [x,y]
    data = [f(idx) for idx in range(1,11)]
    all_x =  np.concatenate([d[0] for d in data],axis=0)
    all_y = []
    for d in data:
        all_y.extend(d[1])

    return all_x,all_y

def load_test(data_folder='/mnt/nfs/scratch1/arighosh/data/imagenet/val/', img_size=32):
    data_file = os.path.join(data_folder, 'val_data')
    d = unpickle(data_file )
    x = d['data']
    y = d['labels']
    y = [i-1 for i in y]
    data_size = x.shape[0]
    img_size2 = img_size * img_size

    x = np.dstack(
        (x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return x, y
    


def add_learner_params():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--name', default='',help='Name for the experiment')
    parser.add_argument('--data', default='cifar',help='keep cifar to allow resnet50 to make the changes with 32x32 image')
    parser.add_argument('--arch', default='ResNet50',help='arch')
    parser.add_argument('--nodes', default='', help='slurm nodes for the experiment')
    parser.add_argument('--slurm_partition', default='',
                        help='slurm partitions for the experiment')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='Base learning rate')
    parser.add_argument('--test_bs', default=512, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='The number of data loader workers')
    parser.add_argument('--seed', default=222, type=int, help='Random seed')
    #
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--iters', default=500000, type=int,
                        help='The number of optimizer updates')

    params = parser.parse_args()
    return params


def main():
    args = add_learner_params()
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    if args.neptune:
        import neptune
        project = "arighosh/cvprw21"
        neptune.init(project_qualified_name=project,
                     api_token=os.environ["NEPTUNE_API_TOKEN"])
        neptune.create_experiment(
            name=args.name, send_hardware_metrics=False, params=vars(args))
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.cuda:
        assert device.type == 'cuda', 'no gpu found!'

    encoder = models.encoder.EncodeProject(args)
    n_classes = 1000
    linear_model = torch.nn.Linear(2048, n_classes).to(device)
    linear_model.weight.data.zero_()
    linear_model.bias.data.zero_()
    model = models.encoder.Model(encoder, linear_model).to(device)

    #
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,weight_decay=1e-3,
            momentum=0.9,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100000, gamma=0.5, last_epoch=-1)
    #
    cur_iter,epoch =0,0
    continue_training = cur_iter < args.iters
    best_acc = 0.
    train_loader, test_loader =data_loader(args)
    while continue_training:
        train_logs,test_logs = [], []
        model.train()
        epoch +=1
        for _, batch in enumerate(train_loader):
            cur_iter += 1
            batch = [x.to(device) for x in batch]
            h,y  = batch
            p = model(h)
            loss = F.cross_entropy(p, y)
            acc = (p.argmax(1) == y).float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            logs = {'loss':loss, 'acc':acc}
            train_logs.append({k: utils.tonp(v) for k, v in logs.items()})
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = [x.to(device) for x in batch]
                h,y  = batch
                p = model(h)
                loss = F.cross_entropy(p, y)
                acc = (p.argmax(1) == y).float()
                logs = {'loss':loss, 'acc':acc}
                test_logs.append(logs)
        test_logs = utils.agg_all_metrics(test_logs)
        train_logs = utils.agg_all_metrics(train_logs)
        if float(test_logs['acc'])<best_acc-0.1 and cur_iter>=50000:
            continue_training = False
            break
        if float(test_logs['acc'])>best_acc:
            save_model(model,args)
        best_acc = max(best_acc, float(test_logs['acc']))
        test_logs['best_acc'] = best_acc
        if args.neptune:
            for k, v in test_logs.items():
                neptune.log_metric('test_'+k, float(v))
            for k,v in train_logs.items():
                neptune.log_metric('train_'+k, float(v))
            neptune.log_metric('epoch', epoch)
            neptune.log_metric('train_iter', cur_iter)

        if cur_iter >= args.iters:
            continue_training = False
            break
    
def save_model(model,args):
    fname = os.path.join('output/checkpoint_'+args.name+'.pth.tar')
    ckpt = {'state_dict': model.state_dict()}
    torch.save(ckpt, fname)


if __name__ == '__main__':
    main()