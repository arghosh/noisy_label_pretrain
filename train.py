import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import models
from utils.logger import Logger
import yaml
import argparse
from utils import utils
import sys


def add_learner_params():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--problem', default='finetune',
                        help='The problem to train',
                        choices=models.REGISTERED_MODELS,
                        )
    parser.add_argument('--name', default='demo',
                        help='Name for the experiment')
    parser.add_argument('--nodes', default='',
                        help='slurm nodes for the experiment')
    parser.add_argument('--slurm_partition', default='',
                        help='slurm partitions for the experiment')
    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-const')
    parser.add_argument('--loss', default='ce')
    parser.add_argument('--opt', default='sgd',
                        help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=-1, type=int,
                        help='The number of optimizer updates')
    parser.add_argument('--warmup', default=0, type=float,
                        help='The number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Base learning rate')
    parser.add_argument('--inner_lr', default=0.1, type=float,
                        help='inner learning rate')
    parser.add_argument('--q', default=0.66, type=float,
                        help='generalized ce loss q value')
    parser.add_argument('--wd', '--weight_decay',
                        default=1e-4, type=float, dest='weight_decay')
    # noise params
    parser.add_argument('--corruption_prob', type=float, default=0.9,
                        help='label noise')
    parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                        help='Type of corruption ("unif" or "flip" or "flip2" or "asym").')

    # trainer params
    parser.add_argument('--save_freq', default=1000000000000,
                        type=int, help='Frequency to save the model')
    parser.add_argument('--log_freq', default=100,
                        type=int, help='Logging frequency')
    parser.add_argument('--eval_freq', default=100000000000,
                        type=int, help='Evaluation frequency')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='The number of data loader workers')
    parser.add_argument('--seed', default=222, type=int, help='Random seed')
    # parallelizm params:
    parser.add_argument('--test_bs', default=256, type=int)
    parser.add_argument('--encoder_ckpt', default='',
                        help='Path to the encoder checkpoint')
    parser.add_argument('--encoder_type', default='simclr',
                        help='pretrained "imagenet" or "simclr" for simclr')
    parser.add_argument('--config', default='configs/cifar10.yaml', type=str,
                        help='the number of nodes (scripts launched)',
                        )
    #
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--neptune', action='store_true')

    params = parser.parse_args()
    d = vars(params)
    if params.config:
        with open(params.config, 'r') as stream:
            try:
                params_2 = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    for k, v in params_2.items():
        d[k] = v
    return params


def main():
    args = add_learner_params()
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    args.root = 'logs/'+args.name+'/'

    if args.neptune:
        import neptune
        project = "arighosh/pretrain_noisy_label"
        neptune.init(project_qualified_name=project,
                     api_token=os.environ["NEPTUNE_API_TOKEN"])
        neptune.create_experiment(
            name=args.name, send_hardware_metrics=False, params=vars(args))
    fmt = {
        'train_time': '.3f',
        'val_time': '.3f',
        'train_epoch': '.1f',
        'lr': '.1e',
    }
    logger = Logger('logs', base=args.root, fmt=fmt)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.cuda:
        assert device.type == 'cuda', 'no gpu found!'

    with open(args.root+'config.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    # create model
    model = models.REGISTERED_MODELS[args.problem](args, device=device)
    cur_iter = 0
    # Data loading code
    model.prepare_data()

    continue_training = cur_iter < args.iters
    data_time, it_time = 0, 0
    best_acc = 0.
    best_valid_acc, best_acc_with_valid = 0, 0

    while continue_training:
        train_loader, test_loader, valid_loader, meta_loader = model.dataloaders(
            iters=args.iters)
        train_logs = []
        model.train()
        start_time = time.time()
        for _, batch in enumerate(train_loader):
            cur_iter += 1
            batch = [x.to(device) for x in batch]
            data_time += time.time() - start_time
            logs = {}
            if args.problem not in {'finetune'}:
                meta_batch = next(iter(meta_loader))
                meta_batch = [x.to(device) for x in meta_batch]
                logs = model.train_step(batch, meta_batch, cur_iter)
            else:
                logs = model.train_step(batch, cur_iter)

            # save logs for the batch
            train_logs.append({k: utils.tonp(v) for k, v in logs.items()})
            if cur_iter % args.eval_freq == 0 or cur_iter >= args.iters:
                test_start_time = time.time()
                test_logs, valid_logs = [], []
                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        batch = [x.to(device) for x in batch]
                        logs = model.test_step(batch)
                        test_logs.append(logs)
                    for batch in valid_loader:
                        batch = [x.to(device) for x in batch]
                        logs = model.test_step(batch)
                        valid_logs.append(logs)
                model.train()
                test_logs = utils.agg_all_metrics(test_logs)
                valid_logs = utils.agg_all_metrics(valid_logs)
                best_acc = max(best_acc, float(test_logs['acc']))
                test_logs['best_acc'] = best_acc
                if float(valid_logs['acc']) > best_valid_acc:
                    best_valid_acc = float(valid_logs['acc'])
                    best_acc_with_valid = float(test_logs['acc'])
                test_logs['best_acc_with_valid'] = best_acc_with_valid
                #

                if args.neptune:
                    for k, v in test_logs.items():
                        neptune.log_metric('test_'+k, float(v))
                    for k, v in valid_logs.items():
                        neptune.log_metric('valid_'+k, float(v))
                    test_it_time = time.time()-test_start_time
                    neptune.log_metric('test_it_time', test_it_time)
                    neptune.log_metric('test_cur_iter', cur_iter)
                logger.add_logs(cur_iter, test_logs, pref='test_')
            it_time += time.time() - start_time
            if (cur_iter % args.log_freq == 0 or cur_iter >= args.iters):
                train_logs = utils.agg_all_metrics(train_logs)
                if args.neptune:
                    for k, v in train_logs.items():
                        neptune.log_metric('train_'+k, float(v))
                    neptune.log_metric('train_it_time', it_time)
                    neptune.log_metric('train_data_time', data_time)
                    neptune.log_metric(
                        'train_lr', model.optimizer.param_groups[0]['lr'])
                    neptune.log_metric('train_cur_iter', cur_iter)
                logger.add_logs(cur_iter, train_logs, pref='train_')
                logger.add_scalar(
                    cur_iter, 'lr', model.optimizer.param_groups[0]['lr'])
                logger.add_scalar(cur_iter, 'data_time', data_time)
                logger.add_scalar(cur_iter, 'it_time', it_time)
                logger.iter_info()
                logger.save()
                data_time, it_time = 0, 0
                train_logs = []
            if cur_iter >= args.iters:
                continue_training = False
                break
            start_time = time.time()


if __name__ == '__main__':
    main()
