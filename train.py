import argparse
import torch
import os
import random
import numpy as np
import torch.backends.cudnn
import torch.distributed as dist
from Functions.logger import get_logger
from model_solver import NetManager

local = False


def main(index, options, path, logger):
    """The main function."""

    srcc_all = np.zeros((1, options['round']), dtype=float)
    plcc_all = np.zeros((1, options['round']), dtype=float)

    for i in range(0, options['round']):
        logger.info('')
        logger.info('                                      Round %d                                         ' % (i + 1))
        # set random seed
        seed = i + 1
        random.seed(seed)
        # randomly split train-test set
        random.shuffle(index)
        train_index = index[0:round(0.8 * len(index))]
        test_index = index[round(0.8 * len(index)):len(index)]
        options['train_index'] = train_index
        options['test_index'] = test_index

        if (options['dataset'] == 'live') | (options['dataset'] == 'csiq') | (options['dataset'] == 'tid2013') | (
                options['dataset'] == 'mlive'):
            logger.info('Test index {}'.format(options['test_index']))

        manager = NetManager(options, path, round=i+1, logger=logger)
        out = manager.train()
        srcc_all[0][i] = out['srocc']
        plcc_all[0][i] = out['plcc']


    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)

    logger.info('-------------------------------------------------------------------------------------------------')
    logger.info('srocc:{}'.format(srcc_all))
    logger.info('plcc :{}'.format(plcc_all))
    logger.info('-------------------------------------------------------------------------------------------------')
    logger.info('Sort_srocc:{}'.format(np.sort(srcc_all)))
    logger.info('-------------------------------------------------------------------------------------------------')

    logger.info('Avg_srocc:%4.4f' % srcc_mean)
    logger.info('Avg_plcc :%4.4f' % plcc_mean)
    logger.info('-------------------------------------------------------------------------------------------------')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train IDF for BIQA.')
    parser.add_argument("--local_rank", dest='local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('--round', dest='round', type=int, default=10, help='Rounds.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-4, help='Base learning rate for training.')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--schedule', dest='schedule', type=int, nargs='+', default=[5, 10, 15],
                        help='Decrease learning rate at these epochs.')

    parser.add_argument('--dataset', dest='dataset', type=str, default='live',
                        help='dataset: live|livec')
    parser.add_argument('--dtype', dest='dtype', type=int, default=5,
                        help='Distortion types')  # live:5; csiq:6; tid2013:24, livemd:2
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Patch size.')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25,
                        help='Patch number of training data.')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25,
                        help='Patch number of testing data.')
    args = parser.parse_args()

    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    if args.dataset == 'live':
        index = list(range(0, 29))  # 29

    elif args.dataset == 'livec':
        index = list(range(0, 1162))  # 1162

    # initial DDP backend
    dist.init_process_group(backend="nccl", init_method='env://')
    torch.cuda.set_device(args.local_rank)

    ops = {
        'local_rank': args.local_rank,
        'round': args.round,
        'base_lr': args.base_lr,
        'schedule': args.schedule,
        'gamma': args.gamma,
        'lr_ratio': args.lr_ratio,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'dataset': args.dataset,
        'dtype': args.dtype,
        'train_index': [],
        'test_index': [],
        'patch_size': args.patch_size,
        'train_patch_num': args.train_patch_num,
        'test_patch_num': args.test_patch_num,
    }

    data_root = '/home/vip/hdd2/IQA'
    save_log_root = '/home/vip/jl/HDA/logger'
    save_model_root = '/home/vip/jl/HDA/models'
    if not os.path.exists(save_log_root):
        os.mkdir(save_log_root)
    if not os.path.exists(save_model_root):
        os.mkdir(save_model_root)

    pt = {
        'live': os.path.join(data_root, 'databaserelease2'),
        'livec': os.path.join(data_root, 'ChallengeDB_release'),

        'logger_saving_path': os.path.join(save_log_root, (args.dataset + 'HDA')),
        'model_saving_path': save_model_root,
    }

    log = get_logger(pt['logger_saving_path'], args.local_rank)

    log.info('use {} gpus!'.format(torch.cuda.device_count()))
    log.info('Database {}'.format(ops['dataset']))
    main(index, ops, pt, log)
