import torch
import os
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from scipy import stats
from Functions.evaluation_metrics import regular_metrics
from ddptestsampler import SequentialDistributedSampler
from HDLaQA_model import HDLaQA as predictor


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


class NetManager(object):

    def __init__(self, options, path, round, logger):
        """Prepare the network, criterion, solver, and data.
        Args:
            options, dict: Hyperparameters.
        """

        self._options = options
        self._path = path
        self._round = round
        self.logger = logger
        self.lr = self._options['base_lr']
        self.lr_ratio = self._options['lr_ratio']

        device = torch.device("cuda", self._options['local_rank'])

        model = predictor()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

        self._criterion = torch.nn.L1Loss()  # l2 loss

        from DataLoad.data_loader import datas
        train_dataset = datas(self._options['dataset'], self._path[self._options['dataset']],
                              None, self._options['train_index'], self._options['patch_size'],
                              self._options['train_patch_num'], self._options['batch_size'],  istrain=True)
        test_dataset = datas(self._options['dataset'], self._path[self._options['dataset']],
                             None, self._options['test_index'], self._options['patch_size'],
                             self._options['test_patch_num'], options['batch_size'], istrain=False)

        # Network.
        self.predictor = DistributedDataParallel(model,
                                                 device_ids=[self._options['local_rank']],
                                                 output_device=self._options['local_rank'],
                                                 find_unused_parameters=True
                                                 )

        self.train_sampler = DistributedSampler(train_dataset)
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self._options['batch_size'],
                                                        shuffle=False,
                                                        num_workers=2,
                                                        pin_memory=True,
                                                        sampler=self.train_sampler)

        self.test_sampler = SequentialDistributedSampler(test_dataset, options['batch_size'])
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=options['batch_size'],
                                                       shuffle=False,
                                                       num_workers=2,
                                                       pin_memory=True,
                                                       sampler=self.test_sampler)

        res_params = list(map(id, self.predictor.module.res.parameters()))
        self.params = filter(lambda p: id(p) not in res_params, self.predictor.module.parameters())
        self.paras = [{'params': self.params, 'lr': self.lr*10},
                      {'params': self.predictor.module.res.parameters(), 'lr': self.lr}
                      ]
        self.extractor_solver = torch.optim.Adam(self.paras, weight_decay=self._options['weight_decay'])

    def train(self):
        """Train the network."""
        best_srcc, plcc, krocc, rmse, mae = 0.0, 0.0, 0.0, 0.0, 0.0
        best_epoch = 0

        self.logger.info(' Epoch\tTrain_loss\t\tTrain_SRCC\t\tTest_SRCC\t\tTest_PLCC')

        for t in range(self._options['epochs']):
            epoch_loss = []
            pscores = []
            tscores = []
            num_total = 0
            self.train_loader.sampler.set_epoch(t)
            for X, y in self.train_loader:
                # Data.
                X = torch.as_tensor(X.cuda())
                y = torch.as_tensor(y.cuda())

                # Clear the existing gradients.
                self.extractor_solver.zero_grad()

                score = self.predictor(X)
                # Backward pass.
                loss = self._criterion(score, y.view(len(score), 1).detach())

                loss.backward()
                self.extractor_solver.step()
                # Prediction.
                num_total += y.size(0)
                pscores = pscores + score.cpu().tolist()
                tscores = tscores + y.cpu().tolist()

                epoch_loss.append(loss.item())

            train_srocc = stats.spearmanr(pscores, tscores)[0]

            test_srocc, test_plcc = self._consitency(self.test_loader)

            if test_srocc > best_srcc:
                best_srcc = test_srocc
                plcc = test_plcc
                best_epoch = t + 1

                # save model
                if dist.get_rank() == 0:
                    modelpath = os.path.join(self._path['model_saving_path'],
                                             (self._options['dataset'] + str(self._round) + '.pkl'))
                    torch.save(self.predictor.module.state_dict(), modelpath)

            self.logger.info('  %d\t\t%4.3f\t\t\t%4.4f\t\t\t%4.4f\t\t\t%4.4f' %
                             (t + 1, sum(epoch_loss)/len(epoch_loss), train_srocc, test_srocc, test_plcc))
、
            if t in self._options['schedule']:
                self.lr = self.lr * 0.1
                self.paras = [{'params': self.params, 'lr': self.lr * 10},
                              {'params': self.predictor.module.res.parameters(), 'lr': self.lr}
                              ]
                self.extractor_solver = torch.optim.Adam(self.paras, weight_decay=self._options['weight_decay'])
            dist.barrier()

        self.logger.info('Best at epoch %d, test srcc %f, plcc %f' % (best_epoch, best_srcc, plcc))

        out = {}
        out['srocc'] = best_srcc
        out['plcc'] = plcc

        return out

    def _consitency(self, data_loader):

        self.predictor.eval()

        pscores = []
        tscores = []
        with torch.no_grad():
            for X, y in data_loader:
                # Data.
                X = torch.as_tensor(X.cuda())
                y = torch.as_tensor(y.cuda())

                score = self.predictor(X)

                pscores.append(score)
                tscores.append(y)

        # gather scores from 2 ranks
        pscores = distributed_concat(torch.cat(pscores, dim=0), len(self.test_sampler.dataset))
        tscores = distributed_concat(torch.cat(tscores, dim=0), len(self.test_sampler.dataset))
        pscores = np.mean(np.reshape(np.array(pscores.cpu()), (-1, self._options['test_patch_num'])), axis=1)
        tscores = np.mean(np.reshape(np.array(tscores.cpu()), (-1, self._options['test_patch_num'])), axis=1)

        test_srcc, test_plcc = regular_metrics(pscores, tscores)

        return test_srcc, test_plcc
