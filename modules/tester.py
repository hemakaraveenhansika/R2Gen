import os
from abc import abstractmethod

import json
import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _test_R2Gen(self, epoch):
        raise NotImplementedError

    def test(self):

        print("start test")
        logs, result_caption = self._test_R2Gen()

        for key, value in logs.items():
            print('\t{:15s}: {}'.format(str(key), value))

        self.__save_json(logs, 'R2Gen_base_model_test_logs')
        self.__save_json(result_caption, 'R2Gen_base_model_test_results')

    def __save_json(self, result, record_name):
        result_path = self.args.record_dir

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(record_name)), 'w') as f:
            json.dump(result, f)
        print("logs saved in", result_path)


    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        try:
            resume_path = str(resume_path)
            print("Loading checkpoint: {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        except Exception as err:
            print("[Load visual_extractor_and_bert_model Failed {}!]\n".format(err))

class Tester(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        # self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _test_R2Gen(self):

        result_caption = {}
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []

            # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
            for images_id, images, reports_ids, reports_masks in tqdm(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

                print(ground_truths)
                print(reports, "\n")
                print("each results\n")
                for index in range(len(images_id)):
                    image_id, real_sent, pred_sent = images_id[index], ground_truths[index], reports[index]
                    print(image_id, real_sent, pred_sent)
                    result_caption[image_id] = {
                        'Image id': image_id,
                        'Real Sent': real_sent,
                        'Pred Sent': pred_sent,
                    }

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log = {'test_' + k: v for k, v in test_met.items()}



        return log, result_caption
