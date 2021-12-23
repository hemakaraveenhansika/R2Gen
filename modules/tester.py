import os
from abc import abstractmethod

import json
import time
import torch
import pandas as pd
from numpy import inf, inexact
from tqdm import tqdm
from transformers import BertTokenizer


class _BaseR2GenTester(object):
    def __init__(self, visual_extractor_model, r2gen_model, criterion, metric_ftns, args):
        self.args = args

        # setup GPU device if available, move r2gen_model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.visual_extractor_model = visual_extractor_model.to(self.device)
        self.r2gen_model = r2gen_model.to(self.device)
        if len(device_ids) > 1:
            self.visual_extractor_model = torch.nn.DataParallel(visual_extractor_model, device_ids=device_ids)
            self.r2gen_model = torch.nn.DataParallel(r2gen_model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric_test = 'test_' + args.monitor_metric


        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume_contrastive_model is not None:
            self._load_visual_extractor_model_checkpoint(args.resume_contrastive_model)

        if args.resume_r2gen is not None:
            self._load_r2gen_model_checkpoint(args.resume_r2gen)


    @abstractmethod
    def _test_R2Gen(self):
        raise NotImplementedError

    def test(self):

        print("start r2gen model test")
        logs, result_caption = self._test_R2Gen()

        # print logged informations to the screen
        for key, value in logs.items():
            print('\t{:15s}: {}'.format(str(key), value))
        self.__save_json(logs, 'R2Gen_model_test_logs')
        self.__save_json(result_caption, 'R2Gen_model_test_results')
        
        print("end r2gen model test")


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
            print( "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))

        return device, list_ids


    def _load_visual_extractor_model_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading visual_extractor_model checkpoint: {} ...".format(resume_path))

        try:
            checkpoint = torch.load(resume_path)
            self.visual_extractor_model.load_state_dict(checkpoint['visual_extractor_model'])
            print("Checkpoint loaded. resume visual_extractor_model from epoch {}".format(checkpoint['epoch']))
        except Exception as err:
            print("[Load visual_extractor_model Failed {}!]\n".format(err))

    def _load_r2gen_model_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading r2gen_model checkpoint: {} ...".format(resume_path))

        try:
            checkpoint = torch.load(resume_path)
            self.mnt_best = checkpoint['monitor_best']
            self.r2gen_model.load_state_dict(checkpoint['r2gen_model'])

            print("Checkpoint loaded. resume_r2gen training from epoch {}".format(checkpoint['epoch']))
        except Exception as err:
            print("[Load r2gen_model Failed {}!]\n".format(err))




class R2GenTester(_BaseR2GenTester):
    def __init__(self, visual_extractor_model, r2gen_model, criterion, metric_ftns, args, test_dataloader):
        super(R2GenTester, self).__init__(visual_extractor_model, r2gen_model, criterion, metric_ftns, args)

        self.test_dataloader = test_dataloader

    def _test_R2Gen(self):

        result_caption = {}
        self.visual_extractor_model.eval()
        self.r2gen_model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []

            # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
            for images_id, images, reports_ids, reports_masks, captions in tqdm(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to( self.device), reports_masks.to(self.device)

                att_feats, fc_feats = self.visual_extractor_model(images)
                print(att_feats.shape, fc_feats.shape)

                output = self.r2gen_model(att_feats, fc_feats, mode='sample')
                reports = self.r2gen_model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.r2gen_model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
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
