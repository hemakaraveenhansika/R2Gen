import os
from abc import abstractmethod

import json
import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
from transformers import BertTokenizer

class BaseContrastiveTrainer(object):
    def __init__(self, visual_extractor_model, bert_model, NTXentLoss, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move bert_model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.visual_extractor_model = visual_extractor_model.to(self.device)
        self.bert_model = bert_model.to(self.device)

        # if self.args.cuda:
        #     self.bert_model = bert_model.cuda()

        if len(device_ids) > 1:
            self.visual_extractor_model = torch.nn.DataParallel(visual_extractor_model, device_ids=device_ids)
            self.bert_model = torch.nn.DataParallel(bert_model, device_ids=device_ids)

        self.NTXentLoss = NTXentLoss
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = 'min'
        self.mnt_metric = 'val_contrastive_loss'
        self.mnt_metric_test = 'test_contrastive_loss'
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume_contrastive_model is not None:
            self.load_visual_extractor_and_bert_model_checkpoint(args.resume_contrastive_model)

        self.bert_tokenizer = self._init_bert_tokenizer()
        self.nt_xent_criterion = self._init_nt_xent()

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}


    def _init_nt_xent(self):
        nt_xent = self.NTXentLoss(self.device, self.args.batch_size, self.args.contrastive_temperature, self.args.use_cosine_similarity, self.args.alpha_weight)
        if self.args.cuda:
            nt_xent = nt_xent.cuda()
        return nt_xent

    def _init_bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        return tokenizer

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        complete_reslts = {}

        print("start contrastive learn model train")
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_reslts = {}
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}

            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                epoch_reslts[str(key)]= value
                print('\t{:15s}: {}'.format(str(key), value))

            complete_reslts[epoch] = epoch_reslts

            # evaluate contrastive_model performance according to configured metric, save best checkpoint as contrastive_model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether bert_model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "contrastive_model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format( self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self.save_contrastive_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()
        self.__save_json(complete_reslts, 'contrastive_model_train_logs')

        print("end contrastive learn model train")

    def __save_json(self, result, record_name):
        result_path = self.args.record_dir

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(record_name)), 'w') as f:
            json.dump(result, f)
        print("logs saved in", result_path)

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_contrastive_model_from'] = 'val'
        self.best_recorder['test']['best_contrastive_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        print(self.best_recorder['val'])
        print(self.best_recorder['test'])

        record_table.to_csv(record_path, index=False)

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
        print("device, list_ids")
        print(device, list_ids)
        return device, list_ids

    def save_contrastive_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'visual_extractor_model': self.visual_extractor_model.state_dict(),
            'bert_model': self.bert_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_contrastive_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'contrastive_model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: contrastive_model_best.pth ...", epoch)

    def load_visual_extractor_and_bert_model_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading visual_extractor_model checkpoint: {} ...".format(resume_path))

        try:
            checkpoint = torch.load(resume_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            self.visual_extractor_model.load_state_dict(checkpoint['visual_extractor_model'])
            self.bert_model.load_state_dict(checkpoint['bert_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("Checkpoint loaded. resume visual_extractor_model from epoch {}".format(checkpoint['epoch']))
        except Exception as err:
            print("[Load visual_extractor_and_bert_model Failed {}!]\n".format(err))


    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format('contrastive_loss'))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format('contrastive_loss'))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class BaseR2GenTrainer(object):
    def __init__(self, visual_extractor_model, r2gen_model, criterion, metric_ftns, optimizer, args):
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
        self.optimizer = optimizer

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

        if args.resume_contrastive_model is not None:
            self._load_visual_extractor_model_checkpoint(args.resume_contrastive_model)

        if args.resume_r2gen is not None:
            self._load_r2gen_model_checkpoint(args.resume_r2gen)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        print("start r2gen model train")
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate r2gen_model performance according to configured metric, save best checkpoint as r2gen_model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether r2gen_model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "r2gen_model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self.save_r2gen_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()
        print("end r2gen model train")

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_r2gen_model_from'] = 'val'
        self.best_recorder['test']['best_r2gen_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

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

    def save_r2gen_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'r2gen_model': self.r2gen_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_r2gen_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'r2gen_model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: r2gen_model_best.pth ...")

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
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            self.r2gen_model.load_state_dict(checkpoint['r2gen_model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded. resume_r2gen training from epoch {}".format(checkpoint['epoch']))
        except Exception as err:
            print("[Load r2gen_model Failed {}!]\n".format(err))



    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class ContrastiveModelTrainer(BaseContrastiveTrainer):
    def __init__(self, visual_extractor_model, bert_model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(ContrastiveModelTrainer, self).__init__(visual_extractor_model, bert_model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):

        train_contrastive_losss = 0
        self.visual_extractor_model.train()
        self.bert_model.train()

        # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
        for images_id, images, reports_ids, reports_masks, captions in tqdm(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)

            att_feats, fc_feats = self.visual_extractor_model(images)

            bert_tokens = self.bert_tokenizer(list(captions), return_tensors="pt", padding=True, truncation=True)
            bert_tokens = bert_tokens.to(self.device)

            text_features = self.bert_model(bert_tokens)
            # print(att_feats.shape, fc_feats.shape)

            train_loss = self.nt_xent_criterion(fc_feats, text_features)
            train_contrastive_losss += train_loss.item()
            self.optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_value_(self.visual_extractor_model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_contrastive_loss': train_contrastive_losss / len(self.train_dataloader)}


        valid_contrastive_losss = 0
        self.visual_extractor_model.eval()
        self.bert_model.eval()
        with torch.no_grad():

            # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
            for images_id, images, reports_ids, reports_masks, captions in tqdm(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to( self.device), reports_masks.to(self.device)

                att_feats, fc_feats = self.visual_extractor_model(images)

                bert_tokens = self.bert_tokenizer(list(captions), return_tensors="pt", padding=True, truncation=True)
                bert_tokens = bert_tokens.to(self.device)

                text_features = self.bert_model(bert_tokens)
                # print(fc_feats.shape, text_features.shape)

                valid_loss = self.nt_xent_criterion(fc_feats, text_features)
                valid_contrastive_losss += valid_loss.item()
            log.update(**{'val_contrastive_loss': valid_contrastive_losss / len(self.val_dataloader)})


        self.lr_scheduler.step()

        return log


class R2GenTrainer(BaseR2GenTrainer):
    def __init__(self, visual_extractor_model, r2gen_model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(R2GenTrainer, self).__init__(visual_extractor_model, r2gen_model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch(self, epoch):

        train_loss = 0
        self.visual_extractor_model.eval()
        self.r2gen_model.train()

        # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
        for images_id, images, reports_ids, reports_masks, captions in tqdm(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)

            att_feats, fc_feats = self.visual_extractor_model(images)
            # print(att_feats.shape, fc_feats.shape)

            output = self.r2gen_model(att_feats, fc_feats, reports_ids, mode='train')

            # print(output.shape, reports_ids.shape, reports_masks.shape)
            loss = self.criterion(output, reports_ids, reports_masks)

            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.r2gen_model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}


        valid_loss = 0
        self.visual_extractor_model.eval()
        self.r2gen_model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []

            # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
            for images_id, images, reports_ids, reports_masks, captions in tqdm(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to( self.device), reports_masks.to(self.device)

                att_feats, fc_feats = self.visual_extractor_model(images)
                output = self.r2gen_model(att_feats, fc_feats, mode='sample')

                # print(output.shape, reports_ids.shape, reports_masks.shape)
                # loss = self.criterion(output, reports_ids, reports_masks)
                # valid_loss += loss.item()

                reports = self.r2gen_model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.r2gen_model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)}, {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            # log.update(**{'valid_loss': valid_loss / len(self.val_dataloader)})


        self.visual_extractor_model.eval()
        self.r2gen_model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []

            # for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
            for images_id, images, reports_ids, reports_masks, captions in tqdm(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to( self.device), reports_masks.to(self.device)

                att_feats, fc_feats = self.visual_extractor_model(images)
                # print(att_feats.shape, fc_feats.shape)

                output = self.r2gen_model(att_feats, fc_feats, mode='sample')
                reports = self.r2gen_model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.r2gen_model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)

                # print(ground_truths)
                # print(reports, "\n")

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log
