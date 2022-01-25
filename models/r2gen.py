import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from models.r2gen_visual_extractor import R2GenVisualExtractorModel

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        # self.visual_extractor = VisualExtractor(args)
        self.visual_extractor_model = R2GenVisualExtractorModel(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        if args.resume_contrastive_model is not None:
            self._load_visual_extractor_model_checkpoint(args.resume_contrastive_model)

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def _load_visual_extractor_model_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading visual_extractor_model checkpoint: {} ...".format(resume_path))

        try:
            checkpoint = torch.load(resume_path)
            self.visual_extractor_model.load_state_dict(checkpoint['visual_extractor_model'])
            print("Checkpoint loaded. resume visual_extractor_model from epoch {}".format(checkpoint['epoch']))
        except Exception as err:
            print("[Load visual_extractor_model Failed {}!]\n".format(err))

    def forward_iu_xray(self, images, targets=None, mode='train'):
        # att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        # att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        # att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        att_feats_0, fc_feats_0 = self.visual_extractor_model(images)
        att_feats_1, fc_feats_1 = self.visual_extractor_model(images)
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        # att_feats, fc_feats = self.visual_extractor(images)
        att_feats, fc_feats = self.visual_extractor_model(images)

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

