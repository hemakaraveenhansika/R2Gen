import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor

class R2GenVisualExtractorModel(nn.Module):
    def __init__(self, args):
        super(R2GenVisualExtractorModel, self).__init__()
        self.args = args
        self.visual_extractor = VisualExtractor(args)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        return att_feats, fc_feats

    def forward_mimic_cxr(self, images):
        att_feats, fc_feats = self.visual_extractor(images)

        return att_feats, fc_feats

