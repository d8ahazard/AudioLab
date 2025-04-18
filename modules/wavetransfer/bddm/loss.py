# Adapted from https://github.com/tencent-ailab/bddm under the Apache-2.0 license.

#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  Implements Training Losses for BDDMs
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################


import torch
import torch.nn as nn


class StepLoss(nn.Module):

    def __init__(self, config, diff_params):
        """
        Step Loss Class, implements BDDM's step loss (see Eq. 14 in BDDM's paper)

        Parameters:
            config (namespace): BDDM Configuration
            diff_params (dict): Dictionary that stores pre-computed diffusion parameters
        """
        super().__init__()
        self.config = config
        self.num_steps = diff_params["T"]
        self.alpha = diff_params["alpha"]
        self.tau = diff_params["tau"]

    def forward(self, model, features):
        """
        Compute the training loss for learning phi

        Parameters:
            model (nn.Module):  the score network & the schedule network
            features (dict): a dict containing audio and spectrogram features
        Returns:
            step loss (tensor): shape=(batch size,)
        """
        audio = features['audio'].unsqueeze(1)
        spectrogram = features['spectrogram']

        batch_size = audio.size(0)
        ts = torch.randint(self.tau, self.num_steps-self.tau, size=(batch_size,)).cuda()
        alpha_cur = self.alpha.index_select(0, ts).view(batch_size, 1, 1)
        alpha_nxt = self.alpha.index_select(0, ts+self.tau).view(batch_size, 1, 1)
        b_nxt = 1 - (alpha_nxt / alpha_cur)**2.
        delta = (1 - alpha_cur**2.).sqrt()
        z = torch.normal(0, 1, size=audio.shape).cuda()
        noisy_audio = alpha_cur * audio + delta * z
        e = model(noisy_audio.squeeze(1), spectrogram, alpha_cur.squeeze(dim=(1, 2)))
        beta_bounds = (b_nxt.view(batch_size, 1), delta.view(batch_size, 1)**2.)
        b_hat = model.schedule_net(noisy_audio.squeeze(1), beta_bounds)
        delta, b_hat, z, e = delta.squeeze(1), b_hat.squeeze(1), z.squeeze(1), e.squeeze(1)
        phi_loss = delta**2. / (2. * (delta**2. - b_hat))
        phi_loss = phi_loss * (z - b_hat / (delta**2.) * e).square()
        phi_loss = phi_loss + torch.log(1e-8 + delta**2. / (b_hat + 1e-8)) / 4.
        phi_loss = phi_loss.sum(-1) + (b_hat / delta**2 - 1) / 2. * audio.size(-1)
        return phi_loss
