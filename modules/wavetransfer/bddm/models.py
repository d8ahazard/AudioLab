# Adapted from https://github.com/tencent-ailab/bddm under the Apache-2.0 license.

#!/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
#  Globally Attentive Locally Recurrent (GALR) Networks
#  (https://arxiv.org/abs/2101.05014)
#
#  Author: Max W. Y. Lam (maxwylam@tencent.com)
#  Copyright (c) 2021Tencent. All Rights Reserved
#
########################################################################

from modules.wavetransfer.bddm.galr import GALR

def get_schedule_network(config):
    if config.schedule_net == 'GALR':
        conf_keys = GALR.__init__.__code__.co_varnames
        model_config = {k: v for k, v in vars(config).items() if k in conf_keys}
        return GALR(**model_config)
