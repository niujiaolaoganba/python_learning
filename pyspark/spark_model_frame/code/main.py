#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from build_model.model_frame import CreateModel

if __name__=='__main__':
    CreateModel().buildModel(save_pipe_path='/user/chenchen/spark_lr')