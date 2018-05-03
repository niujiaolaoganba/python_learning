#!/usr/bin/env python
# -*- coding:utf-8 -*-

import ConfigParser
import os

class ConfigInfo(object):
    def __init__(self,cfg_path):
        self.read_cfg=ConfigParser.ConfigParser()
        cfg_path=os.path.abspath(os.path.dirname(__file__))+'/../../config/%s'%cfg_path
        self.read_cfg.read(cfg_path)

    def getSQL(self,sql_part,sql_name):
        return self.read_cfg.get(sql_part,sql_name)