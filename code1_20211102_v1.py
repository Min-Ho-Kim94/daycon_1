# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: cook
#     language: python
#     name: cook
# ---

import numpy as np
import pandas as pd

train = pd.read_csv('./input/235713_신용카드 사용자 연체 예측 AI 경진대회_data/open/train.csv')
test = pd.read_csv('./input/235713_신용카드 사용자 연체 예측 AI 경진대회_data/open/test.csv')
sub = pd.read_csv('./input/235713_신용카드 사용자 연체 예측 AI 경진대회_data/open/sample_submission.csv')

`train.head(5)
















































