# -*- coding:utf-8 -*-
# Yujay@2023/10/16
import random
from const import *

def get_samples():
    domain_indicators = []
    labels = []
    features = [[] for _ in range(FEATURE_NUM)]
    for i in range(BATCH_SIZE):
        for j in range(FEATURE_NUM):
            features[j].append(random.randint(0, FEATURE_BUCKET_SIZE-1))
        labels.append([1 if random.random()<0.05 else 0])
        domain_indicators.append(random.randint(0, DOMAIN_BUCKET_SIZE-1))

    return {
        "labels": labels,
        "features": features,
        "domain_indicators": domain_indicators
    }