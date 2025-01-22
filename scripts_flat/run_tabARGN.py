import pandas as pd
from pathlib import Path
import logging
import sys
import time
import csv
from mostlyai.engine import split, encode, analyze, train, generate

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="[%(asctime)s] %(levelname)-7s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')


DATASETS = ['adult', 'acs-income']
timing_engine = '/home/ubuntu/engine_paper/tabsyn_repo/tabsyn/engine/engine_timing.csv'
methods = []
datasets = []
stages = []
times = []
header = ['method','dataset','stage','time']
for dataset in DATASETS:

    ws = Path(f'/home/ubuntu/engine_paper/tabsyn_repo/tabsyn/engine/mostly_{dataset}')

    tgt_fn = f'/home/ubuntu/engine_paper/tabsyn_repo/tabsyn/data/{dataset}/train.csv'
    tgt_df = pd.read_csv(tgt_fn)
    print(f'training:{dataset}')
    t00 = time.time()
    split(
        tgt_data=tgt_df,
        workspace_dir=ws,
    )
    analyze(workspace_dir=ws)
    encode(workspace_dir=ws)
    pt = time.time() - t00
    t0 = time.time()
    train(
        max_training_time=300.0, # 5 hours
        workspace_dir=ws,
    )
    t1 = time.time()
    tt = t1 - t0
    ptt = t1 - t00
    t0 = time.time()
    generate(
        workspace_dir=ws,
    )
    gt = time.time() - t0
    
    methods = methods + ['engine']*4
    datasets = datasets + [dataset]*4
    stages = stages + ['preprocessing','training','full_training','generation']
    times = times + [pt,tt,ptt,gt]
timings = pd.DataFrame({'method':methods,'dataset':datasets,'stage':stages,'time':times})
timings.to_csv(timing_engine)


