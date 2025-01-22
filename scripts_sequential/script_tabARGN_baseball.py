import time
import shutil
import pandas as pd
from pathlib import Path
import sys
import logging

from mostlyai.engine import split, encode, analyze, train, generate
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="[%(asctime)s] %(levelname)-7s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

ctx_fn = 'data/baseball/train/players/players.parquet'
tgt_fn = 'data/baseball/train/fielding/fielding.parquet'

ctx = pd.read_parquet(ctx_fn)
tgt = pd.read_parquet(tgt_fn)

ws_dir = Path('sequential-ws')
flat_ws_dir = Path('flat-ws')

primary_key = 'playerID'
foreign_key = 'playerID'

# first generate a parent (flat) table "players"
t0 = time.time()
split(
    tgt_data=ctx,
    tgt_primary_key=primary_key,
    workspace_dir=flat_ws_dir,
)
analyze(workspace_dir=flat_ws_dir)
encode(workspace_dir=flat_ws_dir)
pt = time.time() - t0

t0 = time.time()
train(
    max_training_time=300,
    workspace_dir=flat_ws_dir
)
tt = time.time() - t0
generate(
    sample_size=ctx.shape[0], # generate as many subjects as in the original data 
    workspace_dir=flat_ws_dir,
)
gt = time.time() - t0
shutil.copytree(flat_ws_dir / "SyntheticData", 'MOSTLYAI_baseball_players')
print(f'"Players" - Sample: {ctx.shape[0]}; Preprocessing: {pt:.3f} s, Training: {tt:.3f} s; Generation: {gt:.3f} s')


# now generate a child (sequential) table "fielding"
t0 = time.time()
split(
    tgt_data=tgt,
    tgt_context_key=foreign_key,
    ctx_data=ctx,
    ctx_primary_key=primary_key,
    workspace_dir=ws_dir,
)
analyze(workspace_dir=ws_dir)
encode(workspace_dir=ws_dir)
pt = time.time() - t0
t0 = time.time()
train(
    max_training_time=300,
    workspace_dir=ws_dir
)
tt = time.time() - t0
t0 = time.time()
generate(
    ctx_data=pd.read_parquet('MOSTLYAI_baseball_players'),
    workspace_dir=ws_dir,
)
gt = time.time() - t0
shutil.copytree(ws_dir / "SyntheticData", 'MOSTLYAI_baseball_fielding')
print(f'"Fielding" - Sample: {ctx.shape[0]}; Preprocessing: {pt:.3f} s, Training: {tt:.3f} s; Generation: {gt:.3f} s')
