import time
import shutil
import pandas as pd
from pathlib import Path
import sys
import logging

from mostlyai.engine import split, encode, analyze, train, generate
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="[%(asctime)s] %(levelname)-7s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

base_path = Path(__file__).resolve().parent.parent

ctx_fn = base_path / 'data_train' / 'california-household-train.parquet'
tgt_fn = base_path / 'data_train' / 'california-individual-train.parquet'

ctx = pd.read_parquet(ctx_fn)
tgt = pd.read_parquet(tgt_fn)

ws_dir = Path('california_sequential-ws')
flat_ws_dir = Path('california_flat-ws')

ctx_primary_key = 'household_id'
tgt_foreign_key = 'household_id'
tgt_primary_key = 'individual_id'

# first generate a parent (flat) table "household"
t0 = time.time()
split(
    tgt_data=ctx,
    tgt_primary_key=ctx_primary_key,
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
shutil.copytree(flat_ws_dir / 'SyntheticData', 'MOSTLYAI_california_household')
gt = time.time() - t0
print(f'"household" - Sample: {ctx.shape[0]}; Preprocessing: {pt:.3f} s, Training: {tt:.3f} s; Generation: {gt:.3f} s')


# now generate a child (sequential) table "individual"
t0 = time.time()
split(
    tgt_data=tgt,
    tgt_context_key=tgt_foreign_key,
    tgt_primary_key=tgt_primary_key,
    ctx_data=ctx,
    ctx_primary_key=ctx_primary_key,
    workspace_dir=ws_dir,
)
analyze(workspace_dir=ws_dir)
encode(workspace_dir=ws_dir)
pt = time.time() - t0
t0 = time.time()
train(
    max_training_time=300,
    workspace_dir=ws_dir,
#    differential_privacy={'max_epsilon':10},
)
tt = time.time() - t0
t0 = time.time()
generate(
    ctx_data=pd.read_parquet('MOSTLYAI_california_household'),
    workspace_dir=ws_dir,
)
gt = time.time() - t0
shutil.copytree(ws_dir / 'SyntheticData', 'MOSTLYAI_california_individual')
print(f'"individual" - Sample: {ctx.shape[0]}; Preprocessing: {pt:.3f} s, Training: {tt:.3f} s; Generation: {gt:.3f} s')
