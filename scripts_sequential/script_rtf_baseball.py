import pandas as pd
import time
import os
from pathlib import Path

from realtabformer import REaLTabFormer

base_path = Path(__file__).resolve().parent.parent

dataset_dir = base_path / 'data_train'
parent_df = pd.read_parquet(dataset_dir / 'baseball-players-train.parquet')
child_df = pd.read_parquet(dataset_dir / 'baseball-fielding-train.parquet')
join_on = "playerID"

t0 = time.time()

parent_model = REaLTabFormer(
    model_type="tabular", 
    train_size=0.8, 
    batch_size=8, 
)
parent_model.fit(parent_df.drop(join_on, axis=1), device='cuda')

pdir = Path("rtf_parent/")
parent_model.save(pdir, allow_overwrite=True)

parent_model_path = sorted([
    p for p in pdir.glob("id*") if p.is_dir()],
    key=os.path.getmtime)[-1]

child_model = REaLTabFormer(
    model_type="relational",
    parent_realtabformer_path=parent_model_path,
    batch_size=8,
    train_size=0.8
)
child_model.fit(
    df=child_df,
    in_df=parent_df,
    join_on=join_on, 
    device='cuda'
)
tt = time.time() - t0

t0 = time.time()
# Generate parent samples.
parent_samples = parent_model.sample(len(parent_df))
# Create the unique ids based on the index.
parent_samples.index.name = join_on
parent_samples = parent_samples.reset_index()

# Generate the relational observations.
child_samples = child_model.sample(
    input_unique_ids=parent_samples[join_on],
    input_df=parent_samples.drop(join_on, axis=1),
    gen_batch=64
)

gt = time.time() - t0
print(f'Training: {tt:.3f} s; Generation: {gt:.3f} s')
parent_samples.to_parquet(f'RTF_baseball_players.parquet')
child_samples[join_on]=child_samples.index
child_samples.reset_index(drop=True).to_parquet(f'RTF_baseball_fielding.parquet')
