import pandas as pd
import time
from pathlib import Path

from sdv.metadata import Metadata
from sdv.utils import get_random_sequence_subset
from sdv.sequential import PARSynthesizer


dataset_dir = Path(f'data/baseball')
data = pd.read_parquet(dataset_dir / 'data.parquet')
metadata = Metadata.load_from_json(filepath=dataset_dir / 'metadata.json')
context_cols = ['birthCountry','birthDate','deathDate','nameFirst','nameLast','weight','height','bats','throws']

target_cols = ['yearID','teamID','lgID','POS','G','GS','InnOuts','PO','A','E','DP']

t0 = time.time()
synthesizer = PARSynthesizer(metadata,
        context_columns=context_cols,
#                                     segment_size=5,
        cuda=True,
        verbose=True
)
synthesizer.fit(data)
tt = time.time() - t0
t0 = time.time()
synthetic_data = synthesizer.sample(num_sequences=data['playerID'].nunique())
gt = time.time() - t0
print(f'Training: {tt:.3f} s; Generation: {gt:.3f} s')

synthetic_data[['playerID'] + context_columns].groupby('playerID').first().reset_index().to_parquet(wdir / 'SDV_baseball_players.parquet')
synthetic_data[['playerID'] + target_columns].reset_index(drop=True).to_parquet(wdir / 'SDV_baseball_fielding.parquet')
