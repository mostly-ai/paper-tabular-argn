import pandas as pd
import time
from pathlib import Path

from sdv.metadata import Metadata
from sdv.utils import drop_unknown_references
from sdv.multi_table import HMASynthesizer

base_path = Path(__file__).resolve().parent.parent

dataset_dir = base_path / 'data_train'
data = {
        'players': pd.read_parquet(dataset_dir / 'baseball-players-train.parquet'),
        'fielding': pd.read_parquet(dataset_dir / 'baseball-fielding-train.parquet')
    }
metadata = Metadata.detect_from_dataframes(data=data)
metadata.update_column(column_name='playerID', table_name='fielding', sdtype='id')
metadata.relationships = [
        {
            "parent_table_name": "players",
            "parent_primary_key": "playerID",
            "child_table_name": "fielding",
            "child_foreign_key": "playerID"
        }
]
metadata.validate()
clean_data = drop_unknown_references(data, metadata)
synthesizer = HMASynthesizer(metadata,verbose=True)
print('Training...')
t0 = time.time()
synthesizer.fit(clean_data)
tt = time.time() - t0
print('Generation...')
t0 = time.time()
synthetic_data = synthesizer.sample()
gt = time.time() - t0
print(f'Training: {tt:.3f} s; Generation: {gt:.3f} s')

for table_name, table_data in synthetic_data.items():
    table_data.to_parquet(f'SDV_HMA_baseball_{table_name}.parquet')
