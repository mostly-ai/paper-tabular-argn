import pandas as pd
import time
from pathlib import Path

from sdv.metadata import Metadata
from sdv.utils import drop_unknown_references
from sdv.multi_table import HMASynthesizer

dataset_dir = Path(f'data/california/')
data = {
        'household': pd.read_parquet(dataset_dir / 'household'),
        'individual': pd.read_parquet(dataset_dir / 'individual').drop('individual_id',axis=1)
    }
metadata = Metadata.detect_from_dataframes(data=data)
metadata.update_column(column_name='household_id', table_name='individual', sdtype='id')
metadata.update_column(column_name='household_id', table_name='household', sdtype='id')
metadata.relationships = [
        {
            "parent_table_name": "household",
            "parent_primary_key": "household_id",
            "child_table_name": "individual",
            "child_foreign_key": "household_id"
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
    table_data.to_parquet(f'SDV_HMA_california_{table_name}.parquet')
