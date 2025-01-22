import pandas as pd
import time
from pathlib import Path

from sdv.metadata import Metadata
from sdv.utils import get_random_sequence_subset
from sdv.sequential import PARSynthesizer


dataset_dir = Path(f'data/california')
data = pd.read_parquet(dataset_dir / 'data.parquet')
metadata = Metadata.load_from_json(filepath=dataset_dir / 'metadata.json')
context_cols = ['FARM',
    'OWNERSHP',
    'ACREHOUS',
    'TAXINCL',
    'PROPINSR',
    'COSTELEC',
    'VALUEH',
    'ROOMS',
    'PLUMBING',
    'PUMA'
]
target_cols = ['RELATE',
    'SEX',
    'AGE',
    'MARST',
    'RACE',
    'CITIZEN',
    'SPEAKENG',
    'SCHOOL',
    'EDUC',
    'GRADEATT',
    'SCHLTYPE',
    'EMPSTAT',
    'CLASSWKR',
    'INCTOT',
    'DISABWRK',
]

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
synthetic_data = synthesizer.sample(num_sequences=data['household_id'].nunique())
gt = time.time() - t0
print(f'Training: {tt:.3f} s; Generation: {gt:.3f} s')

synthetic_data[['household_id'] + context_columns].groupby('household_id').first().reset_index().to_parquet(wdir / 'SDV_california_household.parquet')
synthetic_data[['household_id'] + target_columns].reset_index(drop=True).to_parquet(wdir / 'SDV_california_individual.parquet')
