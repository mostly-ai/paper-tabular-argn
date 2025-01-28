import pickle
import pandas as pd
import time
from pathlib import Path
import json

from rctgan import Metadata
from rctgan.relational import RCTGAN

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

base_path = Path(__file__).resolve().parent.parent

wdir = base_path / 'data_train'

datasets = {
    'baseball':{
        'primary_key': 'playerID',
        'foreign_key': 'playerID',
        'tables':{
            'players':{
                'playerID': {
                    'type': 'id',
                    'subtype': 'string'
                }, 
                'birthCountry': {
                    'type': 'categorical'
                }, 
                'birthDate': {
                    'type': 'datetime', 
                    'format': '%Y-%m-%d'
                }, 
                'deathDate': {
                    'type': 'datetime', 
                    'format': '%Y-%m-%d'
                }, 
                'nameFirst': {
                    'type': 'categorical'
                }, 
                'nameLast': {
                    'type': 'categorical'
                }, 
                'weight': {
                    'type': 'numerical',
                    'subtype': 'float'
                }, 
                'height': {
                    'type': 'numerical',
                    'subtype': 'float'
                }, 
                'bats': {
                    'type': 'categorical'
                }, 
                'throws': {
                    'type': 'categorical'
                }
            }, 
            'fielding':{
                'playerID': {
                    'type': 'id',
                    'subtype': 'string'
                }, 
                'yearID': {
                    'type': 'numerical',
                    'subtype': 'integer'
                }, 
                'teamID': {
                    'type': 'categorical'
                }, 
                'lgID': {
                    'type': 'categorical'
                }, 
                'POS': {
                    'type': 'categorical'
                }, 
                'G': {
                    'type': 'numerical',
                    'subtype': 'integer'
                }, 
                'GS': {
                    'type': 'numerical',
                    'subtype': 'float'
                }, 
                'InnOuts': {
                    'type': 'numerical',
                    'subtype': 'float'
                }, 
                'PO': {
                    'type': 'numerical',
                    'subtype': 'integer'
                }, 
                'A': {
                    'type': 'numerical',
                    'subtype': 'integer'
                }, 
                'E': {
                    'type': 'numerical',
                    'subtype': 'float'
                }, 
                'DP': {
                    'type': 'numerical',
                    'subtype': 'integer'
                }
            }
        }
    }
}

metadata = Metadata()
table_names = list(datasets['baseball']['tables'].keys())
primary_key = datasets['baseball']['primary_key']
foreign_key = datasets['baseball']['foreign_key']
df_parent = pd.read_parquet(wdir / 'baseball-players-train.parquet')
df_child = pd.read_parquet(wdir / 'baseball-fielding-train.parquet')

tables = dict(zip(table_names, [df_parent,df_child]))

metadata.add_table(
    name=table_names[0],
    data=tables[table_names[0]],
    primary_key=primary_key,
    fields_metadata = datasets['baseball']['tables'][table_names[0]]
)
metadata.add_table(
    name=table_names[1],
    data=tables[table_names[1]],
    fields_metadata = datasets['baseball']['tables'][table_names[1]]
)

metadata.add_relationship(parent=table_names[0], child=table_names[1], foreign_key=foreign_key)
hyper = {table: {'cuda':True, 'batch_size':32, 'verbose': True, 'pac':32, 'epochs':300} for table in table_names}

t0 = time.time()
model = RCTGAN(metadata, hyper)
model.fit(tables)
pickle.dump(model, open(f'model_rctgan_baseball.p', "wb" ) )
tt = time.time() - t0
t0 = time.time()

synthetic_data = model.sample()
gt = time.time() - t0
print(f'Training: {tt:.3f} s; Generation: {gt:.3f} s')
for table_name,table_data in synthetic_data.items():
    fn = f'RCTGAN_baseball_{table_name}.parquet'
    table_data.to_parquet(fn)
