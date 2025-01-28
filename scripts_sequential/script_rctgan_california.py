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
    'california':{
        'primary_key': 'household_id',
        'foreign_key': 'household_id',
        'tables':{
            'household':{
                'household_id': {
                    'type': 'id',
                    'subtype': 'string'
                }, 
                'FARM': {
                    'type': 'numerical',
                    'subtype': 'integer',
                }, 
                'OWNERSHP': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },
                'ACREHOUS': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'TAXINCL': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },   
                'PROPINSR': {
                    'type': 'numerical',
                    'subtype': 'float',
                },
                'COSTELEC': {
                    'type': 'numerical',
                    'subtype': 'float',
                },    
                'VALUEH': {
                    'type': 'numerical',
                    'subtype': 'float',
                },   
                'ROOMS': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },   
                'PLUMBING': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },   
                'PUMA': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },        
            }, 
            'individual':{
                'household_id': {
                    'type': 'id',
                    'subtype': 'string'
                }, 
                'RELATE': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },    
                'SEX': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'AGE': {
                    'type': 'numerical',
                    'subtype': 'float',
                },  
                'MARST': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'RACE': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'CITIZEN': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'SPEAKENG': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'SCHOOL': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'EDUC': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'GRADEATT': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'SCHLTYPE': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'EMPSTAT': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'CLASSWKR': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
                'INCTOT': {
                    'type': 'numerical',
                    'subtype': 'float',
                },  
                'DISABWRK': {
                    'type': 'numerical',
                    'subtype': 'integer',
                },  
            }
        }
    }
}

metadata = Metadata()
table_names = list(datasets['california']['tables'].keys())
primary_key = datasets['california']['primary_key']
foreign_key = datasets['california']['foreign_key']
df_parent = pd.read_parquet(wdir / 'california-household-train.parquet')
df_child = pd.read_parquet(wdir / 'california-individual-train.parquet')

tables = dict(zip(table_names, [df_parent,df_child]))

metadata.add_table(
    name=table_names[0],
    data=tables[table_names[0]],
    primary_key=primary_key,
    fields_metadata = datasets['california']['tables'][table_names[0]]
)
metadata.add_table(
    name=table_names[1],
    data=tables[table_names[1]],
    fields_metadata = datasets['california']['tables'][table_names[1]]
)

metadata.add_relationship(parent=table_names[0], child=table_names[1], foreign_key=foreign_key)
hyper = {table: {'cuda':True, 'batch_size':128, 'verbose': True, 'pac':128, 'epochs':300} for table in table_names}

t0 = time.time()
model = RCTGAN(metadata, hyper)
model.fit(tables)
pickle.dump(model, open(f'model_rctgan_california.p', "wb" ) )
tt = time.time() - t0
t0 = time.time()

synthetic_data = model.sample()
gt = time.time() - t0
print(f'Training: {tt:.3f} s; Generation: {gt:.3f} s')
for table_name,table_data in synthetic_data.items():
    fn = f'RCTGAN_california_{table_name}.parquet'
    table_data.to_parquet(fn)
