import numpy as np
import pandas as pd
from pymatgen.io.cif import CifParser
from tqdm import tqdm
from joblib import Parallel, delayed

from dense2sparse import convert_to_sparse_representation
from struct2graph import FlattenGaussianDistanceConverter, AtomFeaturesExtractor, SimpleCrystalConverter

df = pd.read_pickle('examples/pilot/data.pickle.gz')
sizes = pd.read_csv('examples/descriptors.csv', index_col='_id')
structures = pd.Series(data=df['initial_structure'].values, index=df['descriptor_id'].values, name='structures')
unit_cell_sizes = sizes['cell']

res = pd.merge(structures, unit_cell_sizes, left_index=True, right_index=True)
unit_cell = CifParser("examples/MoS2.cif").get_structures(primitive=False)[0]

res = res.values.tolist()
res = [[p[0], eval(p[1])] for p in res]
dataset = Parallel(n_jobs=-1)(
            delayed(convert_to_sparse_representation)(p[0], unit_cell, p[1], True) for p in tqdm(res))


bond_converter = FlattenGaussianDistanceConverter(
    centers=np.linspace(0, 32, 10)
)
atom_converter = AtomFeaturesExtractor('werespecies')
converter = SimpleCrystalConverter(
    bond_converter=bond_converter,
    atom_converter=atom_converter,
    cutoff=10,
    add_z_bond_coord=True,
    add_eos_features=False,
)

train_structures = Parallel(n_jobs=-1)(
            delayed(converter.convert)(s) for s in tqdm(dataset))
