# MEGNetSparse

### Installation

```
pip install MEGNetSparse
```

1) You must first install the torch and torch-geometric
2) The notebook provided in the examples will only work 
with pymatgen=2023.1.30, so you may need to reinstall it.

### Usage

The library provides the ability to use a function 
convert_to_sparse_representation and a class MEGNetTrainer

```
convert_to_sparse_representation(
    structure,
    unit_cell,
    supercell_size,
    skip_eos=True,
    skip_was=False,
    skip_state=False,
    copy_unit_cell_properties=False
)
```

- structure : Structure - the structre to convert to
sparse representation
- unit_cell : Structure - unit cell of base material
- supercell_size : List[int] - list with three integers to copy 
a cell along three coordinates
- skip_eos : bool - if True will not add eos to properties and will speed up 
computations
- skip_was: bool - if True will not add was to properties
- skip_state : bool - if True will not add global state
- copy_unit_cell_properties: bool - if True will also copy unit cell properties
in case of name collisions structure properties will be overwritten 

return : sparse representation of structure

```
MEGNetTrainer(
    config,
    device,
)
```

- config : dict - template config can be found in examples notebook
- device : str - device in torch format

```
MEGNetTrainer.prepare_data(
    self,
    train_data,
    train_targets,
    test_data,
    test_targets,
    target_name,
):
```

- train_data : List[Structure] - list of structures in 
sparse or dense representation
- train_targets : List[float32] - list of targets
- test_data : List[Structure] - list of structures in 
sparse or dense representation
- test_targets : List[float32] - list of targets
- target_name : str - target name

```
MEGNetTrainer.train_one_epoch(self)
```

return : mae on train data, mse on train data

```
MEGNetTrainer.evaluate_on_test(
    self, 
    return_predictions=False
)
```

return : if return_predictions=True, mae on test data, predictions else
 only mae on test data

```
MEGNetTrainer.predict_structures(
    self, 
    structures_list
)
```

- structures_list : List[Structure] - list of structures in 
sparse or dense representation

return : predictions for structures

```
MEGNetTrainer.save(self, path)
```

- path : str - where to store model data

```
MEGNetTrainer.load(self, path)
```

- path : str - where to load model data from
