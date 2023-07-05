import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.core.sites import PeriodicSite

from .eos import EOS


def strucure_to_dict(structure, precision=3):
    res = {}
    for site in structure:
        res[tuple(np.round(site.frac_coords, precision))] = site
    return res


def add_was(structure, unit_cell, supercell_size):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)

    sites = []

    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)

    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            continue
        else:
            cur_site = structure_dict[coords]
            cur_site.properties['was'] = reference_site.specie.Z
            sites.append(
                PeriodicSite(
                    species=cur_site.species,
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                    properties=cur_site.properties,
                )
            )

    structure_with_was = Structure.from_sites(sites)
    return structure_with_was


def get_sparse_defect(structure, unit_cell, supercell_size):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)

    defects = []

    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)

    for coords, reference_site in reference_structure_dict.items():
        # Vacancy
        if coords not in structure_dict:
            defects.append(
                PeriodicSite(
                    species=DummySpecies(),
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                    properties={},
                ))
        # Substitution
        elif structure_dict[coords].specie != reference_site.specie:
            defects.append(structure_dict[coords])

    res = Structure.from_sites(defects)
    return res


def add_eos(structure, unit_cell, supercell_size):
    structure = structure.copy()
    unit_cell = EOS().get_augmented_struct(unit_cell)
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)

    sites = []

    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)

    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            continue
        else:
            cur_site = structure_dict[coords]
            cur_site.properties.update(reference_site.properties)
            sites.append(
                PeriodicSite(
                    species=cur_site.species,
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                    properties=cur_site.properties,
                )
            )

    structure_with_eos = Structure.from_sites(sites)
    return structure_with_eos


def add_state(structure, unit_cell):
    reference_species = set(unit_cell.species)
    structure = structure.copy()
    structure.state = [sorted([element.Z for element in reference_species])]
    return structure


def add_unit_cell_properties(structure, unit_cell, supercell_size):
    structure = structure.copy()
    reference_supercell = unit_cell.copy()
    reference_supercell.make_supercell(supercell_size)

    sites = []

    structure_dict = strucure_to_dict(structure)
    reference_structure_dict = strucure_to_dict(reference_supercell)

    for coords, reference_site in reference_structure_dict.items():
        if coords not in structure_dict:
            continue
        else:
            cur_site = structure_dict[coords]
            cur_site.properties.update(reference_site.properties)
            sites.append(
                PeriodicSite(
                    species=cur_site.species,
                    coords=coords,
                    coords_are_cartesian=False,
                    lattice=structure.lattice,
                    properties=cur_site.properties,
                )
            )

    res_structure = Structure.from_sites(sites)
    return res_structure


def convert_to_sparse_representation(
        structure,
        unit_cell,
        supercell_size,
        skip_eos=False,
        skip_was=False,
        skip_state=False,
        copy_unit_cell_properties=False,
):
    structure = structure.copy()
    unit_cell = unit_cell.copy()
    if not skip_eos:
        structure = add_eos(structure, unit_cell, supercell_size)

    structure = get_sparse_defect(structure, unit_cell, supercell_size)

    if not skip_was:
        structure = add_was(structure, unit_cell, supercell_size)

    if copy_unit_cell_properties:
        structure = add_unit_cell_properties(structure, unit_cell, supercell_size)

    if not skip_state:
        structure = add_state(structure, unit_cell)

    return structure
