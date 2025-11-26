from __future__ import absolute_import, division, print_function

import os
import warnings
import copy

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from scipy.spatial.distance import pdist

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool

from tqdm import tqdm

from ..config import MODEL_CONFIG
from ..utils import logger
from ..weights import WEIGHT_DIR, weight_download
from .dictionary import Dictionary

from .conformer import coords2unimol, inner_coords, inner_smi2coords


class MultiConformerGen(object):
    def __init__(self, **params):
        self._init_features(**params)

    def _init_features(self, **params):
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.remove_hs = params.get('remove_hs', False)
        self.mmff_optimize = params.get('mmff_optimize', True)

        self.method = params.get('method', 'rdkit_etkdg')

        self.num_conformers = params.get('num_conformers', 200)
        self.energy_window = params.get('energy_window', 2.0)

        if self.data_type == 'molecule':
            name = "no_h" if self.remove_hs else "all_h"
            name = self.data_type + '_' + name
            self.dict_name = MODEL_CONFIG['dict'][name]
        else:
            self.dict_name = MODEL_CONFIG['dict'][self.data_type]
        
        if not os.path.exists(os.path.join(WEIGHT_DIR, self.dict_name)):
            weight_download(self.dict_name, WEIGHT_DIR)
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)

        if os.name == 'posix':
            self.multi_process = params.get('multi_process', True)
        else:
            self.multi_process = params.get('multi_process', False)
            if self.multi_process:
                logger.warning(
                    'Please use "if __name__ == "__main__":" to wrap the main function when using multi_process on Windows.'
                )

    def single_process(self, smiles):
        logger.debug(f'Generating multiple conformers for {smiles} using multi-conformer method')

        atoms, coordinates, mol, selection_info = multi_conformer_generation(
            smiles, 
            seed=self.seed, 
            num_conformers=self.num_conformers,
            energy_window=self.energy_window,
            remove_hs=self.remove_hs,
            mmff_optimize=self.mmff_optimize
        )

        if selection_info.get('reason') == 'fallback_single_conformer' or selection_info.get('reason') == 'total_failure':
            logger.warning(f'Fallback used for {smiles}: {selection_info.get("reason", "unknown")}')
        else:
            logger.debug(f'Selected conformer for {smiles}: energy={selection_info.get("energy", "N/A"):.2f}')
        
        feat = coords2unimol(
            atoms,
            coordinates,
            self.dictionary,
            self.max_atoms,
            remove_hs=self.remove_hs,
        )
        return feat, mol

    def transform_raw(self, atoms_list, coordinates_list):
        inputs = []
        for atoms, coordinates in zip(atoms_list, coordinates_list):
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def transform_mols(self, mols_list):
        inputs = []
        for mol in mols_list:
            atoms = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            inputs.append(
                coords2unimol(
                    atoms,
                    coordinates,
                    self.dictionary,
                    self.max_atoms,
                    remove_hs=self.remove_hs,
                )
            )
        return inputs

    def transform(self, smiles_list):
        logger.info('Start generating multiple conformers...')
        if self.multi_process:
            pool = Pool(processes=min(8, os.cpu_count()))
            results = [
                item for item in tqdm(pool.imap(self.single_process, smiles_list))
            ]
            pool.close()
        else:
            results = [self.single_process(smiles) for smiles in tqdm(smiles_list)]

        inputs, mols = zip(*results)
        inputs = list(inputs)
        mols = list(mols)

        failed_conf = [(item['src_coord'] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf)) * 100
            )
        )
        failed_conf_indices = [
            index for index, value in enumerate(failed_conf) if value
        ]
        if len(failed_conf_indices) > 0:
            logger.info('Failed conformers indices: {}'.format(failed_conf_indices))
            logger.debug(
                'Failed conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_indices]
                )
            )

        failed_conf_3d = [(item['src_coord'][:, 2] == 0.0).all() for item in inputs]
        logger.info(
            'Succeeded in generating 3d conformers for {:.2f}% of molecules.'.format(
                (1 - np.mean(failed_conf_3d)) * 100
            )
        )
        failed_conf_3d_indices = [
            index for index, value in enumerate(failed_conf_3d) if value
        ]
        if len(failed_conf_3d_indices) > 0:
            logger.info(
                'Failed 3d conformers indices: {}'.format(failed_conf_3d_indices)
            )
            logger.debug(
                'Failed 3d conformers SMILES: {}'.format(
                    [smiles_list[index] for index in failed_conf_3d_indices]
                )
            )
        return inputs, mols

def multi_conformer_generation(smi, seed=42, num_conformers=200, energy_window=2.0,
                             remove_hs=False, return_mol=False, mmff_optimize=True,
                             mmff_max_iters=200):

    try:
        logger.debug(f"Processing SMILES: {smi} with multi-conformer ETKDG method")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError('Invalid SMILES: {}'.format(smi))
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        assert len(atoms) > 0, 'No atoms in molecule: {}'.format(smi)
        if len(atoms) <= 10:
            actual_num_conformers = min(num_conformers, 50)
        elif len(atoms) <= 20:
            actual_num_conformers = min(num_conformers, 100)
        else:
            actual_num_conformers = num_conformers
        logger.debug(f"Generating {actual_num_conformers} conformers for molecule with {len(atoms)} atoms (SMILES: {smi})")
        conformers_data = generate_multiple_conformers(
            mol, seed, actual_num_conformers, mmff_optimize=mmff_optimize,
            mmff_max_iters=mmff_max_iters
        )
        if not conformers_data:
            logger.warning(f"Failed to generate any conformers for {smi}, falling back to single conformer")
            return fallback_single_conformer(smi, seed, remove_hs, return_mol)
        logger.debug(f"Successfully generated {len(conformers_data)} conformers for {smi}")
        best_conformer, selection_info = select_best_conformer(
            conformers_data, energy_window
        )
        coordinates = best_conformer['coordinates']
        mol_result = best_conformer['mol']
        if return_mol:
            return mol_result, selection_info
        if remove_hs:
            idx = [i for i, atom in enumerate(atoms) if atom != 'H']
            atoms_no_h = [atom for atom in atoms if atom != 'H']
            coordinates_no_h = coordinates[idx]
            assert len(atoms_no_h) == len(
                coordinates_no_h
            ), "coordinates shape is not align with {}".format(smi)
            return atoms_no_h, coordinates_no_h, mol_result, selection_info
        else:
            return atoms, coordinates, mol_result, selection_info
    except Exception as e:
        logger.error(f"Failed to generate multi-conformers for {smi}: {str(e)}, using fallback")
        return fallback_single_conformer(smi, seed, remove_hs, return_mol)

def generate_multiple_conformers(mol, seed, num_conformers, mmff_optimize=True, mmff_max_iters=200):
    conformers_data = []
    try:
        if mol is None:
            logger.error("Input molecule is None")
            return []
        if mol.GetNumAtoms() == 0:
            logger.error("Molecule has no atoms")
            return []
        logger.debug(f"Generating {num_conformers} conformers using ETKDGv3")
        mol_with_hs = AllChem.AddHs(mol)
        all_conformers = []
        max_attempts = 5
        attempts = 0
        while len(all_conformers) < num_conformers and attempts < max_attempts:
            attempts += 1
            temp_mol = Chem.Mol(mol_with_hs)
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = seed + attempts
                params.numThreads = 0
                params.enforceChirality = True
                params.useRandomCoords = True
                needed = num_conformers - len(all_conformers)
                if needed <= 0:
                    break
                new_conf_ids = AllChem.EmbedMultipleConfs(
                    temp_mol,
                    numConfs=needed,
                    params=params
                )
                if new_conf_ids:
                    for conf_id in new_conf_ids:
                        conf = temp_mol.GetConformer(conf_id)
                        new_conf_id = mol_with_hs.AddConformer(conf, assignId=True)
                        all_conformers.append(new_conf_id)
                    if len(all_conformers) >= num_conformers:
                        break
            except Exception as e:
                logger.debug(f"ETKDGv3 attempt {attempts} failed: {str(e)}")
        logger.info(f"Successfully embedded {len(all_conformers)} conformers")
        if len(all_conformers) == 0:
            logger.warning("ETKDGv3 completely failed, trying fallback method")
            try:
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol_with_hs,
                    numConfs=num_conformers,
                    randomSeed=seed,
                    clearConfs=True
                )
                logger.debug(f"Fallback method generated {len(conf_ids)} conformers")
                all_conformers = list(conf_ids)
            except Exception as e2:
                logger.error(f"All conformer generation methods failed: {str(e2)}")
                return []
        if len(all_conformers) == 0:
            logger.warning("No conformers generated")
            return []
        if mmff_optimize:
            try:
                AllChem.MMFFOptimizeMoleculeConfs(
                    mol_with_hs, maxIters=int(mmff_max_iters)
                )
            except Exception as e:
                print(f"{str(e)}")
        for i, conf_id in enumerate(all_conformers):
            try:
                single_mol = Chem.Mol(mol_with_hs)
                single_mol.RemoveAllConformers()
                single_mol.AddConformer(mol_with_hs.GetConformer(conf_id), assignId=True)
                if mmff_optimize:
                    try:
                        mp_opt = AllChem.MMFFGetMoleculeProperties(single_mol)
                        if mp_opt:
                            AllChem.MMFFOptimizeMolecule(single_mol, maxIters=int(mmff_max_iters))
                        else:
                            AllChem.UFFOptimizeMolecule(single_mol, maxIters=int(mmff_max_iters))
                    except Exception as _:
                        pass
                energy = None
                converged = True
                try:
                    mp = AllChem.MMFFGetMoleculeProperties(single_mol)
                    if mp:
                        ff = AllChem.MMFFGetMoleculeForceField(single_mol, mp)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                        else:
                            ff = AllChem.UFFGetMoleculeForceField(single_mol)
                            if ff is not None:
                                energy = ff.CalcEnergy()
                            else:
                                logger.debug(f"UFF force field setup also failed for conformer {i}")
                                energy = 0.0
                    else:
                        ff = AllChem.UFFGetMoleculeForceField(single_mol)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                        else:
                            energy = 0.0
                except Exception as e:
                    logger.debug(f"Energy calculation failed for conformer {i}: {str(e)}")
                    energy = 0.0
                coordinates = single_mol.GetConformer().GetPositions().astype(np.float32)
                conformers_data.append({
                    'mol': single_mol,
                    'coordinates': coordinates,
                    'conf_id': i,
                    'energy': energy,
                    'converged': converged
                })
            except Exception as e:
                logger.debug(f"Failed to process conformer {conf_id}: {str(e)}")
                continue
        logger.debug(f"Successfully processed {len(conformers_data)} conformers")
        return conformers_data
    except Exception as e:
        logger.error(f"Conformer generation failed: {str(e)}")
        return []


def select_best_conformer(conformers_data, energy_window):
    if not conformers_data:
        raise ValueError("No valid conformers to select from")
    valid_energy_conformers = [conf for conf in conformers_data if conf['energy'] is not None]
    if not valid_energy_conformers:
        logger.warning("No conformers with valid energy, selecting first conformer")
        best_conformer = conformers_data[0]
        selection_info = {
            'reason': 'no_valid_energy',
            'energy': best_conformer['energy'],
            'total_generated': len(conformers_data),
            'energy_filtered': 0,
            'final_candidates': 1
        }
        return best_conformer, selection_info
    energies = [conf['energy'] for conf in valid_energy_conformers]
    min_energy = min(energies)
    max_energy = max(energies)
    if max_energy - min_energy < 1e-6:
        filtered_conformers = valid_energy_conformers
        logger.debug("All conformers have similar energy, skipping energy filtering")
    else:
        filtered_conformers = [
            conf for conf in valid_energy_conformers 
            if conf['energy'] - min_energy <= energy_window
        ]
    logger.debug(f"After energy filtering: {len(filtered_conformers)} conformers within {energy_window} kcal/mol")
    best_conformer = select_for_energetic_materials(filtered_conformers)
    selection_info = {
        'reason': 'multi_criteria_selection',
        'energy': best_conformer['energy'],
        'relative_energy': best_conformer['energy'] - min_energy,
        'total_generated': len(conformers_data),
        'energy_filtered': len(filtered_conformers),
        'final_candidates': len(filtered_conformers),
        'molecular_volume': calculate_molecular_volume(best_conformer['mol']),
        'compactness_score': calculate_compactness_score(best_conformer['coordinates'])
    }
    return best_conformer, selection_info

def select_for_energetic_materials(conformers_data):
    if len(conformers_data) == 1:
        return conformers_data[0]
    energies = [conf['energy'] for conf in conformers_data if conf['energy'] is not None]
    energy_none_count = len([conf for conf in conformers_data if conf['energy'] is None])
    if not energies:
        logger.debug("No energy information available, selecting converged conformer")
        converged_conformers = [conf for conf in conformers_data if conf.get('converged', False)]
        if converged_conformers:
            selected_conformer = converged_conformers[0]
            return selected_conformer
        else:
            selected_conformer = conformers_data[0]
            return selected_conformer
    min_energy = min(energies)
    max_energy = max(energies)
    for conf in conformers_data:
        mol = conf['mol']
        coordinates = conf['coordinates']
        if conf['energy'] is not None and max_energy > min_energy:
            conf['energy_score'] = (max_energy - conf['energy']) / (max_energy - min_energy)
        else:
            conf['energy_score'] = 1.0
        conf['compactness_score'] = calculate_compactness_score(coordinates)

        conf['total_score'] = (
            0.50 * conf['energy_score'] +
            0.50 * conf['compactness_score']
        )
    best_conformer = max(conformers_data, key=lambda x: x['total_score'])
    best_index = conformers_data.index(best_conformer)
    molecular_volume = calculate_molecular_volume(best_conformer['mol'])
    logger.debug(f"Selected conformer scores - Energy: {best_conformer['energy_score']:.3f}, "
                f"Compactness: {best_conformer['compactness_score']:.3f}, "
                f"Total: {best_conformer['total_score']:.3f}, "
                f"Actual Energy: {best_conformer['energy']:.2f} kcal/mol")
    return best_conformer

def calculate_compactness_score(coordinates):
    try:
        center = np.mean(coordinates, axis=0)
        distances = np.linalg.norm(coordinates - center, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances ** 2))
        max_distance = np.max(pdist(coordinates))
        if max_distance > 0:
            compactness = radius_of_gyration / max_distance
            score = 1.0 / (1.0 + compactness)
        else:
            score = 1.0
        return min(1.0, max(0.0, score))
    except:
        return 0.5

def calculate_molecular_volume(mol):
    try:
        return Descriptors.MolVolume(mol)
    except:
        return 0.0

def fallback_single_conformer(smi, seed, remove_hs, return_mol):
    try:
        from .conformer import inner_smi2coords
        if return_mol:
            mol = inner_smi2coords(smi, seed=seed, remove_hs=remove_hs, return_mol=True)
            selection_info = {'reason': 'fallback_single_conformer', 'method': 'ETKDG_single'}
            return mol, selection_info
        else:
            atoms, coordinates, mol = inner_smi2coords(smi, seed=seed, remove_hs=remove_hs)
            selection_info = {'reason': 'fallback_single_conformer', 'method': 'ETKDG_single'}
            return atoms, coordinates, mol, selection_info
    except Exception as e:
        logger.error(f"Fallback method also failed for {smi}: {str(e)}")
        mol = Chem.MolFromSmiles('C')
        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = np.zeros((len(atoms), 3), dtype=np.float32)
        selection_info = {'reason': 'total_failure', 'method': 'zero_coordinates'}
        if return_mol:
            return mol, selection_info
        else:
            return atoms, coordinates, mol, selection_info 