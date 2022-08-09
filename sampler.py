#functions to perform extra sampling based on a given conformer
#since this is only for demo purpose, only MD sampling is remade here

#TODO: separate MD and QM, add QM server so that we can run QM for new structures while doing ML-potential MD
#TODO: h5py perform poor when editing, consider using MongoDB
#TODO: switch to multiprocessing after fixing the CUDA problem in multiprocessing

import numpy as np
import h5py, os, sys, re, pathlib, torch, torchani, random, ase.io

from ase import Atoms, units
from ase.md.langevin import Langevin
from pathlib import Path

from utils import *

from pyscf import gto, dft



#MD settings
MD_MAXSTEPS = 100   #DEBUG: larger upper bound
MD_STEPLENGTH = 0.5 * units.fs
CHECK_INTERVAL = 5
#TODO: there is no need to hard code this species order, should be possible to use the dict in the model with extra transformation
SPECIES_ORDER=('H', 'C', 'N', 'O', 'F', 'S', 'Cl')   
QBC_CUTOFF = 0.0002   #DEBUG: change it to a reasonable value, for demo purpose use small value to ensure MD triggered in the first step
#query by committee

TMP='/dev/shm'
DELETE_TMP=True


def convert_char_list(char_list):
    '''
    creating an ASE Atoms instance requires a sequence of chemical symbols as the element type input
    e.g. when creating a H2O molecule, you need sequence "HHO" and corresponding coordinates
    however, out dataset record elements by array of atomic numbers like np.array([1,1,8]) for H2O
    and the helper function numbers2species can only conver it into char array like np.array(['H','H','O'])
    so here is a function to make chemical symbols sequence
    '''
    s = ''
    for c in char_list:
        s += c
    return s

def pyscf_QM(atoms):
    s = ''
    for a, c in zip(atoms.get_chemical_symbols(),atoms.get_positions()):
        s += "   %s       %f         %f        %f;"%(a, c[0],c[1],c[2])
    
    ###DEBUG: this is just a temporary treatment 
    ### switch to well-developed QM packages later
    mol = gto.M(atom = s, basis = '6-31g**')
    mf = dft.RKS(mol)
    mf.xc = 'pbe,pbe'   
    ret = mf.kernel()
    return float(ret)


def MD_sampler(source, model, collector):
    
    for p in source:
        key, species, coordinates = p
        device = list(model.neural_networks[0].parameters())[0].device
        
        species = np.array(species).astype(str)
        atoms = Atoms(convert_char_list(species), positions=coordinates)

        idx = {k: i for i, k in enumerate(SPECIES_ORDER)}
        s = np.array([idx[s] for s in species], dtype='i8')
        s = torch.tensor([s]).to(device)
        
        temp = random.randint(50,800) * units.kB
        calculator = model.ase()
        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, MD_STEPLENGTH, temperature_K=temp, friction=0.02)

        steps = 0
        sampled = False
        while steps <= MD_MAXSTEPS:
            dyn.run(CHECK_INTERVAL)
            c = atoms.get_positions()
            c = torch.tensor([c]).float().to(device)
            _,_,qbc = model.energies_qbcs((s,c))
            if float(qbc) > QBC_CUTOFF:
                sampled = True
                break
            else:
                steps += CHECK_INTERVAL

        if sampled:
            energy = pyscf_QM(atoms)
            
        else:
            energy = np.nan

        #return atoms.get_positions(), energy
        #print((key, species, atoms.get_positions(), energy))
        if not np.isnan(energy):  
            collector.append((key, species, atoms.get_positions().copy(), energy))
        else:
            pass

def dataset_sampler(h5file, model, save_to='ext.h5'):
    
    collector = []
    source = []
    db = h5py.File(h5file,'r')
    for key in list(db.keys())[:1]:
        species = np.array(db[key]['species']).copy()
        #Due to some problem in previous vrsion of h5py
        #Species array in ani1x dataset is not purely str and need a transformation here
        #species = np.array(db[key]['species']).astype(str).copy()
        for c in db[key]['coordinates'][:1]:
            coordinates = np.array(c).copy()
            source.append((key, species, coordinates))
                
        #db.close()   #DEBUG: this should be enabled if you want to write to existing files
        
    MD_sampler(source, model, collector)
    
    if os.path.isfile(save_to):
        db_write = h5py.File(save_to,'a')
    else:
        db_write = h5py.File(save_to,'w')
    
    for info in collector:
        key, species, coordinate, energy = info
        if key in list(db_write.keys()):
            np.append(db_write[key]['coordinates'], coordinate)
        else:
            g = db_write.create_group(key)
            g['species'] = species.astype('S')
            g['coordinates'] = np.array([coordinate])
            g['energies'] = np.array([energy])
    
    db_write.close()
    
    print('extra sampling done, new datapoints saved to '+ str(save_to))
    
    return collector

            
    
    