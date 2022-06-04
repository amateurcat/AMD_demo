#functions to perform extra sampling based on a given conformer
#since this is only for demo purpose, only MD sampling is remade here

#TODO: separate MD and QM, add QM server so that we can run QM for new structures while doing ML-potential MD
#TODO: h5py perform poor when editing, consider using MongoDB
#TODO: switch to multiprocessing after fixing the CUDA problem in multiprocessing

import numpy as np
import h5py, os, sys, re, pathlib, torch, torchani, random, subprocess
import threading, queue

from ase import Atoms, units
from ase.md.langevin import Langevin
from pathlib import Path

from utils import *

from multiprocessing import Pool, Manager
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass



#MD settings
MD_MAXSTEPS = 100   #DEBUG: larger upper bound
MD_STEPLENGTH = 0.5 * units.fs
CHECK_INTERVAL = 5
#TODO: there is no need to hard code this species order, should be possible to use the dict in the model with extra transformation
SPECIES_ORDER=('H', 'C', 'N', 'O', 'F', 'S', 'Cl')   
QBC_CUTOFF = 0.0002   #DEBUG: change it to a reasonable value, for demo purpose use small value to ensure MD triggered in the first step

TMP='/dev/shm'
DELETE_TMP=True

ORCA_PATH = Path('/storage/users/roman/Apps/orca_5_0_1_linux_x86-64_shared_openmpi411')


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

def write_smear_input(atoms, coords, output_name, dft='PBE', etemp=0, charge=0, spin=1):   #DEBUG: change it back to WB97X, use fastest PBE for test purpose
    ####################################
    #The energy output is in Hartree!!!#
    ####################################
    #smear DFT input generator that also accept open shell cases
    #please note that !TRAH can only be used in ORCA5
    method = 'RKS'
    if spin >1:
        method='UKS'
        
    trah = '! TRAH' if etemp==0 else ''
    with open(output_name,'w') as f:
        f.write('''!%s %s tightscf scfconvforced
%s
%%scf SmearTemp %d
MaxIter=500
end 
* xyz %d %d\n'''%(method, dft, trah, etemp, charge, spin))
        for a, c in zip(atoms,coords):
            f.write("   %s       %f         %f        %f\n"%(a, c[0],c[1],c[2]))
        f.write('*\n')
        
def get_energy(filename):
    ####################################
    #The energy output is in Hartree!!!#
    ####################################
    #get single point energy result from ORCA output
    try:
        with open(filename,'r') as f:
            #print(f.read())
            for l in f.readlines():
                if re.search('FINAL SINGLE POINT ENERGY',l):
                    result = float(re.search('-?[1-9]\d*\.\d*|-?0\.\d*[1-9]\d*',l)[0])
    except:
        result = np.nan
    return result


def MD_sampler(source, model, collector):
    
    while source.qsize()>0:
        #species and coordinates in the input are in the same format as in h5 file
        key, species, coordinates = source.get()
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
            with in_tempdir(basedir=TMP, delete_temp=DELETE_TMP):
                write_smear_input(species, atoms.get_positions(), '1.inp')
                result = subprocess.run('export LD_LIBRARY_PATH="%s:$LD_LIBRARY_PATH"&&%s %s.inp > %s.out'%(ORCA_PATH, ORCA_PATH/'orca', 1, 1), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output_file = str(1)+'.out'
                energy = get_energy(output_file)   #sometimes DFT would fail and get_energy returns np.nan
        else:
            energy = np.nan

        #return atoms.get_positions(), energy
        #print((key, species, atoms.get_positions(), energy))
        if not np.isnan(energy):  
            collector.append((key, species, atoms.get_positions().copy(), energy))
        else:
            pass
        source.task_done()

def dataset_sampler(h5file, model_pool, save_to='ext.h5'):
    with Manager() as cl:
        collector = cl.list()
        source = queue.Queue()
        db = h5py.File(h5file,'r')
        for key in list(db.keys())[:1]:
            species = np.array(db[key]['species']).copy()
            #Due to some problem in previous vrsion of h5py
            #Species array in ani1x dataset is not purely str and need a transformation here
            #species = np.array(db[key]['species']).astype(str).copy()
            for c in db[key]['coordinates'][:1]:
                coordinates = np.array(c).copy()
                source.put((key, species, coordinates))
                
        #db.close()   #DEBUG: this should be enabled if you want to write to existing files
        
        for model in model_pool:
            threading.Thread(target=MD_sampler(source=source, model=model, collector=collector), daemon=True).start()
            
        source.join()
        ret = list(collector)
    
    if os.path.isfile(save_to):
        db_write = h5py.File(save_to,'a')
    else:
        db_write = h5py.File(save_to,'w')
    
    for info in ret:
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
    
    return ret

            
    
    