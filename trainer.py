#TODO: make it a trainer class

import numpy as np
import h5py, os, math, tqdm, sys, pathlib, torch, torchani
from torchani.units import hartree2kcalmol, HARTREE_TO_KCALMOL
from torch_optimizer import Lamb

#training settings
start_lr = 1e-4   #start learning rate
stop_lr = 1e-6   #early stop learning rate
patience = 5   #how many epoches to wait before lower the learning rate

batch_size = 1024   #training batch size
max_epochs = 2   #DEBUG: max training epochs, for demo purpose we set only 2

def validate(model, validation):
    device = list(model.neural_networks[0].parameters())[0].device
    mse_sum = torch.nn.MSELoss(reduction='sum')   #by setting reduction='sum', the MSELoss function will return the summation of RMSE of the batch
    total_mse = 0.0
    count = 0
    for properties in validation:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        _, predicted_energies = model((species, coordinates))
        total_mse += mse_sum(predicted_energies, true_energies).item()
        count += predicted_energies.shape[0]
    return hartree2kcalmol(math.sqrt(total_mse / count))   #return mean of the loss in Kcal/mol


def train(model, training, validation, save_best='best.pt', save_latest='latest.pt'):
    #initializing optimizer, the hyperparameters here are from
    #https://aiqm.github.io/torchani/examples/nnp_training.html
    #but here we use Lamb optimizer instead of AdamW and SGD
    device = list(model.neural_networks[0].parameters())[0].device
    W = []
    b = []
    for n, p in model.named_parameters():
        #when running MD, the requires_grad will be set to False
        #reopen it here
        p.requires_grad = True
        if n.endswith('2.weight'):
            W.append({'params': p, 'weight_decay': 0.00001})
        elif n.endswith('4.weight'):
            W.append({'params': p, 'weight_decay': 0.000001})
        elif n.endswith('0.weight') or n.endswith('6.weight'):
            W.append({'params': p})
        elif n.endswith('.bias'):
            b.append(p)
    WeightOptimizer = Lamb(W,lr=start_lr)
    BiasOptimizer = Lamb(b,lr=start_lr)

    WeightScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(WeightOptimizer, factor=0.5, patience=patience, threshold=0)
    BiasScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(BiasOptimizer, factor=0.5, patience=patience, threshold=0)
    
    mse = torch.nn.MSELoss(reduction='none')


    print("training starting from epoch", WeightScheduler.last_epoch + 1)
    best_rmse = 1e10
    for e in range(WeightScheduler.last_epoch + 1, max_epochs):
        torch.cuda.empty_cache()
        rmse = validate(model, validation)
        print('RMSE:', rmse, 'at epoch', e+1)

        learning_rate = WeightOptimizer.param_groups[0]['lr']

        if learning_rate < stop_lr:
            break

        # checkpoint
        if WeightScheduler.is_better(rmse, WeightScheduler.best):
            torch.save(model.state_dict(), save_best)
            best_rmse = rmse

        WeightScheduler.step(rmse)
        BiasScheduler.step(rmse)


        for i, properties in tqdm.tqdm(
            enumerate(training),
            total=len(training),
            desc="epoch {}".format(WeightScheduler.last_epoch)
        ):
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device).float()
            num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype).to(device)
            _, predicted_energies = model((species,coordinates))

            loss = (mse(predicted_energies.float(), true_energies) / num_atoms.sqrt()).mean()
            
            WeightOptimizer.zero_grad()
            BiasOptimizer.zero_grad()
            loss.backward()
            WeightOptimizer.step()
            BiasOptimizer.step()
        
        
        torch.save({
            'model': model.state_dict(),
            'WeightOptimizer': WeightOptimizer.state_dict(),
            'BiasOptimizer': BiasOptimizer.state_dict(),
            'WeightScheduler': WeightScheduler.state_dict(),
            'BiasScheduler': BiasScheduler.state_dict(),
        }, save_latest)
    
    print('best RMSE at this time is:%.3f'%(best_rmse))
    
    return model


'''
#Abandoned
def selector(model, datapool, qbc_cutoff=0.0020):
    
    ret = {'species':None,'coordinates':None}
    for properties in datapool:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float()
        s, predicted_energies, qbc = model.energies_qbcs((species,coordinates))
        selection = (qbc > qbc_cutoff)
        if ret['species']!=None:
            ret['species'] = torch.cat((ret['species'], properties['species'].to('cpu')[selection]))
            ret['coordinates'] = torch.cat((ret['coordinates'], properties['coordinates'].to('cpu')[selection]))
        else:
            ret['species'] = properties['species'].to('cpu')[selection]
            ret['coordinates'] = properties['coordinates'].to('cpu')[selection]
    return ret
'''
