###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from person_activity import PersonActivity, variable_time_collate_fn_activity

from sklearn import model_selection
import random

### Data for Repressilator ###

class Lambda(nn.Module):

    def forward(self, t, s):
        m1 = s[0,0]
        p1 = s[0,1]
        m2 = s[0,2]
        p2 = s[0,3]
        m3 = s[0,4]
        p3 = s[0,5]

        m1ddt = alpha0 + alpha/(1+p3**n) - m1
        p1ddt = beta*(m1-p1)
        m2ddt = alpha0 + alpha/(1+p1**n) - m2
        p2ddt = beta*(m2-p2)
        m3ddt = alpha0 + alpha/(1+p2**n) - m3
        p3ddt = beta*(m3-p3)

        return torch.tensor([[m1ddt, p1ddt, m2ddt, p2ddt, m3ddt, p3ddt]])

class Hill(nn.Module):

    def forward(self, p):
        n = 2
        return 1/(1 + p**n)

hill_fn = Hill()
alpha0 = 60*(5e-4) 
alpha = 60*(5-alpha0)
beta = 0.2
class Repressilator(nn.Module):
    def __init__(self):
        super(Repressilator, self).__init__()
        self.W1 = torch.tensor([[-1,0,0,0,0,0], [beta,-beta,0,0,0,0], 
                                [0,0,-1,0,0,0], [0,0,beta,-beta,0,0], 
                                [0,0,0,0,-1,0], [0,0,0,0,beta,-beta]])
        self.W2 = torch.tensor([[0,0,0,0,0,alpha], [0,0,0,0,0,0], 
                                [0,alpha,0,0,0,0], [0,0,0,0,0,0], 
                                [0,0,0,alpha,0,0], [0,0,0,0,0,0]])
        self.b = torch.tensor([alpha0, 0,alpha0, 0,alpha0, 0]).reshape(6,1)

    def forward(self, t, s):
        H = hill_fn(s)
        out = self.W1 @ s.T + self.W2 @ H.T + self.b
        return out.reshape(1,6)

class FitzHughNagumo(nn.Module):
    def __init__(self, I_ext):
        super(FitzHughNagumo, self).__init__()
        self.p = torch.tensor([0.7, 0.8, 12.5])
        self.I = I_ext

    def forward(self, t, s):
        v = s[0,0]
        w = s[0,1]
        a = self.p[0]
        b = self.p[1]
        tau = self.p[2]
        return torch.tensor([[v - (v**3)/3 - w + self.I[int(t.item())],
                              (v + a - b*w)/tau]])

from torchdiffeq import odeint_adjoint as odeint

def sample_fn(time_steps_extrap, n_samples = 1000):
    data = torch.zeros(n_samples, 100, 2)
    s0 = torch.tensor([[1.0, 0.0]])
    t = torch.linspace(0., 1000., 10000)
    I_ext = torch.tensor([0.3242, 0.3243, 0.3253, 0.3353, 0.4353,
                      0.3242, 0.3243, 0.3253, 0.3353, 0.4353])
    I_ext = I_ext.view(-1,1).repeat(1,101).view(-1,len(I_ext)*101).squeeze()
    with torch.no_grad():
        s = odeint(FitzHughNagumo(I_ext), s0, t, method='dopri5')
    s = s.squeeze()
    v = s[:,0]
    v = 2*(v - torch.min(v)) / (torch.max(v) - torch.min(v))
    for i in range(n_samples):
        start = int(random.random()*(10000 - 1000))
        v_data = v[start : start+1000]
        v_data = v_data[::10].reshape(-1,1)
        I_ext = 
        data[i] = 


# Todo: 1. generalize training data to multiple initial conditions
#       2. include at least one full cycle but irregularly sampled
def sample_biotraj(time_steps_extrap, n_samples = 1000, noise_weight = 0.05, stochastic=False):
    
    data = torch.zeros(n_samples, 100, 1)

    if not stochastic:
        s0 = torch.tensor([[0.2,  0.1, 0.3, 0.1, 0.4, 0.5]]) # [m1 p1 m2 p2 m3 p3]
        # trajs = []
        # t = torch.linspace(0., 300., 3000)
        # for _ in range(100):
        #     s0 = 100*torch.rand(6).reshape(1,6)
        #     with torch.no_grad():
        #         s = odeint(Repressilator(), s0, t, method='dopri5')
        #     trajs.append(s)
        # t = torch.linspace(0., 200., 2000)
        t = torch.linspace(0., 1000., 10000)
        # t = time_steps_extrap

        with torch.no_grad():
            s = odeint(Repressilator(), s0, t, method='dopri5')

        s = s.squeeze()
        gfp = s[:,5]
        gfp = 2*(gfp - torch.min(gfp)) / (torch.max(gfp) - torch.min(gfp))

        for i in range(n_samples):
            # rand_traj = int(random.random()*len(trajs))
            # s = trajs[rand_traj]
            # s = s.squeeze()
            # gfp = s[:,5]
            # gfp = 2*(gfp - torch.min(gfp)) / (torch.max(gfp) - torch.min(gfp))
            start = int(random.random()*(10000 - 1000))
            gfp_data = gfp[start : start+1000]
            data[i] = gfp_data[::10].reshape(-1,1)
    else:
        df = pd.read_csv("data/StochasticRepressilator.csv")
        gfp = torch.from_numpy(np.array(df["p3"]))
        gfp = 2*(gfp - torch.min(gfp)) / (torch.max(gfp) - torch.min(gfp))
        for i in range(n_samples):
            group = random.randint(1, 5)
            start = int(random.random()*(10000 - 1000)) * group
            gfp_data = gfp[start : start+1000]
            data[i] = gfp_data[::10].reshape(-1,1)
    return data

#####################################################################################################
def parse_datasets(args, device):
    

    def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
        batch = torch.stack(batch)
        data_dict = {
            "data": batch, 
            "time_steps": time_steps}

        data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
        return data_dict


    dataset_name = args.dataset

    n_total_tp = args.timepoints + args.extrap
    max_t_extrap = args.max_t / args.timepoints * n_total_tp

    ##################################################################
    # MuJoCo dataset
    if dataset_name == "hopper":
        dataset_obj = HopperPhysics(root='data', download=True, generate=False, device = device)
        dataset = dataset_obj.get_dataset()[:args.n]
        dataset = dataset.to(device)


        n_tp_data = dataset[:].shape[1]

        # Time steps that are used later on for exrapolation
        time_steps = torch.arange(start=0, end = n_tp_data, step=1).float().to(device)
        time_steps = time_steps / len(time_steps)

        dataset = dataset.to(device)
        time_steps = time_steps.to(device)

        if not args.extrap:
            # Creating dataset for interpolation
            # sample time points from different parts of the timeline, 
            # so that the model learns from different parts of hopper trajectory
            n_traj = len(dataset)
            n_tp_data = dataset.shape[1]
            n_reduced_tp = args.timepoints

            # sample time points from different parts of the timeline, 
            # so that the model learns from different parts of hopper trajectory
            start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
            end_ind = start_ind + n_reduced_tp
            sliced = []
            for i in range(n_traj):
                  sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
            dataset = torch.stack(sliced).to(device)
            time_steps = time_steps[:n_reduced_tp]

        # Split into train and test by the time sequences
        train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

        n_samples = len(dataset)
        input_dim = dataset.size(-1)

        batch_size = min(args.batch_size, args.n)
        train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
            collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "train"))
        test_dataloader = DataLoader(test_y, batch_size = n_samples, shuffle=False,
            collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "test"))
        
        data_objects = {"dataset_obj": dataset_obj, 
                    "train_dataloader": utils.inf_generator(train_dataloader), 
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader)}
        return data_objects

    ##################################################################
    # Physionet dataset

    if dataset_name == "physionet":
        train_dataset_obj = PhysioNet('data/physionet', train=True, 
                                        quantization = args.quantization,
                                        download=True, n_samples = min(10000, args.n), 
                                        device = device)
        # Use custom collate_fn to combine samples with arbitrary time observations.
        # Returns the dataset along with mask and time steps
        test_dataset_obj = PhysioNet('data/physionet', train=False, 
                                        quantization = args.quantization,
                                        download=True, n_samples = min(10000, args.n), 
                                        device = device)

        # Combine and shuffle samples from physionet Train and physionet Test
        total_dataset = train_dataset_obj[:len(train_dataset_obj)]

        if not args.classif:
            # Concatenate samples from original Train and Test sets
            # Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
            total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, 
            random_state = 42, shuffle = True)

        record_id, tt, vals, mask, labels = train_data[0]

        n_samples = len(total_dataset)
        input_dim = vals.size(-1)

        batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
        data_min, data_max = get_data_min_max(total_dataset)

        train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
            collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
                data_min = data_min, data_max = data_max))
        test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
            collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
                data_min = data_min, data_max = data_max))

        attr_names = train_dataset_obj.params
        data_objects = {"dataset_obj": train_dataset_obj, 
                    "train_dataloader": utils.inf_generator(train_dataloader), 
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "attr": attr_names, #optional
                    "classif_per_tp": False, #optional
                    "n_labels": 1} #optional
        return data_objects

    ##################################################################
    # Human activity dataset

    if dataset_name == "activity":
        n_samples =  min(10000, args.n)
        dataset_obj = PersonActivity('data/PersonActivity', 
                            download=True, n_samples =  n_samples, device = device)
        print(dataset_obj)
        # Use custom collate_fn to combine samples with arbitrary time observations.
        # Returns the dataset along with mask and time steps

        # Shuffle and split
        train_data, test_data = model_selection.train_test_split(dataset_obj, train_size= 0.8, 
            random_state = 42, shuffle = True)

        train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
        test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

        record_id, tt, vals, mask, labels = train_data[0]
        input_dim = vals.size(-1)

        batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
        train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
            collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
        test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False, 
            collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

        data_objects = {"dataset_obj": dataset_obj, 
                    "train_dataloader": utils.inf_generator(train_dataloader), 
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "input_dim": input_dim,
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader),
                    "classif_per_tp": True, #optional
                    "n_labels": labels.size(-1)}

        return data_objects

    ########### 1d datasets ###########

    # Sampling args.timepoints time points in the interval [0, args.max_t]
    # Sample points for both training sequence and explapolation (test)
    distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
    time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
    time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
    time_steps_extrap = torch.sort(time_steps_extrap)[0]

    dataset_obj = None
    ##################################################################
    # Sample a periodic function
    if dataset_name == "periodic":
        dataset_obj = Periodic_1d(
            init_freq = None, init_amplitude = 1.,
            final_amplitude = 1., final_freq = None, 
            z0 = 1.)

    ##################################################################

    # if dataset_obj is None:
    #   raise Exception("Unknown dataset: {}".format(dataset_name))

    if dataset_name == "periodic":
        dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, 
            noise_weight = args.noise_weight)
    elif dataset_name == "repressilator":
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = sample_biotraj(time_steps_extrap, n_samples = args.n, noise_weight = 0.05)
    elif dataset_name == "repressilator-sde":
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = sample_biotraj(time_steps_extrap, n_samples = args.n, noise_weight = 0.05, stochastic=True)
    elif dataset_name == "fitzhugh-nagumo":
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = sample_fn(time_steps_extrap, n_samples = args.n)

    # Process small datasets
    dataset = dataset.to(device)
    time_steps_extrap = time_steps_extrap.to(device)

    if dataset_name == "fitzhugh-nagumo":
        # split by I_ext

    else:
        train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

    n_samples = len(dataset)
    input_dim = dataset.size(-1)

    batch_size = min(args.batch_size, args.n)
    train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
    test_dataloader = DataLoader(test_y, batch_size = args.n, shuffle=False,
        collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))
    
    data_objects = {#"dataset_obj": dataset_obj, 
                "train_dataloader": utils.inf_generator(train_dataloader), 
                "test_dataloader": utils.inf_generator(test_dataloader),
                "input_dim": input_dim,
                "n_train_batches": len(train_dataloader),
                "n_test_batches": len(test_dataloader)}

    return data_objects


