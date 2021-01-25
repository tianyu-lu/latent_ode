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
    I_extdata = I_ext.view(-1,1).repeat(1,1001).view(-1,len(I_ext)*1001).squeeze()
    with torch.no_grad():
        s = odeint(FitzHughNagumo(I_ext), s0, t, method='dopri5')
    s = s.squeeze()
    v = s[:,0]
    v = 2*(v - torch.min(v)) / (torch.max(v) - torch.min(v))
    for i in range(n_samples):
        start = int(random.random()*(10000 - 1000))
        v_data = v[start : start+1000]
        v_data = v_data[::10].reshape(-1,1)
        I_curr = I_extdata[start : start+1000]
        I_curr = I_curr[::10].reshape(-1,1)
        data[i] = torch.cat((v_data, I_curr), dim=-1)
    return data


# from https://github.com/niallmm/iSINDy/blob/master/bioutils/yeastglycolysisNM.m
def glycolysis(s, t):
    # S_1 = S(1); # Glucose
    # S_2 = S(2); # phosphate pool
    # S_3 = S(3); # 1,3-bisphosophoglycerate
    # S_4 = S(4); # Cytosolic pyruvate/ acetaldehyde pool
    # S_5 = S(5); # extracellular concentration S4 ^
    # A_3 = S(6); # ATP
    # N_2 = S(7); # NADH
    S_1, S_2, S_3, S_4, S_5, A_3, N_2 = s


    # total ADP+ATP
    A_tot = 4;
    A_2 = A_tot -A_3;

    # total NADH + NAD+
    N_tot = 1;
    N_1 = N_tot -N_2;

    # incoming flux of glucose
    J_G = 2.5; #mM/min
    # J_G = 0.2;
    # Hexokinase phosophoglucoisomerase and phosphofructokinase cooperatively
    # inhibited with ATP 
    k_1 = 100;  #mM/min max rxn rate
    K_I = 0.52; # mM inhibition constant (related to half max)
    q = 4;      # q is cooperativity coefficient   
    v_1 = k_1*S_1*A_3/(1+(A_3/K_I)**q); # mM/min rxn velocity
    #v_1 = k_1*S_1*A_3;

    # Glyceraldehydes-3-phosphate dehydrogenase linear in phosphate and ADP
    k_2 = 6.0; #mM/min rxn constant
    v_2 = k_2*S_2*N_1; # mM^3/min <WTF!!!!!!! [k_4] must be 1/(min* mM^2)

    # Phosphoglycerate kinase, phosophglycerate mutase, enolase, and pyruvate
    # kinase
    k_3 = 16.0; # mM/min rxn rate
    v_3 = k_3*S_3*A_2; # mM^3/min <WTF!!!!!!! [k_4] must be 1/(min* mM^2)

    # Alcohol dehydrogenase
    k_4 = 100; #mM/min
    v_4 = k_4*S_4*N_2; # mM^3/min <WTF!!!!!!! [k_4] must be 1/(min* mM^2)

    # Nonglycolytic ATP consumption
    k_5 = 1.28; # 1/min from supplement
    # k_5 = 18; # infered from main text
    v_5 = k_5*A_3; #mM/min 

    # Formation of glycerol from triose phosphates
    k_6 = 12.0; #mM/min supplement value
    # k_6 = 6.0; # mM/min text value Table 3.
    v_6 = k_6*S_2*N_2; # mM^3/min <WTF!!!!!!! [k_6] must be 1/(min* mM^2)

    # Degredation of pyruvate and acetaldehyde in the extracellular space
    # k = 1.8; # 1/min in supplement
    k = 18; # infered from text 
    v_7 = k*S_5;

    # Carbon sink term to pyruvate pool acounting for carbon loss to cellular
    # synthetic processes (Over specified model only)

    # Membrane transport of pyruvate and acetaldehyde into extra cellular space
    ASPV = 13.0; # 1/min
    J_P = ASPV*(S_4 - S_5); # mM/min

    # chemical species derivatives
    phi = 0.10;

    dS1 = J_G -v_1; # GLucose S1, S6
    dS2 = 2*v_1 - v_2 - v_6; # phosphate pool S1, S6, S2, S7
    dS3 = v_2 -v_3; # 1,3-bisphophoglycerate, S2, S6, S3, S7
    dS4 = v_3 -v_4 -J_P; # cytosolic pyruvate and acetaldehyde pool
    dS5 = phi*(J_P - v_7); # extracellular concentration of S_4 see above^
    dS6 = -2*v_1 + 2*v_3 - v_5; #ATP (A3) S1, S6, S3
    dS7 = v_2 -v_4 - v_6; #NADH (N2)

    return [dS1, dS2, dS3, dS4, dS5, dS6, dS7]


from scipy.integrate import odeint
import numpy as np

def sample_glyco(time_steps_extrap, n_samples = 1000, num_obs=1):
    data = torch.zeros(n_samples, 100, num_obs)
    s0 = [2.475, 0.077, 1.187, 0.193, 0.050, 0.115, 0.077]
    t = np.linspace(0, 20, 1000)
    s = odeint(glycolysis, s0, t)

    obs = s[:,:num_obs]
    # obs = 2*(obs - np.min(obs, axis=0)) / (np.max(obs, axis=0) - np.min(obs, axis=0))

    for i in range(n_samples):
        start = int(random.random()*(1000 - 100))
        data[i] = gfp[start : start+100]

    return data


def crz1data(time_steps_extrap, n_samples = 1000):
    data = torch.zeros(n_samples, 100, 1)
    crz1 = np.genfromtxt("crz1_interp.csv", delimiter=",")  # shape is (601, 2190)
    nExp, nTime = crz1.shape
    crz1 = torch.from_numpy(crz1)
    for i in range(n_samples):
        randExp = int(random.random()*(nExp))
        randTime = int(random.random()*(nTime - 100))
        data[i] = crz1[randExp, randTime:randTime+100].reshape(-1,1)
    return data


def crz1ca2data(time_steps_extrap, n_samples = 1000):
    data = torch.zeros(n_samples, 100, 2)
    crz1 = np.genfromtxt("crz1_interp.csv", delimiter=",")  # shape is (601, 2190)
    crz1 = torch.from_numpy(crz1)
    crz1 = 2*(crz1 - torch.min(crz1)) / (torch.max(crz1) - torch.min(crz1))
    ca2 = np.genfromtxt("ca_interp.csv", delimiter=",") # shape is (601, 2190)
    ca2 = torch.from_numpy(ca2)
    ca2 = 2*(ca2 - torch.min(ca2)) / (torch.max(ca2) - torch.min(ca2))
    assert crz1.shape == ca2.shape
    nExp, nTime = crz1.shape
    for i in range(n_samples):
        randExp = int(random.random()*(nExp))
        randTime = int(random.random()*(nTime - 100))
        crz1Data = crz1[randExp, randTime:randTime+100].reshape(-1,1)
        ca2Data = ca2[randExp, randTime:randTime+100].reshape(-1,1)
        data[i] = torch.cat((crz1Data, ca2Data), dim=-1)
    return data

def potvintrottier(time_steps_extrap, n_samples = 1000):
    data = torch.zeros(n_samples, 100, 1)
    potvin = np.genfromtxt("data/potvin.csv", delimiter=",") # shape is (1790,)
    potvin = torch.from_numpy(potvin)
    for i in range(n_samples):
        randTime = int(random.random()*(potvin.shape[0] - 100))
        data[i] = potvin[randTime:randTime+100].reshape(-1,1)
    return data

# Todo: 1. generalize training data to multiple initial conditions
#       2. include at least one full cycle but irregularly sampled
def sample_biotraj(time_steps_extrap, n_samples = 1000, noise_weight = 0.05, stochastic=False, num_obs = 1):
    
    data = torch.zeros(n_samples, 100, num_obs)

    if not stochastic:
        s0 = torch.tensor([[0.2,  0.1, 0.3, 0.1, 0.4, 0.5]]) # [m1 p1 m2 p2 m3 p3]
        t = torch.linspace(0., 1000., 10000)

        with torch.no_grad():
            s = odeint(Repressilator(), s0, t, method='dopri5')

        s = s.squeeze()
        gfp = s[:,:num_obs]  # originally s[:,5]
        gfp = 2*(gfp - torch.min(gfp, dim=0)) / (torch.max(gfp, dim=0) - torch.min(gfp, dim=0))

        for i in range(n_samples):
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
        dataset = sample_biotraj(time_steps_extrap, n_samples = args.n, noise_weight = 0.05, , num_obs = args.obs)
    elif dataset_name == "glycolysis":
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = sample_glyco(time_steps_extrap, n_samples = args.n, num_obs = args.obs)
    elif dataset_name == "repressilator-sde":
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = sample_biotraj(time_steps_extrap, n_samples = args.n, noise_weight = 0.05, stochastic=True)
    elif dataset_name == "potvin-trottier":
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = potvintrottier(time_steps_extrap, n_samples = args.n)
    elif dataset_name == "fitzhugh-nagumo":
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = sample_fn(time_steps_extrap, n_samples = args.n)
    elif dataset_name == 'crz1':
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = crz1data(time_steps_extrap)
    elif dataset_name == 'crz1ca2':
        time_steps_extrap = torch.linspace(0., 5., 100)
        dataset = crz1ca2data(time_steps_extrap)

    # Process small datasets
    dataset = dataset.to(device)
    time_steps_extrap = time_steps_extrap.to(device)

    if dataset_name == "fitzhugh-nagumo":
        # split by I_ext
        train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)
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


