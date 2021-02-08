# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 01:00:27 2021

@author: AndreaB.Rava
"""

import qaoa
import qucompsys as qucs
import qutip as qu
import numpy as np
#import matplotlib.pyplot as plt
#from   matplotlib import cm
#from   matplotlib.ticker import LinearLocator, FormatStrFormatter
import networkx as nx
#from   networkx.generators.random_graphs import erdos_renyi_graph
#import configparser
#import scipy
#import itertools


def evaluate_energy_ising(list_z, edges, bin_prob_dist, coupling_const=1):
    energy = 0
    for i, edge in enumerate(edges):
        energy += -bin_prob_dist[i]*list_z[edge[0]]*list_z[edge[1]]
    return coupling_const*energy

def evaluate_magnetization_ising(list_z):
    return abs(sum(list_z))

def prob_hamilt_ising(n_qubits, edges, bin_prob_dist, coupling_const=1):
    if n_qubits < 2:
        raise ValueError('number of qubits must be > 1, but is {}'.format(n_qubits))
    list_double_sigmaz = []
    for i, edge in enumerate(edges):
        list_double_sigmaz.append(
            qucs.n_sigmaz(n_qubits,edge[0])*qucs.n_sigmaz(n_qubits,edge[1])#*bin_prob_dist[i]
            ) 
    return -sum(list_double_sigmaz)*coupling_const

def evolution_operator_ising(n_qubits, edges, gammas, betas, bin_prob_dist):
    if len(gammas) < 1:
        raise ValueError('number of gammas must be > 0, but is {}'.format(len(gammas)))
    if len(betas) < 1:
        raise ValueError('number of gammas must be > 0, but is {}'.format(len(betas)))
    if len(betas) != len(gammas):
        raise ValueError('number of gammas must be = number of betas')
    if n_qubits < 2:
        raise ValueError('number of qubits must be > 1, but is {}'.format(n_qubits))
    evol_oper = qucs.n_qeye(n_qubits)
    for i in range(len(gammas)):
        u_mix_hamilt_i = (-complex(0,betas[i])*qaoa.mix_hamilt(n_qubits)).expm()
        #u_prob_hamilt_i = (-complex(0,gammas[i])*qaoa.prob_hamilt(n_qubits, edges)).expm()
        u_prob_hamilt_i = (-complex(0,gammas[i])*prob_hamilt_ising(n_qubits, edges, bin_prob_dist)).expm()
        evol_oper = u_mix_hamilt_i*u_prob_hamilt_i*evol_oper
    return evol_oper

def single_qubit_measurement_ising(qstate, qubit_pos):
    n_qubits = len(qstate.dims[0])
    if qstate.dims[1][0] == 1:
        qstate = qu.ket2dm(qstate)
    M_i = (qucs.n_proj0(n_qubits, qubit_pos)*qstate)
    p0_i = M_i.tr()
    #p1_i = (n_proj1(n_qubits, i)*dm_dummy).tr()
    if np.random.random_sample() <= p0_i:
        outcome = [1]
        qstate = M_i/p0_i
    else:
        outcome = [-1]
        qstate = (qucs.n_proj1(n_qubits, qubit_pos)*qstate)/(1-p0_i)
    return outcome, qstate

def quantum_measurements_ising(n_samples, qstate):
    n_qubits = len(qstate.dims[0])
    if qstate.dims[1][0] == 1:
        qstate = qu.ket2dm(qstate)
    outcomes = []
    for j in range(n_samples):
        outcome = []
        qstate_dummy = qstate.copy()
        for i in range(n_qubits):
            outcome_i, qstate_dummy = single_qubit_measurement_ising(qstate_dummy, i)
            outcome += outcome_i
        outcomes.append(outcome)
    return outcomes

def evaluate_energy_p(params, n_qubits, edges, bin_prob_dist, n_samples):
    gammas = params[:int(len(list(params))/2)]
    betas = params[int(len(list(params))/2):]
    
    # initial state (as density matrix):
    init_state = qaoa.initial_state(n_qubits)
    #obtain final state
    fin_state = evolution_operator_ising(n_qubits, edges, gammas, betas, bin_prob_dist)*init_state

    
    #Perform N measurements on each single qubit of final state
    outcomes = quantum_measurements_ising(n_samples, fin_state)
    dict_outcomes = {}
    for outcome in outcomes:
        dict_outcomes[tuple(outcome)] = outcomes.count(outcome)
    
    #Evaluate Fp
    Ep = 0
    for outcome_w in dict_outcomes:
        Ep += dict_outcomes[outcome_w]*evaluate_energy_ising(outcome_w, edges, bin_prob_dist)
    return Ep/n_samples

def evaluate_magnetization_p(params, n_qubits, edges, bin_prob_dist, n_samples):
    gammas = params[:int(len(list(params))/2)]
    betas = params[int(len(list(params))/2):]
    
    # initial state (as density matrix):
    #dm_init_state = qu.ket2dm(initial_state(n_qubits))
    init_state = qaoa.initial_state(n_qubits)
    #obtain final state
    #dm_fin_state = evolution_operator(n_qubits, edges, gammas, betas)*dm_init_state*evolution_operator(n_qubits, edges, gammas, betas).dag()
    fin_state = evolution_operator_ising(n_qubits, edges, gammas, betas, bin_prob_dist)*init_state
    
    #Perform N measurements on each single qubit of final state
    outcomes = quantum_measurements_ising(n_samples, fin_state)
    dict_outcomes = {}
    for outcome in outcomes:
        dict_outcomes[tuple(outcome)] = outcomes.count(outcome)
    
    #Evaluate Fp
    Mp = 0
    for outcome_w in dict_outcomes:
        Mp += dict_outcomes[outcome_w]*evaluate_magnetization_ising(outcome_w)
    return Mp/n_samples

#main code

# grid graph
n_nodes = 4
nodes = np.arange(0, n_nodes, 1)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

n_qubits = n_nodes
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
edges = list(graph.edges)

n_levels = 1
n_steps = 20
eta = 0.05
beta_1 = 0.8
beta_2 = 0.999
epsilon = 1e-8
N = 10
prob_E_M_4 = []
n_steps = 20

probabilities = np.arange(0.80, 1, 0.02)

for prob in probabilities:
    energies = []
    magnetizations = []
    for j in range(N):
        
        #define bin_prob_dist
        bin_prob_dist = np.random.binomial(1, prob, size=(len(edges),))
        for i, link in enumerate (bin_prob_dist):
            if link == 0:
                bin_prob_dist[i] = -1
                
        #Adam
        parameters = 0.01*np.random.rand(2*n_levels)
        m_t = np.zeros(2*n_levels)
        v_t = np.zeros(2*n_levels)
        t = 0
        for i in range(n_steps):
            g_t = qaoa.fin_diff_grad(evaluate_energy_p, parameters, args=(n_qubits, edges, bin_prob_dist, 100), increment=0.1)
            t = t+1
            m_t = beta_1*m_t + (1-beta_1)*g_t
            v_t = beta_2*v_t + (1-beta_2)*g_t**2
            m_t_hat = m_t/(1-beta_1**t)
            v_t_hat = v_t/(1-beta_2**t)
            parameters = parameters - eta*m_t_hat/(np.sqrt(v_t_hat) + epsilon)
        
        #evalute estimations of energy and magnetization
        energy = evaluate_energy_p(parameters, n_qubits, edges, bin_prob_dist, 1000)
        energies.append(energy)
        magnetization = evaluate_magnetization_p(parameters, n_qubits, edges, bin_prob_dist, 1000)
        magnetizations.append(magnetization)
        
    #store results in a list
    mean_en = sum(energies)/N
    mean_mag = sum(magnetizations)/N
    prob_E_M_4.append([prob, mean_en, mean_mag])

#store result in a file
np.savetxt('Ising_4nodes_prob_en_mag.txt', prob_E_M_4)

# grid graph
n_nodes = 9
nodes = np.arange(0, n_nodes, 1)
edges = [(0, 1), (0,3), (1, 2), (1, 4), (2, 5), (3,4), (3,6), (4, 5), (4, 7), (5,8), (6, 7), (7, 8)]

n_qubits = n_nodes
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
edges = list(graph.edges)

prob_E_M_9 = []

for prob in probabilities:
    energies = []
    magnetizations = []
    for j in range(N):
        
        #define bin_prob_dist
        bin_prob_dist = np.random.binomial(1, prob, size=(len(edges),))
        for i, link in enumerate (bin_prob_dist):
            if link == 0:
                bin_prob_dist[i] = -1
                
        #Adam
        parameters = 0.01*np.random.rand(2*n_levels)
        m_t = np.zeros(2*n_levels)
        v_t = np.zeros(2*n_levels)
        t = 0
        for i in range(n_steps):
            g_t = qaoa.fin_diff_grad(evaluate_energy_p, parameters, args=(n_qubits, edges, bin_prob_dist, 100), increment=0.1)
            t = t+1
            m_t = beta_1*m_t + (1-beta_1)*g_t
            v_t = beta_2*v_t + (1-beta_2)*g_t**2
            m_t_hat = m_t/(1-beta_1**t)
            v_t_hat = v_t/(1-beta_2**t)
            parameters = parameters - eta*m_t_hat/(np.sqrt(v_t_hat) + epsilon)
        
        #evalute estimations of energy and magnetization
        energy = evaluate_energy_p(parameters, n_qubits, edges, bin_prob_dist, 1000)
        energies.append(energy)
        magnetization = evaluate_magnetization_p(parameters, n_qubits, edges, bin_prob_dist, 1000)
        magnetizations.append(magnetization)
        
    #store results in a list
    mean_en = sum(energies)/N
    mean_mag = sum(magnetizations)/N
    prob_E_M_9.append([prob, mean_en, mean_mag])

#store result in a file
np.savetxt('Ising_9nodes_prob_en_mag.txt', prob_E_M_9)






