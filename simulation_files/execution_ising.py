# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 01:00:27 2021

@author: AndreaB.Rava
"""

import qaoa
import qucompsys as qucs
import qutip as qu
import numpy as np
import ising
import networkx as nx



import argparse
parser = argparse.ArgumentParser(description = "Simulate Ising Hamitonian with QAOA")
parser.add_argument('--prob',  type=float, default=1.0, help = "Probability of ferromagnetic edges")

args = parser.parse_args()

prob = args.prob

np.random.seed(4)
#main code

# grid graph
#n_nodes = 4
#nodes = np.arange(0, n_nodes, 1)
#edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

# grid graph
#n_nodes = 6
#nodes = np.arange(0, n_nodes, 1)
#edges = [(0, 1), (1, 2), (1,4), (2, 3), (3, 4), (4, 5), (5, 0)]

# grid graph
n_nodes = 9
nodes = np.arange(0, n_nodes, 1)
edges = [(0, 1), (0,3), (1, 2), (1, 4), (2, 5), (3,4), (3,6), (4, 5), (4, 7), (5,8), (6, 7), (7, 8)]


n_qubits = n_nodes
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
edges = list(graph.edges)

I = qu.tensor([qu.qeye(2)]*n_qubits)
X = []
for i in range(n_qubits):
    X.append(qu.tensor([qu.qeye(2)]*i+[qu.sigmax()]+[qu.qeye(2)]*(n_qubits-i-1)))
Y = []
for i in range(n_qubits):
    Y.append(qu.tensor([qu.qeye(2)]*i+[qu.sigmay()]+[qu.qeye(2)]*(n_qubits-i-1)))
Z = []
for i in range(n_qubits):
    Z.append(qu.tensor([qu.qeye(2)]*i+[qu.sigmaz()]+[qu.qeye(2)]*(n_qubits-i-1)))
P_0 = []
for i in range(n_qubits):
    P_0.append(qu.tensor([qu.qeye(2)]*i+[qu.ket('0').proj()]+[qu.qeye(2)]*(n_qubits-i-1)))
P_1 = []
for i in range(n_qubits):
    P_1.append(qu.tensor([qu.qeye(2)]*i+[qu.ket('1').proj()]+[qu.qeye(2)]*(n_qubits-i-1)))
H_B = sum(X)

def single_qubit_measurement_ising(qstate, qubit_pos):
    M_i = P_0[qubit_pos]*qstate
    if qstate.dims[1][0] == 1:
        p0_i = float(abs((qstate.dag()*M_i).full()))
        if np.random.random_sample() <= p0_i:
            outcome = [1]
            qstate = M_i/np.sqrt(p0_i)
        else:
            outcome = [-1]
            qstate = (P_1[qubit_pos]*qstate)/np.sqrt((1-p0_i))
    else:
        p0_i = M_i.tr()
        if np.random.random_sample() <= p0_i:
            outcome = [1]
            qstate = M_i/p0_i
        else:
            outcome = [-1]
            qstate = (P_1[qubit_pos]*qstate)/(1-p0_i)
    return outcome, qstate

def quantum_measurements_ising(n_samples, qstate):
    n_qubits = len(qstate.dims[0])
    outcomes = []
    for j in range(n_samples):
        outcome = []
        qstate_dummy = qstate.copy()
        for i in range(n_qubits):
            outcome_i, qstate_dummy = single_qubit_measurement_ising(qstate_dummy, i)
            outcome += outcome_i
        outcomes.append(outcome)
    return outcomes

def prob_hamilt_ising(n_qubits, edges, bin_prob_dist, coupling_const=1):
    list_double_sigmaz = []
    for i, edge in enumerate(edges):
        list_double_sigmaz.append(Z[edge[0]]*Z[edge[1]]*bin_prob_dist[i]) 
    return -sum(list_double_sigmaz)*coupling_const

def evolution_operator_ising(n_qubits, edges, gammas, betas, bin_prob_dist):
    evol_oper = I
    for i in range(len(gammas)):
        u_mix_hamilt_i = (-complex(0,betas[i])*H_B).expm()
        u_prob_hamilt_i = (-complex(0,gammas[i])*prob_hamilt_ising(n_qubits, edges, bin_prob_dist)).expm()
        evol_oper = u_mix_hamilt_i*u_prob_hamilt_i*evol_oper
    return evol_oper

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
        Ep += dict_outcomes[outcome_w]*ising.evaluate_energy_ising(outcome_w, edges, bin_prob_dist)
    return Ep/n_samples


n_levels = 1 #depth QAOA
n_steps = 30 #steps gradient descent
eta = 0.05
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
N = 5 #number of times gradient descent is executed for a given probability

init_state = qaoa.initial_state(n_qubits)

file1 = open(f"qaoa_ising_final_nodes_{n_nodes}_level_{n_levels}_prob_{prob}.dat", 'w')
file2 = open(f"qaoa_ising_GD_params_nodes_{n_nodes}_level_{n_levels}_prob_{prob}.dat", 'w')
file3 = open(f"qaoa_ising_confs_nodes_{n_nodes}_level_{n_levels}_prob_{prob}.dat", 'w')

for j in range(N):
    
    #define bin_prob_dist
    bin_prob_dist = ising.binomial_dist(prob, edges)
    print(bin_prob_dist)
    #Adam
    parameters = np.array(np.random.uniform(-np.pi/4, np.pi/4, 2*n_levels)) 
    m_t = np.zeros(2*n_levels)
    v_t = np.zeros(2*n_levels)
    t = 0
    H_P = prob_hamilt_ising(n_qubits, edges, bin_prob_dist) 
    while t <= n_steps:
        gammas = parameters[:n_levels]
        betas = parameters[n_levels:]
        fin_state = evolution_operator_ising(n_qubits, edges, gammas, betas, bin_prob_dist)*init_state
        E_p = qu.expect(H_P, fin_state)
        print([j, t] + parameters.tolist() + [E_p])
        np.savetxt(file2, [[j, t] + parameters.tolist() + [E_p]])        
        g_t = qaoa.fin_diff_grad(evaluate_energy_p, parameters, 
                                 args=(n_qubits, edges, bin_prob_dist, 100), increment=0.1)
        t = t+1
        m_t = beta_1*m_t + (1-beta_1)*g_t
        v_t = beta_2*v_t + (1-beta_2)*g_t**2
        m_t_hat = m_t/(1-beta_1**t)
        v_t_hat = v_t/(1-beta_2**t)
        parameters = parameters - eta*m_t_hat/(np.sqrt(v_t_hat) + epsilon)

   
    #evalute estimations of energy and magnetization

    prob_dist = qucs.comp_basis_prob_dist(fin_state)
    index_max = np.argmax(prob_dist)
    conf_max_prob_bin_str = bin(index_max)[2:].zfill(n_nodes)
    most_probable_state = [1 - 2*(int(_)) for _ in conf_max_prob_bin_str]

    energy = ising.evaluate_energy_ising(most_probable_state, edges, bin_prob_dist)    
    magnetization = ising.evaluate_magnetization_ising(most_probable_state)
    
    print([prob, j]+parameters.tolist()+[energy, magnetization])

    np.savetxt(file1, [[prob, j]+ parameters.tolist()+[energy, magnetization]])
    np.savetxt(file3, [prob_dist])

file1.close()
file2.close()
file3.close()





