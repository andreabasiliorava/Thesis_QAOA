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


#main code

# grid graph
#n_nodes = 4
#nodes = np.arange(0, n_nodes, 1)
#edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

# grid graph
n_nodes = 9
nodes = np.arange(0, n_nodes, 1)
edges = [(0, 1), (0,3), (1, 2), (1, 4), (2, 5), (3,4), (3,6), (4, 5), (4, 7), (5,8), (6, 7), (7, 8)]

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
probabilities = np.arange(1, 0.7, -0.02)

file1 = open('qaoa_ising_nodes='+str(n_nodes)
             +'levels='+str(n_levels)
             +'.dat'
             , 'a')
for prob in probabilities:
    file2 = open('qaoa_ising_nodes='+str(n_nodes)
                 +'levels='+str(n_levels)
                 +'prob='+str(prob)
                 +'.dat'
                 , 'a')
    prob_distributions = []
    energies = []
    magnetizations = []
    ising_results = []
    for j in range(N):
        #adam_steps = []
        
        #define bin_prob_dist
        bin_prob_dist = ising.binomial_dist(prob, edges)
                
        #Adam
        parameters = 0.01*np.random.rand(2*n_levels)
        m_t = np.zeros(2*n_levels)
        v_t = np.zeros(2*n_levels)
        t = 0
        H_P = ising.prob_hamilt_ising(n_qubits, edges, bin_prob_dist) 
        while t <= n_steps:
            g_t = qaoa.fin_diff_grad(ising.evaluate_energy_p, parameters, 
                                     args=(n_qubits, edges, bin_prob_dist, 100), increment=0.1)
            t = t+1
            m_t = beta_1*m_t + (1-beta_1)*g_t
            v_t = beta_2*v_t + (1-beta_2)*g_t**2
            m_t_hat = m_t/(1-beta_1**t)
            v_t_hat = v_t/(1-beta_2**t)
            parameters = parameters - eta*m_t_hat/(np.sqrt(v_t_hat) + epsilon)
            gammas = parameters[:n_levels]
            betas = parameters[n_levels:]
            fin_state = qaoa.evolution_operator(n_qubits, edges, gammas, betas)*qaoa.init_state
            E_p = qu.expect(H_P, fin_state)
            #adam_steps.append([t]+[parameters]+[E_p])
            np.savetxt(file2, [t]+[parameters]+[E_p])
        
        #evalute estimations of energy and magnetization
        outcomes = ising.quantum_measurements_ising(1000, fin_state)
        dict_outcomes = {}
        for outcome in outcomes:
            dict_outcomes[tuple(outcome)] = outcomes.count(outcome)
        max_occurrency = max(list(dict_outcomes.values()))
        index_max_occurrency = list(dict_outcomes.values()).index(max_occurrency)
        most_probable_state = list(dict_outcomes.keys())[index_max_occurrency]
        magnetization = ising.evaluate_magnetization_ising(most_probable_state)
        magnetizations.append(magnetization)
        energy = ising.evaluate_energy_p(parameters, n_qubits, edges, bin_prob_dist, 1000)
        energies.append(energy)
        prob_dist = qucs.comp_basis_prob_dist(fin_state)
        prob_distributions.append(prob_dist)
    
    ising_results.append([prob]+[sum(energies)/N]+[sum(magnetizations)/N]+sum[prob_distributions]/N)
    np.savetxt(file1, [prob]+[sum(energies)/N]+[sum(magnetizations)/N]+sum[prob_distributions]/N)

    file2.close()

file1.close()






