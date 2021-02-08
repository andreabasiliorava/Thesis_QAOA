# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:14:48 2021

@author: AndreaB.Rava
"""

import numpy as np
import qutip as qu
import qucompsys as qucs
import graphs as gr
from collections import Counter

def single_term_cost_fun (z_str):
    list_z = list(z_str)
    return (int(list_z[0]) - int(list_z[1]))**2


def evaluate_cost_fun(z_str, edges):
    """This method evaluates the object function of the MaxCut problem
    Parameters
        z_str : input bit string
        edges : edges of the graph
    Returns
        the integer value the object function"""
    obj = 0
    z_list = list(z_str)
    for edge in edges:
        obj += (int(z_list[edge[0]])-int(z_list[edge[1]]))**2
    return obj

def analitic_F_1(parameters, graph, edges):
    f_1 = 0
    gamma = parameters[0]
    beta = parameters[1]
    for edge in edges:
        degree_u = gr.node_degree(graph, edge[0])
        degree_v = gr.node_degree(graph, edge[1])
        lambda_uv = gr.common_neighbours(graph, edge[0], edge[1])
        c_uv = 0.5+0.25*np.sin(4*beta)*np.sin(gamma)*(np.cos(gamma)**(degree_u-1) + np.cos(gamma)**(degree_v-1))
        -0.25*np.sin(beta)**2*np.cos(gamma)**(degree_u+degree_v-2-2*lambda_uv)*(1-np.cos(2*gamma)**lambda_uv)
        f_1 += c_uv
    return f_1

def minus_analitic_F_1(parameters, graph, edges):
    f_1 = 0
    gamma = parameters[0]
    beta = parameters[1]
    for edge in edges:
        degree_u = gr.node_degree(graph, edge[0])
        degree_v = gr.node_degree(graph, edge[1])
        lambda_uv = gr.common_neighbours(graph, edge[0], edge[1])
        c_uv = 0.5+0.25*np.sin(4*beta)*np.sin(gamma)*(np.cos(gamma)**(degree_u-1) + np.cos(gamma)**(degree_v-1))
        -0.25*np.sin(beta)**2*np.cos(gamma)**(degree_u+degree_v-2-2*lambda_uv)*(1-np.cos(2*gamma)**lambda_uv)
        f_1 += c_uv
    return -f_1


def initial_state(n_qubits):
    """This method initialize the initial state\n
    Parameters:\n
        n_qubits: number of qubits the states is composed of\n
    Returns:\n
        an object of the Qobj class defined in qutip, tensor of n-qubits\n
    Raise:\n
        ValueError if number of qubits is less than 1"""
    if n_qubits < 1:
        raise ValueError('number of qubits must be > 0, but is {}'.format(n_qubits))
    list_s = []
    i = 0
    while i < n_qubits:
        list_s.append((qu.basis(2, 0) + qu.basis(2, 1)).unit())
        i += 1
    init_state = qu.tensor(list_s)
    return init_state


def mix_hamilt(n_qubits):
    """This method generates a tensor that apply the mixing hamiltonian of the
        MaxCut problem on a state of n-qubits\n
    Parameters:\n
        n_qubits: number of qubits the states is composed of\n
    Returns:\n
        a tensor that apply the mixing hamiltonian to a n-qubits state\n
    Raise:\n
        ValueError if number of qubits is less than 2"""
    if n_qubits < 2:
        raise ValueError('number of qubits must be > 1, but is {}'.format(n_qubits))
    list_n_sigmax = []
    for i in range(n_qubits):
        list_n_sigmax.append(qucs.n_sigmax(n_qubits,i))
    return sum(list_n_sigmax)


def prob_hamilt(n_qubits, edges):
    """This method generates a tensor that apply the problem hamiltonian of the
        MaxCut problem on a state of n-qubits\n
    Parameters:\n
        vertices: number of vertices of the graph
        edges: list of tuples corresponding to the edges of the graph\n
    Returns:\n
        a tensor that apply the problem hamiltonian to a n-qubits state\n
    Raise:\n
        ValueError if number of qubits is less than 2"""
    if n_qubits < 2:
        raise ValueError('number of qubits must be > 1, but is {}'.format(n_qubits))
    list_double_sigmaz = []
    for edge in edges:
        list_double_sigmaz.append(
            qucs.n_sigmaz(n_qubits,edge[0])*qucs.n_sigmaz(n_qubits,edge[1])
            )
    return 0.5*(len(edges)*qucs.n_qeye(n_qubits)-sum(list_double_sigmaz))


def evolution_operator(n_qubits, edges, gammas, betas):
    """
    This method generates a tensor that apply the evolution operator U of the
        MaxCut problem on a state of n-qubits\n
    Parameters:\n
        n_qubits: number of qubits is the n-qubits state among this operators acts on\n
        edges: edges of the graph of the MaxCut that define the problem hamiltonian
        gammas: gamma parameters of the MaxCut\n
        betas: betas parameters of the MaxCut\n
    Returns:\n
        a tensor that apply the evolution operator to a n-qubits state\n
    Raise:\n
        ValueError if number of gammas is less than 1\n
        ValueError if number of betas is less than 1\n
        ValueError if number of betas is different than number of gammas\n
        ValueError if number of qubits is less than 2      
    """
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
        u_mix_hamilt_i = (-complex(0,betas[i])*mix_hamilt(n_qubits)).expm()
        u_prob_hamilt_i = (-complex(0,gammas[i])*prob_hamilt(n_qubits, edges)).expm()
        evol_oper = u_mix_hamilt_i*u_prob_hamilt_i*evol_oper
    return evol_oper


def evaluate_F_p(params, n_qubits, edges, n_samples):
    """
    This method perform n quantum measurements on final state and evaluate F_p
    through classical mean of the outcomes

    Parameters
    ----------
    params : 1-D array-like
        array of parameters [gamma_1, gamma_2, ..., gamma_p, beta_1, beta_2, ..., beta_p].
    n_qubits : int
        number of qubits of the state.
    edges : list of tuples
        edges of the graph.
    n_samples : int
        number of quantum measurements performed.

    Returns
    -------
    F_p: float
        expectation value of the cost function

    """
    gammas = params[:int(len(list(params))/2)]
    betas = params[int(len(list(params))/2):]
    
    # initial state (as density matrix):
    #dm_init_state = qu.ket2dm(initial_state(n_qubits))
    init_state = initial_state(n_qubits)
    #obtain final state
    #dm_fin_state = evolution_operator(n_qubits, edges, gammas, betas)*dm_init_state*evolution_operator(n_qubits, edges, gammas, betas).dag()
    fin_state = (evolution_operator(n_qubits, edges, gammas, betas)*init_state)
    
    #Perform N measurements on each single qubit of final state
    c_outcomes = Counter(qucs.quantum_measurements(n_samples, fin_state))
    
    #Evaluate Fp
    list_z = list(c_outcomes.keys())
    list_w = list(c_outcomes.values())
    Fp = 0
    for i in range(len(c_outcomes)):
        Fp += list_w[i]*evaluate_cost_fun(list(list_z)[i], edges)
    return Fp/n_samples


def evaluate_F_p_j(params, n_qubits, index_j, edges, n_samples):
    gammas = params[:int(len(list(params))/2)]
    betas = params[int(len(list(params))/2):]
    edge = edges[index_j]
    
    # initial state (as density matrix):
    #dm_init_state = qu.ket2dm(initial_state(n_qubits))
    init_state = initial_state(n_qubits)
    #obtain final state
    #dm_fin_state = evolution_operator(n_qubits, edges, gammas, betas)*dm_init_state*evolution_operator(n_qubits, edges, gammas, betas).dag()
    fin_state = (evolution_operator(n_qubits, edges, gammas, betas)*init_state)
    
    #perform n_samples measurments on qubits in edge[0] and edge[1]
    outcomes = []
    for j in range(n_samples):
        outcome = ''
        qstate_dummy = fin_state.copy()
        for i in edge:
            outcome_i, qstate_dummy = qucs.single_qubit_measurement(qstate_dummy, i)
            outcome += outcome_i
        outcomes.append(outcome)
    c_outcomes = Counter(outcomes)
    
    #evaluate F_p_j
    list_z = list(c_outcomes.keys())
    list_w = list(c_outcomes.values())
    F_p_j = 0
    for i in range(len(list_w)):
        F_p_j += list_w[i]*single_term_cost_fun(list(list_z)[i])
    return F_p_j/n_samples

#Define a function that evaluate the gradient estimator g_t
def fin_diff_grad(function, params, args=(), increment=0.01):
    """
    This method estimates the gradient of a function through finite
    differences derivative estimation

    Parameters
    ----------
    function: function
        function of which the gradient has to be evaluated
    params : 1-D array like
        array of parameters of the function.
    args_fun : tuple
        optional arguments that the function may need. default = 0.01
    increment: float
        increment that define the finite differences method. default = 0.01

    Returns
    -------
    g_t: 1-D array
        array representing the gradient of the function in that parameters-space point

    """
    if not isinstance(args, tuple):
        args = (args,)
    d = len(list(params))
    a_params = np.array(params)
    g_t = np.zeros(d)
    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        g_t[i] = (function(a_params+e_i*increment, *args)-function(a_params-e_i*increment, *args))/(2*increment)
    return g_t


#Define a function that evaluate the gradient estimator g_t
def doubly_stoc_grad_max_cut(params, n_qubits, edges, n_samples):
    """
    This method estimates the gradient of a function through 
    doubly stochastic method

    Parameters
    ----------
    params : 1-D array like
        array of parameters of the function.


    Returns
    -------
    g_t: 1-D array
        array representing the gradient of the function in that parameters-space point

    """
    d = len(list(params))
    a_params = np.array(params)
    g_t = np.zeros(d)
    m = len(edges)
    for i in range(d):
        
        #obtain indecies for the samplings
        index_j = np.random.randint(m)
        #index_r not necessary (they behave in the same way)
        #index_j' not necessary (they behave in the same way)
        index_k = np.random.randint(2)
        
        #obtain forward parameters
        e_i = np.zeros(d)
        e_i[i] = 1.0
        if index_k == 0:
            forward_params = a_params + np.pi*0.5*e_i
            epsilon_i = 0.5
        else:
            forward_params = a_params - np.pi*0.5*e_i
            epsilon_i = -0.5
            
        #obtain gradient estimator
        g_t[i] = evaluate_F_p_j(forward_params, n_qubits ,index_j, edges, n_samples)*epsilon_i*2*m
    return g_t