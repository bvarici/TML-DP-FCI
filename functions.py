#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:31:54 2022

@author: Burak
"""
'''

Parts of the following code for FCI implementation is imported from 
https://github.com/jc1850/Learning-Causal-Networks-in-Python
and modified accordingly.

Parts of the following code for sieve-and-examine (probe-examine in original code) mechanism
is imported from the official implementation of Priv-PC paper
(https://github.com/sunblaze-ucb/Priv-PC-Differentially-Private-Causal-Graph-Discovery)


'''

from itertools import combinations, permutations
import numpy as np
import networkx as nx
from scipy.integrate import quad
from scipy.optimize import minimize
import copy
from graphs import PAG

def _create_complete_graph(node_ids):
    """Create a complete graph from the list of node ids.
    Args:
        node_ids: a list of node ids
    Returns:
        An undirected graph (as a networkx.Graph)
    """
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for (i, j) in combinations(node_ids, 2):
        g.add_edge(i, j)
    return g

def is_possible_d_sep(X,Y,pag):

    all_paths = nx.all_simple_paths(pag,X,Y)
    for path in all_paths:
        path_sep = True
        for i in range(1,len(path[:-1])):
            collider = (pag.has_directed_edge(path[i-1],path[i]) and pag.has_directed_edge(path[i+1],path[i]))
            triangle = (pag.has_edge(path[i-1],path[i]) and pag.has_edge(path[i+1],path[i]) and pag.has_edge(path[i-1],path[i+1]))
            if not(collider or triangle):
                    path_sep = False
        if path_sep:
            return True
        return False

def possible_d_seps(pag):
    """
    Method to construct the possible d-sep-set of a pag
    
    Parameters
    ----------
        pag: PAG
            PAG to find dseps for
    Returns
    -------
        dict
            keys: nodes
            values: nodes which could d seperate other nodes from the key node

    """
    dseps = {}
    for i in pag:
        dseps[i] = []
        for j in pag:
            if i != j:
                if is_possible_d_sep(i,j,pag):
                    dseps[i].append(j)
    return dseps

def orient_V(pag, sepSet):
    """
    A function to orient the colliders in a PAG
        
    Parameters
    ----------
        pag: PAG
            PAG to be oriented
        sepSet: dict
            separation d#sets of all pairs of nodes in PAG
    Returns
    -------
        PAG
            PAG with v-structures oriented
    """
    for i in pag:
        for j in pag:
            if j != i:
                for k in pag:
                    if k not in [i,j]:
                        if pag.has_edge(i,j) and pag.has_edge(k,j) and not pag.has_edge(i,k):
                            if j not in sepSet[(i,k)]:
                                pag.direct_edge(i,j)
                                pag.direct_edge(k,j)
                                #print('Orienting collider {}, {}, {}'.format(i,j,k))
    
def estimate_first_skeleton(indep_test_func, data_matrix, alpha, max_reach):
    '''
    returns the skeleton, same as first step PC. without privacy
    '''

    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    #sep_set = [[set() for i in range(node_size)] for j in range(node_size)]

    sep_set = {}

    g = _create_complete_graph(node_ids)

    l = 0

    while True:
        cont = False
        
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)

            if len(adj_i) < l:
                continue

            for k in combinations(adj_i, l):
                p_val = indep_test_func(data_matrix, i, j, set(k))[1]

                if p_val < alpha:
                    continue
                if p_val >= alpha:
                    if g.has_edge(i, j):
                        g.remove_edge(i, j)

                    if (i,j) in sep_set.keys():
                        sep_set[(i,j)] |= set(k)
                        sep_set[(j,i)] |= set(k)
                    else:
                        sep_set[(i,j)] = set(k)
                        sep_set[(j,i)] = set(k)
                    break

            cont = True
        l += 1
        if cont is False:
            break
        if l > max_reach :
            break

    return (g, sep_set)

def estimate_second_skeleton(pag,sep_set,p_d_seps,indep_test_func, data_matrix, alpha, max_reach):
    l = 0

    for i in pag:
        adj_i = list(pag.neighbors(i))
        for j in adj_i:
            # (i,j) is an adjacent pair in pag.
            for l in range(1,min(max_reach,len(p_d_seps[i]))+1):
                # check size l subsets of p-d-seps(i)
                for subset_k in combinations(p_d_seps[i],l):
                    subset_k = set(subset_k)
                    if j in subset_k:
                        subset_k.remove(j)

                    p_val = indep_test_func(data_matrix, i, j, subset_k)[1] 

                    if p_val < alpha:
                        continue
                    if p_val >= alpha:
                        if pag.has_edge(i,j):
                            pag.remove_edge(i,j)
                            pag.remove_edge(j,i) # BV update
                            
                            
                        if (i,j) in sep_set.keys():    
                            sep_set[(i,j)] |= set(subset_k)
                            sep_set[(j,i)] |= set(subset_k)
                        else:
                            sep_set[(i,j)] = set(subset_k)
                            sep_set[(j,i)] = set(subset_k)                            
                        break

    return (pag, sep_set)


def estimate_first_skeleton_probe_examine(indep_test_func, data_matrix, alpha, max_reach, q=0.05, eps=1, delta=1e-3, bias=0.02):
    '''
    returns the skeleton, with privacy, same as first step of Priv-PC
    '''

    node_ids = range(data_matrix.shape[1])
    n = data_matrix.shape[0]
    node_size = data_matrix.shape[1]
    #sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    sep_set = {}

    g = _create_complete_graph(node_ids)

    l = 0
    count = 0
    budget_split = 1.0 / 2.0
    eps1 = eps * budget_split
    def noise_scale(x):
        return np.sqrt(x) / np.log(x * (np.exp(eps1)-1) + 1)


    if not isinstance(q,float):
        # if the subsampling rate is not provided, optimize, and clip into [1/20,1] range.
        q = max(min(1. / minimize(noise_scale, [0.5], tol=1e-2).x[0], 1), 1. / 20.)
    
    eps2 = eps - eps1
    # I don't know why Delta is computed this way.
    S, _ = quad(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 0, 6 / np.sqrt(n))
    sigma1 = 2.0 * S / np.sqrt(q) / np.log((np.exp(eps1)-1.)/q + 1)
    sigma2 = 2 * sigma1
    sigma3 = S / eps2
    # tweak the threshold by bias: T0 is \hat T in the paper, and alpha is T in the paper.
    T0 = alpha - bias + np.random.laplace(0, sigma1)
    row_rand = np.arange(n)
    np.random.shuffle(row_rand)
    dm_subsampled = data_matrix[row_rand[0:int(n*q)]]
    while True:
        cont = False
        
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)

            if len(adj_i) < l:
                continue

            for k in combinations(adj_i, l):
                v = np.random.laplace(0, sigma2)
                p_val = indep_test_func(dm_subsampled, i, j, set(k))[1] + v

                if p_val < T0:
                    continue
                if p_val >= T0:
                    count += 1
                    T0 = alpha - bias + np.random.laplace(0, sigma1)
                    np.random.shuffle(row_rand)
                    dm_subsampled = data_matrix[row_rand[0:int(n*q)]]
                    v = np.random.laplace(0, sigma3)
                    p_val = indep_test_func(data_matrix, i, j, set(k))[1] + v
                    
                    if p_val >= alpha:
                        if g.has_edge(i, j):
                            g.remove_edge(i, j)

                        if (i,j) in sep_set.keys():
                            sep_set[(i,j)] |= set(k)
                            sep_set[(j,i)] |= set(k)
                        else:
                            sep_set[(i,j)] = set(k)
                            sep_set[(j,i)] = set(k)                            
                        break
            cont = True
        l += 1
        if cont is False:
            break
        if l > max_reach :
            break

    eps_prime1 = np.sqrt(2*count*np.log(2/delta))*eps2 + count*eps2*(np.exp(eps2)-1)
    eps_prime2 = np.sqrt(2*count*np.log(2/delta))*eps1 + count*eps1*(np.exp(eps1)-1)
    eps_prime = eps_prime1 + eps_prime2

    return (g, sep_set, eps_prime, count)


def estimate_second_skeleton_probe_examine(pag,sep_set,p_d_seps,indep_test_func, data_matrix, alpha, max_reach, q=0.05, eps=1, delta=1e-3, bias=0.02):
    n = data_matrix.shape[0]
    l = 0
    count = 0
    budget_split = 1.0 / 2.0
    eps1 = eps * budget_split
    def noise_scale(x):
        return np.sqrt(x) / np.log(x * (np.exp(eps1)-1) + 1)

    if not isinstance(q,float):
        # if the subsampling rate is not provided, optimize, and clip into [1/20,1] range.
        q = max(min(1. / minimize(noise_scale, [0.5], tol=1e-2).x[0], 1), 1. / 20.)
    
    eps2 = eps - eps1
    # I don't know why Delta is computed this way.
    S, _ = quad(lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 0, 6 / np.sqrt(n))
    sigma1 = 2.0 * S / np.sqrt(q) / np.log((np.exp(eps1)-1.)/q + 1)
    sigma2 = 2 * sigma1
    sigma3 = S / eps2
    # tweak the threshold by bias: T0 is \hat T in the paper, and alpha is T in the paper.
    T0 = alpha - bias + np.random.laplace(0, sigma1)
    
    row_rand = np.arange(n)
    np.random.shuffle(row_rand)
    dm_subsampled = data_matrix[row_rand[0:int(n*q)]]


    for i in pag:
        adj_i = list(pag.neighbors(i))
        for j in adj_i:
            # (i,j) is an adjacent pair in pag.
            for l in range(1,min(max_reach,len(p_d_seps[i]))+1):
                # check size l subsets of p-d-seps(i)
                for subset_k in combinations(p_d_seps[i],l):
                    subset_k = set(subset_k)
                    if j in subset_k:
                        subset_k.remove(j)

                    v = np.random.laplace(0,sigma2)
                    p_val = indep_test_func(dm_subsampled, i, j, subset_k)[1] + v

                    if p_val < T0:
                        continue
                    if p_val >= T0:
                        count += 1
                        T0 = alpha - bias + np.random.laplace(0, sigma1)
                        np.random.shuffle(row_rand)
                        dm_subsampled = data_matrix[row_rand[0:int(n*q)]]
                        v = np.random.laplace(0, sigma3)
                        p_val = indep_test_func(data_matrix, i, j, subset_k)[1] + v
                        if p_val >= alpha:
                            if pag.has_edge(i,j):
                                pag.remove_edge(i,j)
                                pag.remove_edge(j,i) # BV update
                                
                            if (i,j) in sep_set.keys():
                                sep_set[(i,j)] |= set(subset_k)
                                sep_set[(j,i)] |= set(subset_k)
                            else:
                                sep_set[(i,j)] = set(subset_k)
                                sep_set[(j,i)] = set(subset_k)                                
                            break


    eps_prime1 = np.sqrt(2*count*np.log(2/delta))*eps2 + count*eps2*(np.exp(eps2)-1)
    eps_prime2 = np.sqrt(2*count*np.log(2/delta))*eps1 + count*eps1*(np.exp(eps1)-1)
    eps_prime = eps_prime1 + eps_prime2

    return (pag, sep_set, eps_prime, count)


'ORIENTATION RULES FOR FCI ALGORITHM, NOTHING TO DO WITH PRIVACY'

def rule1(pag,i,j,k):
    if pag.has_directed_edge(i,j) and pag.has_o(j,k,j) and not pag.has_edge(i,k):
        pag.fully_direct_edge(j,k)
        #print('Orienting edge {},{} with rule 1'.format(j,k))
    

def rule2(pag,i,j,k):
    chain1 = pag.has_fully_directed_edge(i,j) and pag.has_directed_edge(j,k)
    chain2 = pag.has_fully_directed_edge(j,k) and pag.has_directed_edge(i,j)
    if (chain1 or chain2) and pag.has_o(i,k,k):
        pag.direct_edge(i,k)
        #print('Orienting edge {},{} with rule 2'.format(i,k))
        

def rule3(pag,i,j,k,l):
    chain1 = (pag.has_directed_edge(i,j)) and pag.has_directed_edge(k,j)
    chain2 = (pag.has_o(i,l,l)) and (pag.has_o(k,l,l))
    if chain1 and chain2 and not pag.has_edge(i,k) and pag.has_o(l,j,j):
        pag.direct_edge(l,j)
        #print('Orienting edge {},{} with rule 3'.format(l,j))


def rule4(pag,i,j,k,l, sepSet):
    paths = pag.findDiscPath(l,k,j)
    for path in paths:
        if i in path:
            if path.index(i) == len(path)-3 and  pag.has_o(j,k,j) :
                if j in sepSet[(l,k)]:
                    pag.fully_direct_edge(j,k)
                    #print('Orienting edge {},{} with rule 4'.format(j,k))

                else:
                    pag.direct_edge(i,j)
                    pag.direct_edge(j,k)
                    pag.direct_edge(j,i)
                    pag.direct_edge(k,j)    
                    #print('Orienting edges {},{}, {},{} with rule 4'.format(i,j,j,k))



def rule8(pag,i,j,k):
    chain1 = pag.has_fully_directed_edge(i,j) and pag.has_fully_directed_edge(j,k)#
    chain2 = False
    edge = False
    if pag.has_edge(i,j) and pag.has_edge(i,k):
        chain2 = pag.has_directed_edge(j,k) and pag.get_edge_data(i,j)[j] == 'o' and pag.get_edge_data(i,j)[i] == '-'
        edge =  pag.get_edge_data(i,k)[i] == 'o' and pag.has_directed_edge(i,k)
    if (chain1 or chain2) and edge:
        pag.fully_direct_edge(i,k)
        #print('Orienting edge {},{} with rule 8'.format(k,i))
    
def rule9(pag,i,j,k,l):
    if pag.has_directed_edge(i,k) and pag.has_o(i,k,i):
        for path in nx.all_simple_paths(pag,i,k):
            if pag.isUncovered(path) and pag.isPD(path):
                if path[1] == j and path[2] == l and not pag.has_edge(j,k):
                    pag.fully_direct_edge(i,k)
                    #print('Orienting edge {},{} with rule 9'.format(k,i))
                    break

def rule10(pag,i,j,k,l):
    if pag.has_directed_edge(i,k) and pag.has_o(i,k,i):
        if pag.has_fully_directed_edge(j,k) and pag.has_fully_directed_edge(l,k):
            for path1 in nx.all_simple_paths(pag,i,j):
                for path2 in nx.all_simple_paths(pag,i,l):
                    if pag.isUncovered(path1) and pag.isPD(path1) and pag.isUncovered(path2) and pag.isPD(path2):
                        if path1[1] != path2[1] and not pag.has_edge(path1[1],path2[1]):
                            pag.fully_direct_edge(i,k) 
                            #print('Orienting edge {},{} with rule 10'.format(k,i)) 


def apply_fci_orientation_rules(pag,sep_set):

    old_pag = nx.DiGraph()
    while old_pag.edges != pag.edges:
        old_pag = copy.deepcopy(pag)
        for i in pag:
            for j in pag:
                for k in pag:
                    if k not in [i,j] and j != i:
                        rule1(pag,i,j,k)
                        rule2(pag,i,j,k)
                        rule8(pag,i,j,k)
                        for l in pag:
                            if l not in [i,j,k]:
                                rule3(pag,i,j,k,l)
                                rule4(pag,i,j,k,l,sep_set)                        
                                rule9(pag,i,j,k,l)
                                rule10(pag,i,j,k,l)
                                
    return pag

def PC(indep_test_func, data_matrix, alpha, max_reach):
    # estimate first skeleton
    print('PC: estimating the skeleton..')
    (g, sep_set) = estimate_first_skeleton(indep_test_func, data_matrix, alpha, max_reach)
    # construct pag with all edges are o-o
    g = g.to_directed()
    pag = PAG()
    pag.add_nodes_from(g.nodes)
    pag.add_edges_from(g.edges)
    # orient V-structures in the first-skeleton
    print('PC: orienting V-structures on the skeleton..')
    orient_V(pag,sep_set)

    return (pag, sep_set)

def FCI(indep_test_func, data_matrix, alpha, max_reach):
    # estimate first skeleton
    print('FCI: estimating the first skeleton..')
    (g, sep_set) = estimate_first_skeleton(indep_test_func, data_matrix, alpha, max_reach)
    # construct pag with all edges are o-o
    g = g.to_directed()
    pag = PAG()
    pag.add_nodes_from(g.nodes)
    pag.add_edges_from(g.edges)
    # orient V-structures in the first-skeleton
    print('FCI: orienting V-structures on the first skeleton..')
    orient_V(pag,sep_set)
    # compute p_d_seps
    p_d_seps = possible_d_seps(pag)
    # estimate second skeleton
    print('FCI: estimating the second skeleton..')
    (pag, sep_set) = estimate_second_skeleton(pag,sep_set,p_d_seps,indep_test_func,data_matrix,alpha, max_reach)


    new_pag = PAG()
    new_pag.add_nodes_from(pag)
    new_pag.add_edges_from(pag.edges)
    pag = new_pag
    del new_pag
    # orient V-structures in the second-skeleton
    print('FCI: orienting V-structures on the second skeleton..')
    orient_V(pag,sep_set)
    # finally, apply FCI orientation rules. since there is no selection bias, apply R1-R4 and R8-R10 (no R5-R7)
    print('FCI: applying FCI orientation rules..')
    pag =  apply_fci_orientation_rules(pag, sep_set)
    print('FCI execution is finished')
    return (pag, sep_set)


#%%
def PrivPC(indep_test_func, data_matrix, alpha, max_reach, q=0.05, eps=1, delta=1e-3, bias=0.02):
    # estimate first skeleton
    print('Priv-PC: estimating the skeleton..')
    (g, sep_set, eps_prime_step1, count_step1) = estimate_first_skeleton_probe_examine(indep_test_func, data_matrix, alpha, max_reach, q, eps/2, delta, bias)
    # construct pag with all edges are o-o
    g = g.to_directed()
    pag = PAG()
    pag.add_nodes_from(g.nodes)
    pag.add_edges_from(g.edges)
    # orient V-structures in the first-skeleton
    print('Priv-PC: orienting V-structures on the skeleton..')
    orient_V(pag,sep_set)
    
    count = count_step1
    eps_prime = 2 * (np.sqrt(2*count*np.log(2/delta))*(eps/4) + count*(eps/4)*(np.exp(eps/4)-1))   

    print('Priv-FCI execution is finished, total privacy cost is {}'.format(eps_prime))
    return (pag, sep_set, eps_prime)


def PrivFCI(indep_test_func, data_matrix, alpha, max_reach, q=0.05, eps=1, delta=1e-3, bias=0.02):
    # estimate first skeleton
    print('Priv-FCI: estimating the first skeleton..')
    (g, sep_set, eps_prime_step1, count_step1) = estimate_first_skeleton_probe_examine(indep_test_func, data_matrix, alpha, max_reach, q, eps/2, delta, bias)
    # construct pag with all edges are o-o
    g = g.to_directed()
    pag = PAG()
    pag.add_nodes_from(g.nodes)
    pag.add_edges_from(g.edges)
    # orient V-structures in the first-skeleton
    print('Priv-FCI: orienting V-structures on the first skeleton..')
    orient_V(pag,sep_set)
    # compute p_d_seps
    p_d_seps = possible_d_seps(pag)
    # estimate second skeleton
    print('Priv-FCI: estimating the second skeleton..')
    (pag, sep_set, eps_prime_step2, count_step2) = estimate_second_skeleton_probe_examine(pag,sep_set,p_d_seps,indep_test_func,data_matrix,alpha, max_reach, q, eps/2, delta, bias)

    #eps_prime = eps_prime_step1 + eps_prime_step2
    count = count_step1 + count_step2
    eps_prime = 2 * (np.sqrt(2*count*np.log(2/delta))*(eps/4) + count*(eps/4)*(np.exp(eps/4)-1))

    new_pag = PAG()
    new_pag.add_nodes_from(pag)
    new_pag.add_edges_from(pag.edges)
    pag = new_pag
    del new_pag
    # orient V-structures in the second-skeleton
    print('Priv-FCI: orienting V-structures on the second skeleton..')
    orient_V(pag,sep_set)
    # finally, apply FCI orientation rules. since there is no selection bias, apply R1-R4 and R8-R10 (no R5-R7)
    print('Priv-FCI: applying FCI orientation rules..')
    pag =  apply_fci_orientation_rules(pag, sep_set)
    print('Priv-FCI execution is finished, total privacy cost is {}'.format(eps_prime))
    return (pag, sep_set, eps_prime)

