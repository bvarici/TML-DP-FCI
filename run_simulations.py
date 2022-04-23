#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:28:55 2022

@author: Burak
"""

#from __future__ import print_function
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm                        
from indep_test import bincondKendall, discondKendall
from functions import PC, PrivPC, FCI, PrivFCI
from dataset import bn_data, ground_truth_fci_amat
import os

save_results_directory = os.getcwd()+'/results/estimated/'


def cal_stats(amat,amat_answer):
    skel = np.zeros(amat.shape)
    skel[np.where(amat)] = 1
    skel_answer = np.zeros(amat_answer.shape)
    skel_answer[np.where(amat_answer)] = 1
    adj_pairs = [(np.where(amat)[0][i],np.where(amat)[1][i]) for i in range(len(np.where(amat)[0])) if np.where(amat)[0][i]<np.where(amat)[1][i]]

    n_true_or = 0
    n_false_or = 0
    
    n_tp_skel = 0
    n_fp_skel = 0

    for (i,j) in adj_pairs:
        if skel[i,j] == skel_answer[i,j]:
            if skel[i,j] == 1:
                # i-j edge is correct in skeleton
                n_tp_skel += 1
                if amat[i,j] == amat_answer[i,j] and amat[j,i] == amat_answer[j,i]:
                    # i-j edge has correct orientation
                    n_true_or += 1
                else:
                    # i-j has incorrect orientation
                    n_false_or += 1
                    
        else:
            # skel[i,j]=1, but it is a false positive
            n_fp_skel += 1
            
    n_fn_skel = int(np.sum(skel_answer)/2 - n_tp_skel)
    
    if (n_tp_skel + n_fp_skel) > 0:
        skel_precision = n_tp_skel / (n_tp_skel + n_fp_skel)
    else:
        skel_precision = 0
        
    skel_recall = n_tp_skel / (n_tp_skel + n_fn_skel)
    if skel_precision + skel_recall > 0:
        skel_f1 = 2*skel_precision*skel_recall / (skel_precision+skel_recall)
    else:
        skel_f1 = 0
    
    if (n_tp_skel + n_fp_skel) > 0:
        or_precision = n_true_or / (n_tp_skel + n_fp_skel)
    else:
        or_precision = 9
    
    or_recall = n_true_or / (n_tp_skel + n_fn_skel)
    if or_precision+or_recall > 0:
        or_f1 = 2*or_precision*or_recall / (or_precision+or_recall)
    else:
        or_f1 = 0
    
    return (or_precision, or_recall, or_f1), (skel_precision, skel_recall, skel_f1)

def eval_PC(dataset='earthquake', iters=1, alpha=0.1):
    
    run_time = []
    or_f1_stat = []
    skel_f1_stat = []
    
    if dataset in ['asia', 'cancer', 'earthquake']:
        indep_test_func = bincondKendall
    elif dataset in ['survey']:
        indep_test_func = discondKendall
    
    dm = bn_data(dataset, size=100000)
    max_reach = max(min(np.int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)
    print("Run PC algorithms for %d times..."%iters)
    
    for _ in tqdm(range(iters)):
        start = time.time()
        (pag, sep_set) = PC(indep_test_func=indep_test_func,data_matrix=dm,alpha=alpha,\
                                                max_reach=max_reach)    

        end = time.time()

        node_labels = [node for node in pag]
        amat = pag.to_matrix()
        # re-order the nodes
        amat = amat[node_labels][:,node_labels]
        # load the ground truth
        amat_answer = ground_truth_fci_amat[dataset]
        (or_precision, or_recall, or_f1), (skel_precision, skel_recall, skel_f1) = cal_stats(amat, amat_answer)

        run_time.append(end-start)
        or_f1_stat.append(or_f1)
        skel_f1_stat.append(skel_f1)

    return run_time, or_f1_stat, skel_f1_stat

def eval_FCI(dataset='earthquake', iters=1, alpha=0.1):
    
    run_time = []
    or_f1_stat = []
    skel_f1_stat = []

    if dataset in ['asia', 'cancer', 'earthquake']:
        indep_test_func = bincondKendall
    elif dataset in ['survey']:
        indep_test_func = discondKendall
    
    dm = bn_data(dataset, size=100000)
    max_reach = max(min(np.int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)
    print("Run FCI algorithms for %d times..."%iters)

    for _ in tqdm(range(iters)):
        start = time.time()
        (pag, sep_set) = FCI(indep_test_func=indep_test_func,data_matrix=dm,alpha=alpha,\
                                                max_reach=max_reach)    

        end = time.time()

        node_labels = [node for node in pag]
        amat = pag.to_matrix()
        # re-order the nodes
        amat = amat[node_labels][:,node_labels]
        # load the ground truth
        amat_answer = ground_truth_fci_amat[dataset]
        (or_precision, or_recall, or_f1), (skel_precision, skel_recall, skel_f1) = cal_stats(amat, amat_answer)

        run_time.append(end-start)
        #privacy.append(eps)
        or_f1_stat.append(or_f1)
        skel_f1_stat.append(skel_f1)

    return run_time, or_f1_stat, skel_f1_stat

def eval_PrivPC(dataset='earthquake', iters=1, epsilon=1, delta=1e-3, alpha=0.1, q=0.05,bias=0.02):
    
    run_time = []
    or_f1_stat = []
    skel_f1_stat = []
    privacy = []
    
    if dataset in ['asia', 'cancer', 'earthquake']:
        indep_test_func = bincondKendall
    elif dataset in ['survey']:
        indep_test_func = discondKendall
    
    dm = bn_data(dataset, size=100000)
    max_reach = max(min(np.int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)
    print("Run Priv-PC algorithms for %d times..."%iters)
    
    for _ in tqdm(range(iters)):
        start = time.time()
        (pag, sep_set, eps) = PrivPC(indep_test_func=indep_test_func,data_matrix=dm,alpha=alpha,\
                                                max_reach=max_reach,q=q,eps=epsilon,delta=delta,bias=bias)    

        end = time.time()

        node_labels = [node for node in pag]
        amat = pag.to_matrix()
        # re-order the nodes
        amat = amat[node_labels][:,node_labels]
        # load the ground truth
        amat_answer = ground_truth_fci_amat[dataset]
        (or_precision, or_recall, or_f1), (skel_precision, skel_recall, skel_f1) = cal_stats(amat, amat_answer)

        run_time.append(end-start)
        privacy.append(eps)
        or_f1_stat.append(or_f1)
        skel_f1_stat.append(skel_f1)

    return run_time, or_f1_stat, skel_f1_stat, privacy

def eval_PrivFCI(dataset='earthquake', iters=1, epsilon=1, delta=1e-3, alpha=0.1, q=0.05,bias=0.02):
    
    run_time = []
    or_f1_stat = []
    skel_f1_stat = []
    privacy = []
    #pags = []
    
    if dataset in ['asia', 'cancer', 'earthquake']:
        indep_test_func = bincondKendall
    elif dataset in ['survey']:
        indep_test_func = discondKendall
    
    dm = bn_data(dataset, size=100000)
    max_reach = max(min(np.int(np.log2(dm.shape[0]))-5, dm.shape[1]-2), 0)
    print("Run Priv-FCI algorithms for %d times..."%iters)
    

    for _ in tqdm(range(iters)):
        start = time.time()
        (pag, sep_set, eps) = PrivFCI(indep_test_func=indep_test_func,data_matrix=dm,alpha=alpha,\
                                                max_reach=max_reach,q=q,eps=epsilon,delta=delta,bias=bias)    

        end = time.time()

        node_labels = [node for node in pag]
        amat = pag.to_matrix()
        # re-order the nodes
        amat = amat[node_labels][:,node_labels]
        # load the ground truth
        amat_answer = ground_truth_fci_amat[dataset]
        (or_precision, or_recall, or_f1), (skel_precision, skel_recall, skel_f1) = cal_stats(amat, amat_answer)

        run_time.append(end-start)
        privacy.append(eps)
        or_f1_stat.append(or_f1)
        skel_f1_stat.append(skel_f1)
        #pags.append(pag)
        #print(_,pag.edges)

    return run_time, or_f1_stat, skel_f1_stat, privacy

#python run_simulations.py --dataset cancer --iter 20 --delta 1e-3 --alpha 0.1 --epslogmin -1.5 --epslogmax 0.5 --epsnum 20
#python run_simulations.py --dataset asia --iter 20 --delta 1e-3 --alpha 0.1 --epslogmin -1.5 --epslogmax 0.5 --epsnum 20
#python run_simulations.py --dataset earthquake --iter 20 --delta 1e-3 --alpha 0.1 --epslogmin -1.5 --epslogmax 0.5 --epsnum 20


#%%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='cancer')
    parser.add_argument('--iter', type=int, default=20)
    parser.add_argument('--delta', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--q', type=float, default=0.1)
    parser.add_argument('--bias', type=float, default=0.02)
    parser.add_argument('--epslogmin', type=float, default=0)
    parser.add_argument('--epslogmax', type=float, default=0)
    parser.add_argument('--epsnum', type=int, default=1)

    
    args = parser.parse_args()
    
    datasets = ['asia', 'cancer', 'earthquake', 'survey']
    # range of epsilon values
    #epss = np.logspace(-1.5, 0.6, num=21, base=10)
    
    epss = np.logspace(args.epslogmin, args.epslogmax, num=args.epsnum, base=10)
    
    metrics = {'Priv-FCI': {'run_time':{}, 'or_f1':{}, 'skel_f1':{}, 'privacy':{}}, \
               'FCI':  {'run_time':{}, 'or_f1':{}, 'skel_f1':{}}}
    
    for i in metrics['Priv-FCI']:
        metrics['Priv-FCI'][i]['avg'] = {}
        metrics['Priv-FCI'][i]['std'] = {}
        for eps in epss:
            metrics['Priv-FCI'][i]['avg'][eps] = -1.0
            metrics['Priv-FCI'][i]['std'][eps] = -1.0 

    for i in metrics['FCI']:
        metrics['FCI'][i]['avg'] = -1.0
        metrics['FCI'][i]['std'] = -1.0

    count = 0
    if args.dataset in datasets:
        run_time_fci, or_f1_stat_fci, skel_f1_stat_fci = eval_FCI(args.dataset,args.iter,args.alpha)
        metrics['FCI']['run_time']['avg'] = np.mean(run_time_fci)
        metrics['FCI']['run_time']['std'] = np.std(run_time_fci)
        metrics['FCI']['or_f1']['avg'] = np.mean(or_f1_stat_fci)
        metrics['FCI']['or_f1']['std'] = np.std(or_f1_stat_fci)
        metrics['FCI']['skel_f1']['avg'] = np.mean(skel_f1_stat_fci)
        metrics['FCI']['skel_f1']['std'] = np.std(skel_f1_stat_fci)
        
        for eps in epss:
            count += 1
            print('-------------The eps number--------------')
            print(count, eps)
            run_time, or_f1_stat, skel_f1_stat, privacy = eval_PrivFCI(args.dataset, args.iter, epsilon=eps, delta=args.delta, alpha=args.alpha, q=args.q, bias=args.bias)
            
            metrics['Priv-FCI']['run_time']['avg'][eps] = np.mean(run_time)
            metrics['Priv-FCI']['run_time']['std'][eps] = np.std(run_time)      
            metrics['Priv-FCI']['or_f1']['avg'][eps] = np.mean(or_f1_stat)
            metrics['Priv-FCI']['or_f1']['std'][eps] = np.std(or_f1_stat)
            metrics['Priv-FCI']['skel_f1']['avg'][eps] = np.mean(skel_f1_stat)
            metrics['Priv-FCI']['skel_f1']['std'][eps] = np.std(skel_f1_stat)    
            metrics['Priv-FCI']['privacy']['avg'][eps] = np.mean(privacy)
            metrics['Priv-FCI']['privacy']['std'][eps] = np.std(privacy)
            

        f = open(save_results_directory+args.dataset+'.pkl','wb')
        pickle.dump(metrics, f)
        f.close()

    else:
        print('Invalid dataset')
