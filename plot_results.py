#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 00:10:33 2022

@author: Burak
"""

import os 
import numpy as np 
import pickle as pkl
import matplotlib.pyplot as plt

xticks_size = 14
yticks_size = 14
xlabel_size = 14
ylabel_size = 14
legend_size = 12
legend_loc = 'upper left'
linewidth = 2
linestyle = '--'
markersize = 8

save_results_directory = os.getcwd()+'/results/estimated/'
save_figures_directory = os.getcwd()+'/results/figures/'

def load_res(dataset='asia'):
    res = pkl.load(open(save_results_directory+dataset+'.pkl', 'rb'))  
    return res

def plot_res(dataset='asia',savefig=True):
    res = pkl.load(open(save_results_directory+dataset+'.pkl', 'rb'))  
    epss = list(res['Priv-FCI']['run_time']['avg'].keys())
    privFCI_or_F1_avg = res['Priv-FCI']['or_f1']['avg'].values()
    privFCI_or_F1_std = res['Priv-FCI']['or_f1']['std'].values()
    privFCI_skel_F1_avg = res['Priv-FCI']['skel_f1']['avg'].values()
    privFCI_skel_F1_std = res['Priv-FCI']['skel_f1']['std'].values()
    privFCI_privacy_avg = res['Priv-FCI']['privacy']['avg'].values()
    privFCI_privacy_std = res['Priv-FCI']['privacy']['std'].values()

    FCI_or_F1_avg = [res['FCI']['or_f1']['avg'] for _ in range(len(epss))]
    FCI_or_F1_std = [res['FCI']['or_f1']['std'] for _ in range(len(epss))]
    FCI_skel_F1_avg = [res['FCI']['skel_f1']['avg'] for _ in range(len(epss))]
    FCI_skel_F1_std = [res['FCI']['skel_f1']['std'] for _ in range(len(epss))]
     
    plt.figure()
    ax = plt.subplot()
    ax.set_xscale("log", nonposx='clip')
    #ax.set_ylim([0,1])
    ax.errorbar(privFCI_privacy_avg, privFCI_or_F1_avg, yerr=privFCI_or_F1_std, label='Priv-FCI-orientation',\
                capsize=5,capthick=2,linewidth=linewidth,linestyle='dashed', marker='o',markersize=markersize)

    ax.errorbar(privFCI_privacy_avg, privFCI_skel_F1_avg, yerr=privFCI_skel_F1_std, label='Priv-FCI-skeleton',\
                capsize=5,capthick=2,linewidth=linewidth,linestyle='dashed',marker='o',markersize=markersize)
     
    ax.errorbar(privFCI_privacy_avg, FCI_or_F1_avg, yerr=FCI_or_F1_std, label='FCI-orientation',\
                capsize=5,capthick=1,linewidth=3)

    ax.errorbar(privFCI_privacy_avg, FCI_skel_F1_avg, yerr=FCI_skel_F1_std, label='FCI-skeleton',\
                capsize=5,capthick=2,linewidth=3)
        
    plt.title('{} dataset'.format(dataset),fontsize=18)
    plt.xlabel('Total privacy cost $\epsilon$', size=xlabel_size)
    plt.ylabel('F1 score for edge recovery', size=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.legend()
    plt.grid(True)

    if savefig is True:
        plt.savefig(save_figures_directory+dataset+'.eps')
        
    plt.figure()
    ax = plt.subplot()
    ax.set_xscale("log", nonposx='clip')
    ax.plot(privFCI_privacy_avg, privFCI_or_F1_avg, label='Priv-FCI-orientation',\
                linewidth=linewidth,linestyle='dashed', marker='o',markersize=markersize)

    ax.plot(privFCI_privacy_avg, privFCI_skel_F1_avg, label='Priv-FCI-skeleton',\
                linewidth=linewidth,linestyle='dashed',marker='o',markersize=markersize)
     
    ax.plot(privFCI_privacy_avg, FCI_or_F1_avg, label='FCI-orientation',\
                linewidth=3)

    ax.plot(privFCI_privacy_avg, FCI_skel_F1_avg, label='FCI-skeleton',\
                linewidth=3)
        
    plt.title('{} dataset'.format(dataset),fontsize=18)
    plt.xlabel('Total privacy cost $\epsilon$', size=xlabel_size)
    plt.ylabel('F1 score for edge recovery', size=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.legend()
    plt.grid(True)

    if savefig is True:
        plt.savefig(save_figures_directory+dataset+'_simple.eps')        
        
#%%
#simply run plot_res function with your choice of dataset to plot the saved results

#plot_res('asia')
#plot_res('earthquake')
#plot_res('cancer')

