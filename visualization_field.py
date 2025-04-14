# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:13:49 2024

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def process_g_q(data):
    
    data['g1'] = data['mv1']/data['mv0']
    data['g2'] = data['mv2']/data['mv1']
    data['g3'] = data['mv3']/data['mv2']
    data['g4'] = data['mv4']/data['mv3']
    
    data['q1'] = data['ttc1']/data['ttc0']
    data['q2'] = data['ttc2']/data['ttc1']
    data['q3'] = data['ttc3']/data['ttc2']
    data['q4'] = data['ttc4']/data['ttc3']
    
    return data

def process_G_Q(data):
    
    data['G1'] = data['g1']
    data['G2'] = data['G1']*data['g2']
    data['G3'] = data['G2']*data['g3']
    data['G4'] = data['G3']*data['g4']
    
    data['Q1'] = data['q1']
    data['Q2'] = data['Q1']*data['q2']
    data['Q3'] = data['Q2']*data['q3']
    data['Q4'] = data['Q3']*data['q4']
    
    return data

def process_picud_gq(data):
    data['p1'] = data['picud1']  # fuzzy_0 = 1
    data['p2'] = data['picud2'] / data['picud1'].replace(0, np.nan)
    data['p3'] = data['picud3'] / data['picud2'].replace(0, np.nan)
    data['p4'] = data['picud4'] / data['picud3'].replace(0, np.nan)
    
    data['P1'] = data['p1']
    data['P2'] = data['P1'] * data['p2']
    data['P3'] = data['P2'] * data['p3']
    data['P4'] = data['P3'] * data['p4']
    
    return data

def process_drac_gq(data):
    data['d1'] = data['drac1']  # fuzzy_0 = 1
    data['d2'] = data['drac2'] / data['drac1'].replace(0, np.nan)
    data['d3'] = data['drac3'] / data['drac2'].replace(0, np.nan)
    data['d4'] = data['drac4'] / data['drac3'].replace(0, np.nan)
    
    data['D1'] = data['d1']
    data['D2'] = data['D1'] * data['d2']
    data['D3'] = data['D2'] * data['d3']
    data['D4'] = data['D3'] * data['d4']
    
    return data

def process_spearman(data):
    results_spearman_corr_gq = []
    results_spearman_p_gq = []    
    results_A = []
    
    results_spearman_corr_GQ = []
    results_spearman_p_GQ = []
    
    # picud
    results_spearman_corr_picud_single = []
    results_spearman_p_picud_single = []
    results_spearman_corr_picud_platoon = []
    results_spearman_p_picud_platoon = []

    # drac
    results_spearman_corr_drac_single = []
    results_spearman_p_drac_single = []
    results_spearman_corr_drac_platoon = []
    results_spearman_p_drac_platoon = []
    
    for group, temp in data.groupby('mv0'):
        # 原始单车稳定性与安全性分析
        g_values = temp[['mv1','mv2','mv3','mv4']].values.flatten()
        q_values = temp[['ttc1','ttc2','ttc3','ttc4']].values.flatten()
        spearman_corr_gq, p_value_gq = spearmanr(g_values, q_values)
        results_spearman_corr_gq.append(spearman_corr_gq)
        results_spearman_p_gq.append(p_value_gq)
        results_A.append(temp['mv0'].iloc[0])
        
        G_values = temp['G4'].values.flatten()
        Q_values = temp['Q4'].values.flatten()
        spearman_corr_GQ, p_value_GQ = spearmanr(G_values, Q_values)
        results_spearman_corr_GQ.append(spearman_corr_GQ)
        results_spearman_p_GQ.append(p_value_GQ)
        
        # picud single
        picud_single_values = temp[['picud1','picud2','picud3','picud4']].values.flatten()
        spearman_corr_picud_single, p_value_picud_single = spearmanr(picud_single_values, g_values)
        results_spearman_corr_picud_single.append(spearman_corr_picud_single)
        results_spearman_p_picud_single.append(p_value_picud_single)
        
        # picud platoon
        picud_platoon_values = temp['P4'].values.flatten()
        spearman_corr_picud_platoon, p_value_picud_platoon = spearmanr(picud_platoon_values, G_values)
        results_spearman_corr_picud_platoon.append(spearman_corr_picud_platoon)
        results_spearman_p_picud_platoon.append(p_value_picud_platoon)

        # drac single
        drac_single_values = temp[['drac1','drac2','drac3','drac4']].values.flatten()
        spearman_corr_drac_single, p_value_drac_single = spearmanr(drac_single_values, g_values)
        results_spearman_corr_drac_single.append(spearman_corr_drac_single)
        results_spearman_p_drac_single.append(p_value_drac_single)
        
        # drac platoon
        drac_platoon_values = temp['D4'].values.flatten()
        spearman_corr_drac_platoon, p_value_drac_platoon = spearmanr(drac_platoon_values, G_values)
        results_spearman_corr_drac_platoon.append(spearman_corr_drac_platoon)
        results_spearman_p_drac_platoon.append(p_value_drac_platoon)
            
    columns = ['A', 'spearman_corr_gq', 'spearman_p_gq', 
               'spearman_corr_GQ', 'spearman_p_GQ',
               'spearman_corr_picud_single', 'spearman_p_picud_single',
               'spearman_corr_picud_platoon', 'spearman_p_picud_platoon',
               'spearman_corr_drac_single', 'spearman_p_drac_single',
               'spearman_corr_drac_platoon', 'spearman_p_drac_platoon']
    results = pd.DataFrame({
        'A': results_A,
        'spearman_corr_gq': results_spearman_corr_gq,
        'spearman_p_gq': results_spearman_p_gq,
        'spearman_corr_GQ': results_spearman_corr_GQ,
        'spearman_p_GQ': results_spearman_p_GQ,
        'spearman_corr_picud_single': results_spearman_corr_picud_single,
        'spearman_p_picud_single': results_spearman_p_picud_single,
        'spearman_corr_picud_platoon': results_spearman_corr_picud_platoon,
        'spearman_p_picud_platoon': results_spearman_p_picud_platoon,
        'spearman_corr_drac_single': results_spearman_corr_drac_single,
        'spearman_p_drac_single': results_spearman_p_drac_single,
        'spearman_corr_drac_platoon': results_spearman_corr_drac_platoon,
        'spearman_p_drac_platoon': results_spearman_p_drac_platoon
    })
    
    return results

if __name__ == '__main__':
    
    ttc_u = 40
    file_name = 'TTC_u=' + str(ttc_u)
    
    usecols = ['file','mv0', 'mv1', 'mv2', 'mv3', 'mv4', 'ttc0', 'ttc1', 'ttc2', 'ttc3', 'ttc4', 'd-a', 'picud0', 'picud1', 'picud2', 'picud3', 'picud4', 'drac0', 'drac1', 'drac2', 'drac3', 'drac4']
    data = pd.read_excel('records_field.xlsx',sheet_name=file_name,header=0,usecols=usecols)
    data = data[data['d-a']=='d']
    data.dropna(axis=0,how='any',inplace=True)
    raw_len = len(data)
    data = data[(data['ttc1'] != ttc_u) & (data['ttc2'] != ttc_u) & (data['ttc3'] != ttc_u) &(data['ttc4'] != ttc_u)]
    after_len = len(data)
    
    data = process_g_q(data)
    data = process_G_Q(data)
    data = process_picud_gq(data)
    data = process_drac_gq(data)
    results = process_spearman(data)
    
    bwith = 2
    
    # G and T
    
    coefficients = np.polyfit(data['G4'], data['Q4'], 1)
    polynomial = np.poly1d(coefficients)
    trendline = polynomial(data['G4'])
    
    fig2, ax2 = plt.subplots(1,1,figsize=(20,10))
    ax2.scatter(data['G4'],data['Q4'],label='Measured values',color='blue')
    ax2.plot(data['G4'], trendline, linestyle=(0, (5, 10)), color='red', label='Trend line')
    ax2.set_xlabel('$G_4$', fontdict={'family':'Times New Roman', 'size':25,'weight':'bold'})
    ax2.set_ylabel('$SI_4^t$',fontdict={'family':'Times New Roman', 'size':30,'weight':'bold'})
    ax2.xaxis.set_tick_params(labelsize=25)
    ax2.yaxis.set_tick_params(labelsize=25)
    ax2.legend(fontsize=25)
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['bottom'].set_linewidth(bwith)
    ax2.spines['left'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    
    fig3, ax3 = plt.subplots(1,1,figsize=(20,10))
    ax3.hist(results['spearman_corr_gq'],bins=10)
    ax3.set_xlabel('$r_{s,2}^{t}$', fontdict={'family':'Times New Roman', 'size':25,'weight':'bold'})
    ax3.set_ylabel('Frequency',fontdict={'family':'Times New Roman', 'size':30,'weight':'bold'})
    ax3.xaxis.set_tick_params(labelsize=25)
    ax3.yaxis.set_tick_params(labelsize=25)
    ax3.spines['top'].set_linewidth(bwith)
    ax3.spines['bottom'].set_linewidth(bwith)
    ax3.spines['left'].set_linewidth(bwith)
    ax3.spines['right'].set_linewidth(bwith)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


    # picud

    coefficients = np.polyfit(data['G4'], data['P4'], 1)
    polynomial = np.poly1d(coefficients)
    trendline = polynomial(data['G4'])

    fig4, ax4 = plt.subplots(1,1,figsize=(40,20))
    ax4.scatter(data['G4'],data['P4'],label='Measured values',color='blue')
    ax4.plot(data['G4'], trendline, linestyle=(0, (5, 10)), color='red', label='Trend line')
    ax4.set_xlabel('$G_4$', fontdict={'family':'Times New Roman', 'size':25,'weight':'bold'})
    ax4.set_ylabel('$SI_4^p$',fontdict={'family':'Times New Roman', 'size':30,'weight':'bold'})
    ax4.xaxis.set_tick_params(labelsize=25)
    ax4.yaxis.set_tick_params(labelsize=25)
    ax4.legend(fontsize=25)
    ax4.spines['top'].set_linewidth(bwith)
    ax4.spines['bottom'].set_linewidth(bwith)
    ax4.spines['left'].set_linewidth(bwith)
    ax4.spines['right'].set_linewidth(bwith)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig5, ax5 = plt.subplots(1,1,figsize=(40,20))
    ax5.hist(results['spearman_corr_picud_single'],bins=10)
    ax5.set_xlabel('$r_{s,2}^p$', fontdict={'family':'Times New Roman', 'size':25,'weight':'bold'})
    ax5.set_ylabel('Frequency',fontdict={'family':'Times New Roman', 'size':30,'weight':'bold'})
    ax5.xaxis.set_tick_params(labelsize=25)
    ax5.yaxis.set_tick_params(labelsize=25)
    ax5.spines['top'].set_linewidth(bwith)
    ax5.spines['bottom'].set_linewidth(bwith)
    ax5.spines['left'].set_linewidth(bwith)
    ax5.spines['right'].set_linewidth(bwith)
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # drac

    coefficients = np.polyfit(data['G4'], data['D4'], 1)
    polynomial = np.poly1d(coefficients)
    trendline = polynomial(data['G4'])

    fig6, ax6 = plt.subplots(1,1,figsize=(40,20))
    ax6.scatter(data['G4'],data['D4'],label='Measured values',color='blue')
    ax6.plot(data['G4'], trendline, linestyle=(0, (5, 10)), color='red', label='Trend line')
    ax6.set_xlabel('$G_4$', fontdict={'family':'Times New Roman', 'size':25,'weight':'bold'})
    ax6.set_ylabel('$SI_4^d$',fontdict={'family':'Times New Roman', 'size':30,'weight':'bold'})
    ax6.xaxis.set_tick_params(labelsize=25)
    ax6.yaxis.set_tick_params(labelsize=25)
    ax6.legend(fontsize=25)
    ax6.spines['top'].set_linewidth(bwith)
    ax6.spines['bottom'].set_linewidth(bwith)
    ax6.spines['left'].set_linewidth(bwith)
    ax6.spines['right'].set_linewidth(bwith)
    ax6.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig7, ax7 = plt.subplots(1,1,figsize=(40,20))
    ax7.hist(results['spearman_corr_drac_single'],bins=10)
    ax7.set_xlabel('$r_{s,2}^d$', fontdict={'family':'Times New Roman', 'size':25,'weight':'bold'})
    ax7.set_ylabel('Frequency',fontdict={'family':'Times New Roman', 'size':30,'weight':'bold'})
    ax7.xaxis.set_tick_params(labelsize=25)
    ax7.yaxis.set_tick_params(labelsize=25)
    ax7.spines['top'].set_linewidth(bwith)
    ax7.spines['bottom'].set_linewidth(bwith)
    ax7.spines['left'].set_linewidth(bwith)
    ax7.spines['right'].set_linewidth(bwith)
    ax7.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax7.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    
    plt.show()
    print('usage:', after_len*100/raw_len)
    
    
    
    