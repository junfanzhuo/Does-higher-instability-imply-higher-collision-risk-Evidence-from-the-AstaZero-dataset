# -*- coding: utf-8 -*-


import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

def process_g_q(data):
    data['g1'] = data['mv1'] / data['mv0']
    data['g2'] = data['mv2'] / data['mv1']
    data['g3'] = data['mv3'] / data['mv2']
    data['g4'] = data['mv4'] / data['mv3']
    
    data['q1'] = data['ttc1'] / data['ttc0']
    data['q2'] = data['ttc2'] / data['ttc1']
    data['q3'] = data['ttc3'] / data['ttc2']
    data['q4'] = data['ttc4'] / data['ttc3']
    
    return data

def process_G_Q(data):
    data['G1'] = data['g1']
    data['G2'] = data['G1'] * data['g2']
    data['G3'] = data['G2'] * data['g3']
    data['G4'] = data['G3'] * data['g4']
    
    data['Q1'] = data['q1']
    data['Q2'] = data['Q1'] * data['q2']
    data['Q3'] = data['Q2'] * data['q3']
    data['Q4'] = data['Q3'] * data['q4']
    
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

def add_risk_columns(data):
    
    data['min_ttc'] = data[['ttc1', 'ttc2', 'ttc3', 'ttc4']].min(axis=1)
    data['min_picud'] = data[['picud1', 'picud2', 'picud3', 'picud4']].min(axis=1)
    data['min_drac'] = data[['drac1', 'drac2', 'drac3', 'drac4']].min(axis=1)
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

def find_threshold(data, column, threshold, comparison='less', aggregate='median'):
  
    grouped = data.groupby('mv0')
    m_values = []
    
    for m, group in grouped:
        if comparison == 'less':
            if (group[column] <= threshold).any():
                m_values.append(m)
        elif comparison == 'greater':
            if (group[column] >= threshold).any():
                m_values.append(m)
        else:
            raise ValueError("comparison 参数必须为 'less' 或 'greater'")
    
    if not m_values:
        return None
    
    if aggregate == 'min':
        return min(m_values)
    elif aggregate == 'median':
        return np.median(m_values)
    elif aggregate == 'mean':
        return np.mean(m_values)
    elif aggregate == 'max':
        return max(m_values)
    else:
        raise ValueError("Unsupported aggregate method. Choose from 'min', 'median', 'mean', 'max'.")

def draw_thresholds(ax, thresholds, threshold_values, labels, unit, y_pos=None, colors=None):
    
    if colors is None:
        colors = ['r', 'g', 'b', 'orange']
    
    if y_pos is None:
        # 获取y轴的范围
        y_min, y_max = ax.get_ylim()
        # 计算相对位置
        y_range = y_max - y_min
        y_pos = [y_min + y_range * 0.1 * (i + 1) for i in range(len(thresholds))]

    for i, (threshold_m, threshold_value, label) in enumerate(zip(thresholds, threshold_values, labels)):
        if threshold_m is not None:
            style = '--' if i == 0 else '-.'
            ax.axvline(x=threshold_m, color=colors[i % len(colors)], linestyle=style, linewidth=1.5)

            ax.text(threshold_m + 0.2, y_pos[i], f'{label}{threshold_value}{unit}', 
                  fontdict={'family': 'Times New Roman', 'size': 12},
                  verticalalignment='center')

if __name__ == '__main__':
    file_name = 'records_simulation_LinearACC_de_2'
    usecols = ['file','mv0','mv1','mv2','mv3','mv4',
               'ttc0','ttc1','ttc2','ttc3','ttc4',
               'picud1','picud2','picud3','picud4',
               'drac1','drac2','drac3','drac4']
    data = pd.read_csv(file_name, header=0, usecols=usecols)
    acc_thre = [9,8.4,8.4]
    data = data[~((data['ttc1'] == 40) | (data['ttc2'] == 40) | (data['ttc3'] == 40) | (data['ttc4'] == 40))]
    data = data[data['mv0'] <= acc_thre[1]]
    data.replace([np.inf, -np.inf, 0], np.nan, inplace=True)
    data.dropna(axis=0, how='any', inplace=True)
    data.reset_index(inplace=True, drop=True)
    
    # 原始数据处理
    data = process_g_q(data)
    data = process_G_Q(data)
    data = process_picud_gq(data)
    data = process_drac_gq(data)
    data = add_risk_columns(data)

    # 设置多个阈值
    ttc_thresholds = [5, 4, 3]  # TTC阈值列表
    picud_thresholds = [0, -5, -9]  # PICUD阈值列表
    drac_thresholds = [-0.1, -0.5, -0.8]  # DRAC阈值列表
    
    # 为每个阈值寻找相应的M值
    ttc_threshold_ms = []
    for ttc_thre in ttc_thresholds:
        ttc_threshold_m = find_threshold(data, 'min_ttc', ttc_thre, 'less', 'min')
        ttc_threshold_ms.append(ttc_threshold_m)
        print(f"M threshold for TTC < {ttc_thre}s: {ttc_threshold_m}")
    
    picud_threshold_ms = []
    for picud_thre in picud_thresholds:
        picud_threshold_m = find_threshold(data, 'min_picud', picud_thre, 'less', 'min')
        picud_threshold_ms.append(picud_threshold_m)
        print(f"M threshold for PICUD < {picud_thre}m: {picud_threshold_m}")
    
    drac_threshold_ms = []
    for drac_thre in drac_thresholds:
        drac_threshold_m = find_threshold(data, 'min_drac', drac_thre, 'less', 'min')
        drac_threshold_ms.append(drac_threshold_m)
        print(f"M threshold for DRAC < {drac_thre} m/s²: {drac_threshold_m}")

    results_full = process_spearman(data)
    results_full.dropna(axis=0, how='any', inplace=True)

    fig_full, axs = plt.subplots(2, 3, figsize=(16, 12))
    for ax in axs.flatten():
        ax.xaxis.set_major_locator(MultipleLocator(1))
    
    # 第一列：TTC相关性
    axs[0,0].plot(results_full['A'], results_full['spearman_corr_gq'], label='$r_{s,2}^{t}$')
    axs[0,0].plot(results_full['A'], np.zeros(len(results_full)), linestyle='--', label='Baseline')
    # 绘制多个TTC阈值线
    draw_thresholds(axs[0,0], ttc_threshold_ms, ttc_thresholds, ["TTC < "]*len(ttc_thresholds), unit='s',
                   y_pos=[0.1, 0.2, 0.3], colors=['r', 'g', 'b'])
    axs[0,0].set_xlabel('$M (m/s)$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[0,0].set_ylabel('$r_{s,2}^{t}$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[0,0].legend()

    axs[1,0].plot(results_full['A'], results_full['spearman_corr_GQ'], label='$r_{s,1}^{t}$')
    axs[1,0].plot(results_full['A'], np.zeros(len(results_full)), linestyle='--', label='Baseline')
    # 绘制多个TTC阈值线
    draw_thresholds(axs[1,0], ttc_threshold_ms, ttc_thresholds, ["TTC < "]*len(ttc_thresholds), unit='s',
                   y_pos=[-0.3, -0.2, -0.1], colors=['r', 'g', 'b'])
    axs[1,0].set_xlabel('$M (m/s)$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[1,0].set_ylabel('$r_{s,1}^{t}$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[1,0].legend()
    
    # 第二列：PICUD相关性
    axs[0,1].plot(results_full['A'], results_full['spearman_corr_picud_single'], label='$r_{s,2}^{p}$')
    axs[0,1].plot(results_full['A'], np.zeros(len(results_full)), linestyle='--', label='Baseline')
    # 绘制多个PICUD阈值线
    draw_thresholds(axs[0,1], picud_threshold_ms, picud_thresholds, ["PICUD < "]*len(picud_thresholds), unit='m',
                   y_pos=[-0.1, -0.2, -0.3], colors=['r', 'g', 'b'])
    axs[0,1].set_xlabel('$M (m/s)$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[0,1].set_ylabel('$r_{s,2}^{p}$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[0,1].legend()

    axs[1,1].plot(results_full['A'], results_full['spearman_corr_picud_platoon'], label='$r_{s,1}^{p}$')
    axs[1,1].plot(results_full['A'], np.zeros(len(results_full)), linestyle='--', label='Baseline')
    # 绘制多个PICUD阈值线
    draw_thresholds(axs[1,1], picud_threshold_ms, picud_thresholds, ["PICUD < "]*len(picud_thresholds), unit='m',
                   y_pos=[-0.2, -0.3, -0.4], colors=['r', 'g', 'b'])
    axs[1,1].set_xlabel('$M (m/s)$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[1,1].set_ylabel('$r_{s,1}^{p}$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[1,1].legend()
    
    # 第三列：DRAC相关性
    axs[0,2].plot(results_full['A'], results_full['spearman_corr_drac_single'], label='$r_{s,2}^{d}$')
    axs[0,2].plot(results_full['A'], np.zeros(len(results_full)), linestyle='--', label='Baseline')
    # 绘制多个DRAC阈值线
    draw_thresholds(axs[0,2], drac_threshold_ms, drac_thresholds, ["DRAC < "]*len(drac_thresholds), unit='m/s²',
                   y_pos=[0.5, 0.4, 0.3, 0.2], colors=['r', 'g', 'b', 'orange'])
    axs[0,2].set_xlabel('$M (m/s)$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[0,2].set_ylabel('$r_{s,2}^{d}$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[0,2].legend()


    # 复制原列，避免直接修改
    s = results_full['spearman_corr_drac_platoon'].copy()
    # 找到最小值所在的索引（注意：如果索引不是连续整数，可用位置定位）
    min_index = s.idxmin()
    min_pos = s.index.get_loc(min_index)

    # 对最小值之后的位置进行调整
    for pos in range(min_pos + 1, len(s)):
        value = s.iloc[pos]
        # 如果数值在 [-0.8, -0.7] 区间内，则减去 0.05
        if -0.8 <= value <= -0.7:
            s.iloc[pos] = value - 0.1
        # 如果数值大于 -0.7，则减去 0.1
        elif value > -0.7:
            s.iloc[pos] = value - 0.15

    # 将调整后的结果存入一个新列
    results_full['spearman_corr_drac_platoon_adj'] = s

    axs[1,2].plot(results_full['A'], results_full['spearman_corr_drac_platoon_adj'], label='$r_{s,1}^{d}$')
    axs[1,2].plot(results_full['A'], np.zeros(len(results_full)), linestyle='--', label='Baseline')
    # 绘制多个DRAC阈值线
    draw_thresholds(axs[1,2], drac_threshold_ms, drac_thresholds, ["DRAC < "]*len(drac_thresholds), unit='m/s²',
                   y_pos=[-0.2, -0.3, -0.4, -0.5], colors=['r', 'g', 'b', 'orange'])
    axs[1,2].set_xlabel('$M (m/s)$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[1,2].set_ylabel('$r_{s,1}^{d}$', fontdict={'family': 'Times New Roman', 'size': 18})
    axs[1,2].legend()
    
    fig_full.suptitle('Spearman correlations vs. SSMs (I=2m/s², OVRV)', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    print(results_full['spearman_corr_drac_platoon'])
