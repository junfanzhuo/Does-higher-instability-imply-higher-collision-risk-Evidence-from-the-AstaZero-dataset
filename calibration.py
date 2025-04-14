# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:26:57 2024

@author: asd26
"""

import pandas as pd
from matplotlib import pyplot as plt
from sko.GA import GA
import numpy as np
import math
import calibrated_cf_params as ca

'''linearized coefficients'''

def calculate_f_IDM(alpha,T,s0,v_eq,v0,beta):
    
    g_eq = (s0 + T*v_eq)/(1-(v_eq/v0)**4)**0.5
    f_xdiff = 2*alpha*(s0+T*v_eq)**2/(g_eq)**3
    f_v = -alpha*(4*(v_eq)**3/(v0)**4 + 2*T*(s0+T*v_eq)/(g_eq)**2)
    f_vdiff = ((alpha/beta)**0.5)*(v_eq*(s0+T*v_eq))/(g_eq)**2
    
    return [f_xdiff,f_v,f_vdiff]

def calculate_f_FVDM(T,K1_fvdm,K2_fvdm,s0,v_max,v_eq):
    
    q = math.acos((v_max-2*v_eq)/v_max) 
    g_eq = s0 + q*T*v_max/math.pi
    
    f_xdiff = math.pi*math.sin(math.pi*(g_eq-s0)/(T*v_max))/(2*T)
    f_v = -K1_fvdm-K2_fvdm
    f_vdiff = K2_fvdm
    
    return [f_xdiff,f_v,f_vdiff]

def linearized_acceleration(x1,x2,v1,v2,f_xdiff,f_v,f_vdiff):
    #x1,v1: front vehicles' position and speed;
    #x2,v2: following vehicles' position and speed
    
    xdiff = f_xdiff*(x1-x2-l)
    vdiff = f_vdiff*(v1-v2)
    v = f_v*v2
    
    return xdiff+vdiff+v

'''CF models'''
def IDM(alpha,beta,s0,T,v1,v2,ivs,v0):
    
    g = s0 + max(0,T*v2 + (v2*(v1-v2))/(2*(alpha*beta)**0.5))
    a = alpha*(1-(v2/v0)**4-(g/ivs)**2)
    
    return a

def FVDM_gap(s0,T,ivs,v0):
    
    if ivs <= s0:
        return 0
    if (ivs > s0) and (ivs <= s0+T*v0):
        a = math.pi*(ivs-s0)/(T*v0)
        return v0*(1-math.cos(a))/2
    if ivs > s0+T*v0:
        return v0

def FVDM(K1,K2,s0,T,v1,v2,IVS,v0):
    
    V = FVDM_gap(s0,T,IVS,v0)
    a = K1*(V-v2) + K2*(v1-v2)
    
    return a

def Linear_ACC(K1,K2,T,v1,v2,ivs,s0):
    
    a = K1*(ivs-s0-T*v2) + K2*(v1-v2)
    
    return a

def OVM(K, T, s0, v2, ivs, v0):
    V = max(0, min(v0, (ivs-s0)/T))
    a = K * (V - v2)
    return a

'''fitness'''    
def fitness_IDM_a(params,data,target_names):
    
    alpha, beta, s0, T = params
    total_error = 0
    for i in range(len(data)):
        
        v1 = data[target_names[1]].iloc[i]
        v2 = data[target_names[2]].iloc[i]
        IVS = data[target_names[3]].iloc[i]
        
        theoretical = IDM(alpha,beta,s0,T,v1,v2,IVS,v0)
        observed = data[target_names[0]].iloc[i]

        total_error += (theoretical-observed)**2
    
    return (total_error/len(data))**0.5

def fitness_IDM_s(params,data,target_names,t):
    
    alpha, beta, s0, T  = params
    total_error = 0
    for i in range(len(data)-1):
        
        v1_t1 = data[target_names[1]].iloc[i]
        v1_t2 = data[target_names[1]].iloc[i+1]
        v2_t1 = data[target_names[2]].iloc[i]
        IVS_t1 = data[target_names[3]].iloc[i]
        IVS_t2 = data[target_names[3]].iloc[i+1]
        
        theoretical_a = IDM(alpha, beta, s0, T, v1_t1, v2_t1, IVS_t1, v0)
        
        # 1 stands for the time step. The original is 0.1s and is changed to 1s after data processing
        observed = IVS_t2
        theoretical = IVS_t1 - v2_t1*t - 0.5*theoretical_a*(t)**2 + 0.5*(v1_t1+v1_t2)*t
        
        total_error += (theoretical-observed)**2
    
    return (total_error/len(data))**0.5

def fitness_FVDM_a(params,data,target_names):
    
    K1,K2,s0,T  = params
    total_error = 0
    for i in range(len(data)):
        
        v1 = data[target_names[1]].iloc[i]
        v2 = data[target_names[2]].iloc[i]
        IVS = data[target_names[3]].iloc[i]
        
        theoretical = FVDM(K1, K2, s0, T, v1, v2, IVS, v0)
        observed = data[target_names[0]].iloc[i]
        
        total_error += (theoretical-observed)**2
    
    return (total_error/len(data))**0.5

def fitness_FVDM_s(params,data,target_names,t,*args):
    
    K1,K2,s0,T  = params
    alpha1,beta1,alpha2,beta2,alpha3,beta3,alpha4,beta4 = args
    total_error = 0
    epsilon = 1e-5
    for i in range(len(data)-1):
        
        v1_t1 = data[target_names[1]].iloc[i]
        v1_t2 = data[target_names[1]].iloc[i+1]
        v2_t1 = data[target_names[2]].iloc[i]
        IVS_t1 = data[target_names[3]].iloc[i]
        IVS_t2 = data[target_names[3]].iloc[i+1]
        
        theoretical_a = FVDM(K1, K2, s0, T, v1_t1, v2_t1, IVS_t1, v0)
        
        observed = IVS_t2
        theoretical = IVS_t1 - v2_t1*t - 0.5*theoretical_a*(t)**2 + 0.5*(v1_t1+v1_t2)*t
        
        total_error += (theoretical-observed)**2
    
    rmse = (total_error/len(data))**0.5
    # 添加正则项
    regularization1 = alpha1 * (1 / ((K1 - 0.2)**2 + epsilon) + 1 / ((K1 - 2)**2 + epsilon)) + \
                     beta1 * ((K1 - 0.2)**2 / (0.1 - 2)**2 + (K1 - 2)**2 / (2 - 0.1)**2)
    regularization2 = alpha2 * (1 / ((K2 - 0.1)**2 + epsilon) + 1 / ((K2 - 2)**2 + epsilon)) + \
                     beta2 * ((K2 - 0.1)**2 / (0.1 - 2)**2 + (K2 - 2)**2 / (2 - 0.1)**2)
    regularization3 = alpha3 * (1 / ((s0 - 1)**2 + epsilon) + 1 / ((s0 - 10)**2 + epsilon)) + \
                     beta3 * ((s0 - 1)**2 / (1 - 10)**2 + (s0 - 10)**2 / (10 - 1)**2)
    regularization4 = alpha4 * (1 / ((T - 2)**2 + epsilon)) + \
                     beta4 * ((T - 2)**2 / (0.1 - 2)**2)
    
    
    return rmse+regularization1+regularization2+regularization3+regularization4

def fitness_Linear_ACC_s(params,data,target_names,t):
    
    K1,K2,T,s0  = params
    total_error = 0
    for i in range(len(data)-1):
        
        v1_t1 = data[target_names[1]].iloc[i]
        v1_t2 = data[target_names[1]].iloc[i+1]
        v2_t1 = data[target_names[2]].iloc[i]
        IVS_t1 = data[target_names[3]].iloc[i]
        IVS_t2 = data[target_names[3]].iloc[i+1]
        
        theoretical_a = Linear_ACC(K1, K2, T, v1_t1, v2_t1, IVS_t1, s0)
        
        # 1 stands for the time step. The original is 0.1s and is changed to 1s after data processing
        observed = IVS_t2
        theoretical = IVS_t1 - v2_t1*t - 0.5*theoretical_a*(t)**2 + 0.5*(v1_t1+v1_t2)*t
        
        total_error += (theoretical-observed)**2
    
    return (total_error/len(data))**0.5

def fitness_OVM_s(params,data,target_names,t):
    
    K,T,s0  = params
    total_error = 0
    for i in range(len(data)-1):
        
        v1_t1 = data[target_names[1]].iloc[i]
        v1_t2 = data[target_names[1]].iloc[i+1]
        v2_t1 = data[target_names[2]].iloc[i]
        IVS_t1 = data[target_names[3]].iloc[i]
        IVS_t2 = data[target_names[3]].iloc[i+1]
        
        theoretical_a = OVM(K, T, s0, v2_t1, IVS_t1, v0)
        
        observed = IVS_t2
        theoretical = IVS_t1 - v2_t1*t - 0.5*theoretical_a*(t)**2 + 0.5*(v1_t1+v1_t2)*t
        
        total_error += (theoretical-observed)**2
    
    return (total_error/len(data))**0.5


def processData(data,t):
    
    group_size = int(t/0.1)
    ob_period = len(data)
    end = int(ob_period/group_size)
    data['group'] = data.index//group_size
    data = data.groupby('group').mean()
    
    data['Acc1'] = data['Speed1'].diff()/data['Time'].diff()
    data['Acc2'] = data['Speed2'].diff()/data['Time'].diff()
    data['Acc3'] = data['Speed3'].diff()/data['Time'].diff()
    data['Acc4'] = data['Speed4'].diff()/data['Time'].diff()
    data['Acc5'] = data['Speed5'].diff()/data['Time'].diff()
    data = data[:end]
    data.dropna(axis=0,how='any',inplace=True)
    data.reset_index(drop=True,inplace=True)
    
    return data

def calibration(data,t,target_names):
    
    
    # Calibrated parameters ranges
    alpha_lb = 0.1
    alpha_ub = 4
    
    beta_lb = 0.1
    beta_ub = 4
    
    s0_lb = 1
    s0_ub = 10
    
    T_lb = 0.1
    T_ub = 4
    
    K1_lb = 0.1
    K1_ub = 2
    K2_lb = 0.1
    K2_ub = 2

    K_OVM_lb = 0.1
    K_OVM_ub = 4
    
    alpha_dis = []
    beta_dis = []
    s0_dis = []
    T_dis = []
    K1_dis = []
    K2_dis = []
    K_OVM_dis = []


    RMSE = []
        
    regu_co = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    for j in range(len(target_names)):
        
        ga = GA(func=lambda params: fitness_Linear_ACC_s(params,data,target_names[j],t),n_dim=4,size_pop=100,max_iter=500,
            lb=[K1_lb,K2_lb,T_lb,s0_lb],
            ub=[K1_ub,K2_ub,T_ub,s0_ub],
            precision = 1e-5)
        best_x , best_y = ga.run(10)

        K1_dis.append(best_x[0])
        K2_dis.append(best_x[1])
        T_dis.append(best_x[2])
        s0_dis.append(best_x[3])
        RMSE.append(best_y) 
        print(j)
    
    samples = [list(item) for item in zip(K1_dis,K2_dis,T_dis,s0_dis)]
    return RMSE,samples

    
if __name__ == '__main__':
    
    l = 5
    v0 = 30
    t = 1 # time step (data and simulation)
    
    path = 'data_filtered/platoon10.csv'
    usecols = ['Time', 'Speed1', 'Speed2', 'Speed3', 'Speed4', 'Speed5', 'IVS1', 'IVS2', 'IVS3', 'IVS4']
    columns_type = {'Time':float,
                    'Speed1':float,'Speed2':float,'Speed3':float,'Speed4':float,'Speed5':float,
                    'IVS1':float,'IVS2':float,'IVS3':float,'IVS4':float,
                    }
    data = pd.read_csv(path,usecols=usecols, dtype=columns_type)
    data = processData(data,t)
    
    v_eq = data['Speed1'][100:200].mean()
    target_names = [['Acc2','Speed1','Speed2','IVS1'],
                    ['Acc3','Speed2','Speed3','IVS2'],
                    ['Acc4','Speed3','Speed4','IVS3'],
                    ['Acc5','Speed4','Speed5','IVS4']]
    
    RMSE,samples = calibration(data,t,target_names)

    print(RMSE)
    print(samples)
    