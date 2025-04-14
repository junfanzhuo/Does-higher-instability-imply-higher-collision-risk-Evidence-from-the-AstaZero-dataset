# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:08:06 2024

@author: asd26
"""

import numpy as np

def cf_params_Linear_ACC():

    params1 = [[0.11,0.1,0.75,7.77],
              [0.13,0.19,0.66,8.89],
              [0.11,0.33,1,3.79],
              [0.1,0.33,0.91,6.83]]
    
    params2 = [[0.1,0.23,0.78,7.08],
               [0.11,0.29,0.8,6.4],
               [0.1,0.45,0.92,5.16],
               [0.1,0.26,0.97,3.88]]
    
    params3 = [[0.11,0.31,2.41,2.84],
               [0.1,0.41,2.57,4.75],
               [0.11,0.11,2.82,1.95],
               [0.11,0.18,1.42,3.18]] # HV
    
    params4 = [[0.11,0.33,2,6.92],
               [0.12,0.11,2.17,6.79],
               [0.1,0.11,3.32,3.95],
               [0.1,0.3,2.35,1.21]]
    
    params5 = [[0.1,0.16,0.99,4.24],
               [0.1,0.1,1.11,1.41],
               [0.1,0.2,1.07,2.12],
               [0.1,0.34,0.8,8.39]]
    
    params6 = [[0.1,0.16,0.9,5.69],
               [0.17,0.12,0.86,6.42],
               [0.14,0.19,0.73,7.42],
               [0.11,0.39,0.68,9.89]]
    
    params7 = [[0.1,0.35,0.87,5.78],
               [0.13,0.39,0.73,7.66],
               [0.11,0.13,1.07,4.08],
               [0.1,0.26,1.1,1.95]]
    
    params8 = [[0.1,0.37,0.75,8.66],
               [0.1,0.18,1.01,2.65],
               [0.1,0.2,1.25,1.29],
               [0.1,0.35,1,4.17]]
    
    params9 = [[0.12,0.19,0.78,8.01],
               [0.1,0.17,0.88,4.79],
               [0.11,0.15,1.06,3.4],
               [0.11,0.16,1.06,1.42]]
    
    params10 = [[0.1,0.27,2.23,2.4],
                [0.1,0.28,2.15,1.51],
                [0.12,0.13,1.4,2.26],
                [0.11,0.4,1.94,3.06]]
    
    params = [params1,params2,params3,params4,params5,params6,params7,params8,params9,params10]
    
    return params

def cf_params_IDM(n):
    
    params1 = [[0.55,3.96,3.69,0.81],
               [0.26,2.26,4.56,0.64],
               [0.77,3.77,4.3,0.8],
               [0.89,3.92,4.78,0.69]]
    
    params2 = [[0.52,3.78,3.57,0.8],
               [0.73,3.88,2.92,0.86],
               [0.9,3.57,1.38,0.92],
               [1,3.89,3.16,0.8]]
    
    params3 = [[0.21,1.45,3.29,1.65],
               [0.11,2.08,2.3,2.5],
               [0.13,1.95,4.14,2.07],
               [0.35,3.29,9.49,0.84]]
    
    params4 = [[0.31,0.63,6.25,1.86],
               [0.29,1.78,4.74,1.98],
               [0.42,1.34,6.45,2.89],
               [0.54,3.55,4.47,1.84]]
    
    params5 = [[0.24,3.72,4.45,0.88],
               [0.33,3.55,5.8,0.66],
               [0.29,3.93,3.48,0.7],
               [0.3,2.57,2.4,0.83]]
    
    params6 = [[0.4,4,4.66,0.76],
               [0.44,3.88,4.88,0.68],
               [0.36,3.86,4.74,0.58],
               [0.32,2.97,1.3,0.99]]
    
    params7 = [[0.44,3.87,6.06,0.68],
               [0.37,3.63,3.21,0.79],
               [0.46,3.94,5.48,0.64],
               [0.46,3.97,4.64,0.6]]
    
    params8 = [[0.35,3.95,6.99,0.64],
               [0.38,3.9,1.73,0.85],
               [0.33,3.82,7.1,0.58],
               [0.27,3.77,2.97,0.73]]
    
    params9 = [[0.31,2.82,9.88,0.46],
               [0.53,3.94,6.33,0.64],
               [0.48,3.98,9.62,0.51],
               [0.3,3.93,3.98,0.71]] 
    
    params10 = [[0.31,1.66,5,1.15],
                [0.23,2.27,2.34,1.46],
                [0.22,2.51,3.37,0.98],
                [0.31,2.87,2.06,1.18]]
    
    params = [params1,params2,params3,params4,params5,params6,params7,params8,params9,params10]
    
    return params[n]

def statistics(data_):
    
    data = data_[0:2] + data_[3:9]
    
    flat_alpha = [row[0] for sublist in data for row in sublist]
    min_alpha = min(flat_alpha)
    max_alpha = max(flat_alpha)
    median_alpha = np.median(flat_alpha)
    mean_alpha = np.mean(flat_alpha)
    
    flat_beta = [row[1] for sublist in data for row in sublist]
    min_beta = min(flat_beta)
    max_beta = max(flat_beta)
    median_beta = np.median(flat_beta)
    mean_beta = np.mean(flat_beta)
    
    flat_s0 = [row[2] for sublist in data for row in sublist]
    min_s0 = min(flat_s0)
    max_s0 = max(flat_s0)
    median_s0 = np.median(flat_s0)
    mean_s0 = np.mean(flat_s0)
    
    flat_T = [row[3] for sublist in data for row in sublist]
    min_T = min(flat_T)
    max_T = max(flat_T)
    median_T = np.median(flat_T)
    mean_T = np.mean(flat_T)
    
    info = [[min_alpha,max_alpha,median_alpha,mean_alpha],
            [min_beta,max_beta,median_beta,mean_beta],
            [min_s0,max_s0,median_s0,mean_s0],
            [min_T,max_T,median_T,mean_T]]
    
    return info

info = statistics(cf_params_Linear_ACC())
print(info)