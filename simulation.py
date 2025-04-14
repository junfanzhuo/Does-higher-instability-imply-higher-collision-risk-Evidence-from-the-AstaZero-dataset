from matplotlib import pyplot as plt
import numpy as np
import math
import calibrated_cf_params as ca
import pandas as pd


def IDM(alpha, beta, s0, T, v1, v2, ivs, v0):
    try:
        g = s0 + T * v2 - (v2 * (v1 - v2)) / (2 * (alpha * beta)**0.5)
        a = alpha * (1 - (v2 / v0)**4 - (g / ivs)**2)
        if not np.isfinite(a):
            raise ValueError("Calculated acceleration is not finite")
    except Exception as e:
        print(f"Error in IDM calculation: {e}")
        print(f"alpha: {alpha}, beta: {beta}, s0: {s0}, T: {T}, v1: {v1}, v2: {v2}, ivs: {ivs}, v0: {v0}")
        return np.nan
    return a

def Linear_ACC(K1,K2,T,v1,v2,ivs,s0):
    
    a = K1*(ivs-s0-T*v2) + K2*(v1-v2)
    
    return a

def Linear_ACC(K1,K2,T,v1,v2,ivs,s0):
    
    a = K1*(ivs-s0-T*v2) + K2*(v1-v2)
    
    return a

def calculate_ttc(x1, x2, v1, v2, ttc_u):

    if x1 - x2 <= 0:
        return 0
    if x1 - x2 > 0:
        if v2 - v1 > 0:
            ttc = (x1 - x2) / (v2 - v1)
            return min(ttc, ttc_u)
        if v2 - v1 <= 0:
            return ttc_u

def calculate_picud(v1,v2,ivs,dr,tau):
    temp1 = (v1**2 - v2**2)/(2*dr)
    temp2 = ivs - v1*tau
    return temp1 + temp2

def calculate_drac(v1,v2,ivs):
    if v2 > v1:

        temp1 = (v2 - v1)**2
        temp2 = -2*ivs
        return temp1/temp2

    else:
        return 0


def simulation(params, A, I, draw):
    
    # general
    v_eq = 15  # equilibrium speed
    ttc_u = 40
    v0 = 30
    
    # simulation setting (default)
    timestep = 0.1
    total_time = 300
    initial_position = 1000
    initial_speed = v_eq
    initial_spacing = 40
    total_vehicles = 5
    
    # simulation setting (vary)
    time_disturb_col = 0.5
    time_disturb = int(total_time * time_disturb_col / timestep)
    speedChange = A * (-1)  # acceleration: +; deceleration: -
    celeration = I * (-1)
    disturbDuration = math.ceil(speedChange / (celeration * timestep))
    total_step = int(total_time / timestep)
    
    # 初始化变量：位置、速度、加速度、TTC，以及新增的模糊安全指标数组
    x = np.zeros((total_vehicles, total_step))
    v = np.zeros((total_vehicles, total_step))
    a = np.zeros((total_vehicles, total_step))
    ttc = np.zeros((total_vehicles, total_step))
    picud = np.zeros((total_vehicles, total_step))
    drac = np.zeros((total_vehicles, total_step))
    
    for j in range(total_vehicles):
        x[j][0] = initial_position - j * initial_spacing
        v[j][0] = initial_speed
        ttc[j][0] = 0
        picud[j][0] = 0 
        drac[j][0] = 0
    
    # simulation
    for t in range(1, total_step):
            
        startDisturb = time_disturb
        initial_duration = math.ceil((1.0) / (abs(celeration) * timestep))  # 先加速1 m/s所需步数
        celeration_needed = A + 1   # 由于先加速了1 m/s，所以后续需要减速 (A+1) m/s
        after_duration = math.ceil(celeration_needed / (abs(celeration) * timestep))
        # 总扰动结束时刻
        endDisturb = time_disturb + disturbDuration
        
        if t < startDisturb or t >= endDisturb:
            a[0][t] = 0
        elif startDisturb <= t < startDisturb + initial_duration:
            a[0][t] = -celeration 
        elif startDisturb + initial_duration <= t < endDisturb:
            a[0][t] = celeration       
        
        v[0][t] = v[0][t-1] + 0.5 * (a[0][t-1] + a[0][t]) * timestep
        x[0][t] = x[0][t-1] + v[0][t-1] * timestep + 0.5 * (a[0][t-1]) * (timestep)**2
        
        for k in range(1, total_vehicles):
            
            K1,K2,T,s0 = params[k-1] # Linear ACC
            #alpha, beta, s0, T = params[k-1] # IDM
            v_front = v[k-1][t-1]
            v_rear = v[k][t-1]
            x_front = x[k-1][t-1]
            x_rear = x[k][t-1]
            ivs = x_front - x_rear  # inter-vehicle spacing
            
            # 若数据异常，则返回nan
            if ivs <= 0 or not np.isfinite(v_front) or not np.isfinite(v_rear) \
            or not np.isfinite(x_front) or not np.isfinite(x_rear) \
            or v_front < 0 or v_rear < 0:
                print("Triggering condition:", 
                    "t:", t,
                    "A:", A,
                    "I", I,
                    "ivs<=0:" , ivs<=0, 
                    "not isfinite(v_front):", not np.isfinite(v_front),
                    "not isfinite(v_rear):", not np.isfinite(v_rear),
                    "not isfinite(x_front):", not np.isfinite(x_front),
                    "not isfinite(x_rear):", not np.isfinite(x_rear),
                    "v_front<0:", v_front<0,
                    "v_rear<0:", v_rear<0)
                return [np.nan] * 18
            
            #a[k][t] = IDM(alpha, beta, s0, T, v_front, v_rear, ivs, v0) #IDM 
            a[k][t] = Linear_ACC(K1, K2, T, v_front, v_rear, ivs, s0) # Linear ACC
           
            v[k][t] = v[k][t-1] + 0.5 * (a[k][t-1] + a[k][t]) * timestep
            x[k][t] = x[k][t-1] + v[k][t-1] * timestep + 0.5 * (a[0][t-1]) * (timestep)**2
            
            # 记录
            ttc[k][t] = calculate_ttc(x_front, x_rear, v_front, v_rear, ttc_u)
            picud[k][t] = calculate_picud(v_front, v_rear, ivs, 3.3, 1)
            drac[k][t] = calculate_drac(v_front, v_rear, ivs)
    
    if draw == True:
        # drawings
        colors = [
            '#1f77b4',  # 蓝色
            '#ff7f0e',  # 橙色
            '#2ca02c',  # 绿色
            '#d62728',  # 红色
            '#9467bd',  # 紫色
            '#8c564b',  # 棕色
            '#e377c2',  # 粉色
            '#7f7f7f',  # 灰色
            '#bcbd22',  # 橄榄绿色
            '#17becf'   # 青色
        ]
        
        ob_l = startDisturb - 50
        ob_u = startDisturb + 300
        xx = timestep * np.arange(total_step)[ob_l:ob_u]
        bwith = 2
        
        # Speed
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(total_vehicles):
            if i == 0:
                ax1.plot(xx, v[i][ob_l:ob_u], alpha=0.7, label='V0', color=colors[i])
            elif i == 1:
                ax1.plot(xx, v[i][ob_l:ob_u], alpha=0.7, label='V1', color=colors[1])
            elif i == 2:
                ax1.plot(xx, v[i][ob_l:ob_u], alpha=0.7, label='V2', color=colors[2])
            elif i == 3:
                ax1.plot(xx, v[i][ob_l:ob_u], alpha=0.7, label='V3', color=colors[3])
            elif i == 4:
                ax1.plot(xx, v[i][ob_l:ob_u], alpha=0.7, label='V4', color=colors[4])
            
        ax1.set_xlabel('Time (s)', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax1.set_ylabel('Speed (m/s)', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax1.xaxis.set_tick_params(labelsize=17)
        ax1.yaxis.set_tick_params(labelsize=17)
        ax1.legend(prop={'size': 15})
        ax1.annotate('Start disturbance', 
                     xy=(time_disturb * timestep, initial_speed), 
                     xytext=(time_disturb * timestep - 1, initial_speed - 0.25 * A),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=15)
        ax1.spines['top'].set_linewidth(bwith)
        ax1.spines['bottom'].set_linewidth(bwith)
        ax1.spines['left'].set_linewidth(bwith)
        ax1.spines['right'].set_linewidth(bwith)
        
        # TTC
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(total_vehicles):
            if i == 0:
                ax2.plot(xx, ttc[i][ob_l:ob_u], alpha=0.7, color='none')
            elif i == 1:
                ax2.plot(xx, ttc[i][ob_l:ob_u], alpha=0.7, label='V1', color=colors[1])
            elif i == 2:
                ax2.plot(xx, ttc[i][ob_l:ob_u], alpha=0.7, label='V2', color=colors[2])
            elif i == 3:
                ax2.plot(xx, ttc[i][ob_l:ob_u], alpha=0.7, label='V3', color=colors[3])
            elif i == 4:
                ax2.plot(xx, ttc[i][ob_l:ob_u], alpha=0.7, label='V4', color=colors[4])
        ax2.set_xlabel('Time (s)', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax2.set_ylabel('TTC (s)', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax2.xaxis.set_tick_params(labelsize=17)
        ax2.yaxis.set_tick_params(labelsize=17)
        ax2.legend(prop={'size': 15})
        ax2.spines['top'].set_linewidth(bwith)
        ax2.spines['bottom'].set_linewidth(bwith)
        ax2.spines['left'].set_linewidth(bwith)
        ax2.spines['right'].set_linewidth(bwith)
        
        # picud
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(1, total_vehicles):
            ax3.plot(xx, picud[i][ob_l:ob_u] - picud[i][ob_l], alpha=0.7, label=f'PICUD V{i}', color=colors[i])
        ax3.set_xlabel('Time (s)', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax3.set_ylabel('PICUD', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax3.xaxis.set_tick_params(labelsize=17)
        ax3.yaxis.set_tick_params(labelsize=17)
        ax3.legend(prop={'size': 15})
        ax3.spines['top'].set_linewidth(bwith)
        ax3.spines['bottom'].set_linewidth(bwith)
        ax3.spines['left'].set_linewidth(bwith)
        ax3.spines['right'].set_linewidth(bwith)

        # drac
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(1, total_vehicles):
            ax3.plot(xx, drac[i][ob_l:ob_u], alpha=0.7, label=f'DRAC V{i}', color=colors[i])
        ax3.set_xlabel('Time (s)', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax3.set_ylabel('DRAC', fontdict={'family': 'Times New Roman', 'size': 20, 'weight': 'bold'})
        ax3.xaxis.set_tick_params(labelsize=17)
        ax3.yaxis.set_tick_params(labelsize=17)
        ax3.legend(prop={'size': 15})
        ax3.spines['top'].set_linewidth(bwith)
        ax3.spines['bottom'].set_linewidth(bwith)
        ax3.spines['left'].set_linewidth(bwith)
        ax3.spines['right'].set_linewidth(bwith)
        
        plt.show()

    mv1 = v_eq - np.min(v[1][startDisturb:])
    mv2 = v_eq - np.min(v[2][startDisturb:])
    mv3 = v_eq - np.min(v[3][startDisturb:])
    mv4 = v_eq - np.min(v[4][startDisturb:])
    
    ttc1 = np.min(ttc[1][startDisturb:])
    ttc2 = np.min(ttc[2][startDisturb:])
    ttc3 = np.min(ttc[3][startDisturb:])
    ttc4 = np.min(ttc[4][startDisturb:])
    
    # PICUD
    picud1 = np.min(picud[1][startDisturb:])
    picud2 = np.min(picud[2][startDisturb:])
    picud3 = np.min(picud[3][startDisturb:])
    picud4 = np.min(picud[4][startDisturb:])

    # DRAC
    drac1 = np.min(drac[1][startDisturb:])
    drac2 = np.min(drac[2][startDisturb:])
    drac3 = np.min(drac[3][startDisturb:])
    drac4 = np.min(drac[4][startDisturb:])

    return [A, mv1, mv2, mv3, mv4, ttc_u, ttc1, ttc2, ttc3, ttc4, picud1, picud2, picud3, picud4, drac1, drac2, drac3, drac4]

if __name__ == '__main__':
    
    file_range = [0,1,2,3,4,5,6,7,8,9]  # 2 and 9 hvs
    A_range = 10.1
    I_range = [1,2,3]
    draw = False

    columns = ['file','mv0','mv1','mv2','mv3','mv4','ttc0','ttc1','ttc2','ttc3','ttc4','picud1','picud2','picud3','picud4','drac1','drac2','drac3','drac4']
    
    for I in I_range:

        simulation_results = []
        
        for i in file_range:
            params = ca.cf_params_Linear_ACC(i)
            for j in np.arange(1, A_range, 0.1):
                temp = [i]
                results_temp = simulation(params, j, I, draw)
                temp.extend(results_temp)
                simulation_results.append(temp)

        data = pd.DataFrame(columns=columns, data=simulation_results)
        file_name = 'records_simulation_LinearACC_de' + '_' + str(I)
        data.to_csv(file_name, index=False)
    
    # params = ca.cf_params_Linear_ACC(2)
    # simulation(params, 9, 3, True)
