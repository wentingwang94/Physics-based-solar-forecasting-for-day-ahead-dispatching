# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:45:21 2022
author: Wenting Wang
School of Electrical Engineering and Automation
Harbin Institute of Technology
email: wangwenting3000@gmail.com
"""

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import packages
import gurobipy as grb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# establish optimal model by Gurobi
M = grb.Model()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# structured data
# dictionary "structure" contains three kinds of system information, namely, bus data, branch data, and thermal power generation data.
# source: https://al-roomi.org/multimedia/Power_Flow/30BusSystem/IEEE30BusSystemDATA2.pdf
# source: https://matpower.org/docs/ref/matpower5.0/case_ieee30.html
structure = {}

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# update bus data
#                           Node Load(MW)     Vmin  Vmax
structure['bus'] = np.array([
                            [1,    0,         0.94, 1.06],
                            [2,    21.7,      0.94, 1.06],
                            [3,    2.4,       0.94, 1.06],
                            [4,    7.6,       0.94, 1.06],
                            [5,    94.2,      0.94, 1.06],
                            [6,    0,         0.94, 1.06],
                            [7,    22.8,      0.94, 1.06],
                            [8,    30,        0.94, 1.06],
                            [9,    0,         0.94, 1.06],
                            [10,   5.8,       0.94, 1.06],
                            [11,   0,         0.94, 1.06],
                            [12,   11.2,      0.94, 1.06],
                            [13,   0,         0.94, 1.06],
                            [14,   6.2,       0.94, 1.06],
                            [15,   8.2,       0.94, 1.06],
                            [16,   3.5,       0.94, 1.06],
                            [17,   9,         0.94, 1.06],
                            [18,   3.2,       0.94, 1.06],
                            [19,   9.5,       0.94, 1.06],
                            [20,   2.2,       0.94, 1.06],
                            [21,   17.5,      0.94, 1.06],
                            [22,   0,         0.94, 1.06],
                            [23,   3.2,       0.94, 1.06],
                            [24,   8.7,       0.94, 1.06],
                            [25,   0,         0.94, 1.06],
                            [26,   3.5,       0.94, 1.06],
                            [27,   0,         0.94, 1.06],
                            [28,   0,         0.94, 1.06],
                            [29,   2.4,       0.94, 1.06],
                            [30,   10.6,      0.94, 1.06]
                            ])

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# branch data
#                              from_bus, to_bus, reactance(x), Power Limit
structure["branch"] = np.array([
                               [1,       2,      0.0575,       130],
                               [1,       3,      0.1852,       130],
                               [2,       4,      0.1737,        65],
                               [3,       4,      0.0379,       130],
                               [2,       5,      0.1983,       130],
                               [2,       6,      0.1763,        65],
                               [4,       6,      0.0414,        90],
                               [5,       7,      0.1160,        70],
                               [6,       7,      0.0820,       130],
                               [6,       8,      0.0420,        32],
                               [6,       9,      0.2080,        65],
                               [6,       10,     0.5560,        32],
                               [9,       11,     0.2080,        65],
                               [9,       10,     0.1100,        65],
                               [4,       12,     0.2560,        65],
                               [12,      13,     0.1400,        65],
                               [12,      14,     0.2559,        32],
                               [12,      15,     0.1304,        32],
                               [12,      16,     0.1987,        32],
                               [14,      15,     0.1997,        16],
                               [16,      17,     0.1932,        16],
                               [15,      18,     0.2185,        16],
                               [18,      19,     0.1292,        16],
                               [19,      20,     0.0680,        32],
                               [10,      20,     0.2090,        32],
                               [10,      17,     0.0845,        32],
                               [10,      21,     0.0749,        32],
                               [10,      22,     0.1499,        32],
                               [21,      22,     0.0236,        32],
                               [15,      23,     0.2020,        16],
                               [22,      24,     0.1790,        16],
                               [23,      24,     0.2700,        16],
                               [24,      25,     0.3292,        16],
                               [25,      26,     0.3800,        16],
                               [25,      27,     0.2087,        16],
                               [28,      27,     0.3960,        65],
                               [27,      29,     0.4153,        16],
                               [27,      30,     0.6027,        16],
                               [29,      30,     0.4533,        16],
                               [8,       28,     0.2000,        32],
                               [6,       28,     0.0599,        32]
                               ])

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
# thermal power generator
#                                          no. Bus  Pg_max  Pg_min  ramp-up(RU)  ramp_down(RD)  cost factor_a   cost factor_b   cost factor_c

# structure["thermal_generator"] = np.array([
#                                           [1,  1,   200,    50,     180,         180,           0.00160,        2.00000,        150.000],
#                                           [2,  2,   80,     20,     72,          72,            0.01000,        2.50000,        25.0000],
#                                           [3,  5,   50,     15,     45,          45,            0.06525,        1.00000,        0.00000],
#                                           [4,  8,   35,     10,     31.5,        31.5,          0.00834,        3.25000,        0.00000],
#                                           [5,  11,  30,     10,     27,          27,            0.02500,        3.00000,        0.00000],
#                                           [6,  13,  40,     12,     36,          36,            0.02500,        3.00000,        0.00000]
#                                           ])


structure["thermal_generator"] = np.array([
                                          [1,  2,   80,     20,     72,          72,            0.01000,        2.50000,        25.0000],
                                          [2,  5,   50,     15,     45,          45,            0.06525,        1.00000,        0.00000],
                                          [3,  8,   35,     10,     31.5,        31.5,          0.00834,        3.25000,        0.00000],
                                          [4,  11,  30,     10,     27,          27,            0.02500,        3.00000,        0.00000],
                                          [5,  13,  40,     12,     36,          36,            0.02500,        3.00000,        0.00000]
                                          ])



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# define a dictionary "N_to_B"
# When the power P is a positive value, "N_to_B" indicates that the power is output from the node to the branch.
N_to_B = {}

N_to_B[1] = [(1, 2), (1, 3)]
N_to_B[2] = [(2, 4), (2, 5), (2, 6)]
N_to_B[3] = [(3, 4)]
N_to_B[4] = [(4, 6), (4, 12)]
N_to_B[5] = [(5, 7)]
N_to_B[6] = [(6, 7), (6, 8), (6, 9), (6, 10), (6, 28)]
N_to_B[7] = []
N_to_B[8] = [(8, 28)]
N_to_B[9] = [(9, 10), (9, 11)]
N_to_B[10] = [(10, 17), (10, 20), (10, 21), (10, 22)]
N_to_B[11] = []
N_to_B[12] = [(12, 13), (12, 14), (12, 15), (12, 16)]
N_to_B[13] = []
N_to_B[14] = [(14, 15)]
N_to_B[15] = [(15, 18), (15, 23)]
N_to_B[16] = [(16, 17)]
N_to_B[17] = []
N_to_B[18] = [(18, 19)]
N_to_B[19] = [(19, 20)]
N_to_B[20] = []
N_to_B[21] = [(21, 22)]
N_to_B[22] = [(22, 24)]
N_to_B[23] = [(23, 24)]
N_to_B[24] = [(24, 25)]
N_to_B[25] = [(25, 26), (25, 27)]
N_to_B[26] = []
N_to_B[27] = [(27, 29), (27, 28), (27, 30)]
N_to_B[28] = []
N_to_B[29] = [(29, 30)]
N_to_B[30] = []


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# When the power P is a positive value, "B_to_N" indicates that the input power from the branch to the node. 
B_to_N = {}

B_to_N[1] = []
B_to_N[2] = [(1, 2)]
B_to_N[3] = [(1, 3)]
B_to_N[4] = [(2, 4), (3, 4)]
B_to_N[5] = [(2, 5)]
B_to_N[6] = [(2, 6), (4, 6)]
B_to_N[7] = [(5, 7), (6, 7)]
B_to_N[8] = [(6, 8)]
B_to_N[9] = [(6, 9)]
B_to_N[10] = [(6, 10), (9, 10)]
B_to_N[11] = [(9, 11)]
B_to_N[12] = [(4, 12)]
B_to_N[13] = [(12, 13)]
B_to_N[14] = [(12, 14)]
B_to_N[15] = [(12, 15), (14, 15)]
B_to_N[16] = [(12, 16)]
B_to_N[17] = [(16, 17), (10, 17)]
B_to_N[18] = [(15, 18)]
B_to_N[19] = [(18, 19)]
B_to_N[20] = [(10, 20), (19, 20)]
B_to_N[21] = [(10, 21)]
B_to_N[22] = [(21, 22), (10, 22)]
B_to_N[23] = [(15, 23)]
B_to_N[24] = [(23, 24), (22, 24)]
B_to_N[25] = [(24, 25)]
B_to_N[26] = [(25, 26)]
B_to_N[27] = [(25, 27)]
B_to_N[28] = [(6, 28), (8, 28), (27, 28)]
B_to_N[29] = [(27, 29)]
B_to_N[30] = [(27, 30), (29, 30)]


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# parameter assignment
N_g = 5                        # the number of thermal power generation   
N_t = 8760                    # the number of dispatch period
N_n = 30                       # the number of node
# 10.1109/ACCESS.2019.2932461
Penalty_load_curt = 1000000       # the penalty of curtailment load
Penalty_PV_curt = 20           # the penalty of curtailment PV power
Sbase = 100                    # Base power (100 MVA)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# parameters of the battery 
eta_charge = 0.95              # charging efficiency
eta_discharge = 0.9            # discharging efficiency

SOC_ini = np.zeros(30)         # the initial value of SOC
SOC_max = np.zeros(30)         # the maximum value of SOC
SOC_min = np.zeros(30)         # the minimum value of SOC


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# parameters of PV
PV = np.zeros(30)              # PV


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# node assignment
# two nodes (19 and 21) have batteries installed
# node 19
SOC_ini[18] = 200
SOC_max[18] = 200
SOC_min[18] = 20

# node 21
SOC_ini[20] = 200
SOC_max[20] = 200 
SOC_min[20] = 20

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# three nodes (8, 19, and 21) have PVs installed
PV[15] = 20                  # node 16
PV[20] = 20                  # node 21
PV[22] = 20                  # node 23
# PV[6] = 200                   # node 7
# PV[2] = 200                   # node 3
# PV[4] = 200                   # node 5


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# source-load data
electric_load = pd.read_csv("C:/WWT/论文/会议论文/code/electric_load.csv", index_col=0)
# d_load_T = electric_load['pu_load'].values

# d_load = electric_load['2020-12-01':'2020-12-31']['pu_load'].values
d_load = electric_load['pu_load'].values

# # daily electric load curve
# d_load_MW = np.array([1775.835, 1669.815, 1590.3, 1563.795, 1563.795, 1590.3, 1961.37, 2279.43, 2517.975,  2544.48, 2544.48, 2517.975,
#               2517.975, 2517.975, 2464.965, 2464.965, 2623.995, 2650.5, 2650.5, 2544.48, 2411.955, 2199.915,  1934.865, 1669.815])
# d_load = d_load_MW /  2650.5
# # It should be extended to one year
# d_load = np.tile(d_load, 365)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load solar forecasts.
# load data
power_PV_Jacumba = pd.read_csv("C:/WWT/论文/黄老师特刊/supplement material/data/Results.csv", index_col=0)
NSRDB_time = pd.date_range(start='2019-12-31 15:30:00', end='2020-12-31 14:30:00', freq='1h', tz='Etc/GMT+8')
power_PV_Jacumba.insert(9, "NSRDB_time", NSRDB_time)
power_PV_Jacumba = power_PV_Jacumba.fillna(0)
forecasts_data_target = power_PV_Jacumba[['MC_median_neg_two']]
forecasts_data_target.index = power_PV_Jacumba['NSRDB_time']

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# The proposed forecasting method: NWP GHI forecasts + model chain => PV power forecasts
# one year (2020)
d_solar_MW = forecasts_data_target['2020-01-01':'2020-12-30']['MC_median_neg_two']
d_solar_MW.loc[d_solar_MW > 20] = 20
d_solar_MW.loc[d_solar_MW <  0] = 0
# convert dataframe to array
d_solar_MW = d_solar_MW.values
d_solar = d_solar_MW / 20


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# declare variable
P_TG        = M.addVars(N_g, N_t, vtype = grb.GRB.CONTINUOUS, lb=0, name='Power_of_thermal_generator')              # thermal generator
SOC         = M.addVars(N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=0, name='SOC')                                     # capacity of battery (SOC)
P_charge    = M.addVars(N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=0, name='charging_power')                          # charging power
P_discharge = M.addVars(N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=0, name='discharging_power')                       # discharging power
B_charge    = M.addVars(N_n, N_t, vtype = grb.GRB.BINARY, name='state_charging')                                    # state charging
B_discharge = M.addVars(N_n, N_t, vtype = grb.GRB.BINARY, name='state_discharging')                                 # state discharging
P_PV        = M.addVars(N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=0, name='PV_power')                                # PV consumption
P_PV_curt   = M.addVars(N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=0, name='PV_curtailment')                          # curtailment PV
P           = M.addVars(N_n, N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=-10, ub=10, name='transmission_power')        # transmission power
delta       = M.addVars(N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=-1.57, ub=1.57, name='phase_angle')                # phase angle
Load_curt   = M.addVars(N_n, N_t, vtype = grb.GRB.CONTINUOUS, lb=0, name='load_curtailment')                        # load curtailment
 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# constraints

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# thermal generator power limits
M.addConstrs((P_TG[g,t] <= structure["thermal_generator"][np.where(structure["thermal_generator"][:, 0] == g+1)][0,2] / Sbase for t in range(N_t) for g in range(N_g)), 'Pg_max')
M.addConstrs((P_TG[g,t] >= structure["thermal_generator"][np.where(structure["thermal_generator"][:, 0] == g+1)][0,3] / Sbase for t in range(N_t) for g in range(N_g)), 'Pg_min')
# thermal generator ramp up
M.addConstrs((P_TG[g,t+1] - P_TG[g,t] <= structure["thermal_generator"][np.where(structure["thermal_generator"][:,0] == g+1)][0,4] / Sbase for t in range(N_t-1) for g in range(N_g)), 'RU')
# thermal generator ramp down
M.addConstrs((P_TG[g,t] - P_TG[g,t+1] <= structure["thermal_generator"][np.where(structure["thermal_generator"][:,0] == g+1)][0,5] / Sbase for t in range(N_t-1) for g in range(N_g)), 'RD')


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# power balance
# P_TG + Load_curtailment + P_PV - Load + P_discharge - P_charge == Total_node_output_power - Total_node_input_power
for i in range(N_n):
    if np.where(structure["thermal_generator"][:,1] == i+1)[0].size >= 1:  # np.where(condition): used to find the index that satisfies the (condition)
        # nodes with one thermal generator
        for t in range(N_t):
            M.addConstr(P_TG[structure["thermal_generator"][np.where(structure["thermal_generator"][:,1] == i+1)][0,0] - 1, t] + Load_curt[i, t] + P_PV[i, t] -
                             d_load[t] * structure['bus'][np.where(structure["bus"][:,0] == i+1)][0,1] / Sbase +
                             P_discharge[i,t] - P_charge[i,t] == sum(P[ii - 1, ij - 1, t] for (ii, ij) in N_to_B[i + 1]) -
                             sum(P[ji - 1, jj - 1, t] for (ji, jj) in B_to_N[i + 1]))
            
    else:    # nodes with none thermal generator
        for t in range(N_t):
            M.addConstr(Load_curt[i, t] + P_PV[i, t] - d_load[t] * structure['bus'][np.where(structure["bus"][:, 0] == i + 1)][0, 1] / Sbase +
                         P_discharge[i, t] - P_charge[i, t] == sum(P[ii - 1, ij - 1, t] for (ii, ij) in N_to_B[i + 1]) -
                         sum(P[ji - 1, jj - 1, t] for (ji, jj) in B_to_N[i + 1]))




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DC power flow assumptions
# 1. Line resisteances are negligible compared to line reactances (R_L << X_L, R_L \approx 0). 
#    This assumption implies that grid losses are neglected and line parameters are simplified.
# 2. The voltage profile is flat, meaning that the voltage amplitude is equal for all nodes. (|V_i| = |V_j| \approx 1 p.u.)   
# 3. Voltage angle differences between neighboring nodes are small. (\delta_{ij} \approx 0, sin(\delta_{ij}) \approx \delta_{ij}, cos(\delta_{ij}) \approx 1) 
for t in range(N_t):
    M.addConstr(delta[1 - 1, t] == 0) # the slack bus of the system is node 1.
    for i in range(N_n):
        for j in range(N_n):
            x = np.where(structure["branch"][:, 0] == i+1)
            y = np.where(structure["branch"][:, 1] == j+1)
            if np.intersect1d(x, y).size != 0:   # np.intersect1d: find the same value in two arrays (if they are the same, return True, indicating that the two nodes are connected)
                M.addConstr((P[i, j, t] == (delta[i, t] - delta[j, t]) / structure["branch"][np.intersect1d(x, y)][0, 2])) # branch power expression
                M.addConstr((P[i, j, t] <= structure["branch"][np.intersect1d(x, y)][0, 3] / Sbase))  # the upper bound of branch power
                M.addConstr((P[i, j, t] >= -structure["branch"][np.intersect1d(x, y)][0, 3] / Sbase)) # the lower bound of branch power
            else:
                M.addConstr((P[i, j, t] + P[j, i, t] == 0))   # when the nodes are not connected, the branch power is 0. 




#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# curtailment load
M.addConstrs(((Load_curt[i, t] <= d_load[t]*structure['bus'][np.where(structure["bus"][:,0] == i+1)][0,1] / Sbase) for i in range(N_n) for t in range(N_t)), 'Load_curt')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# curtailment PV power
M.addConstrs((P_PV_curt[i,t] == d_solar[t] * PV[i] / Sbase - P_PV[i, t] for i in range(N_n) for t in range(N_t)), 'Solar_curt')
M.addConstrs((P_PV[i,t] <= d_solar[t] * PV[i] / Sbase for i in range(N_n) for t in range(N_t)), 'PV')

 
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# battery constraints

M.addConstrs((B_charge[i,t] + B_discharge[i,t] <= 1 for i in range(N_n) for t in range(N_t)), 'charge_and_discharge_binary_state')

M.addConstrs((P_charge[i,t] <= 0.2 * B_charge[i,t] * SOC_max[i] / Sbase for i in range(N_n) for t in range(N_t)), 'charging_power')
M.addConstrs((P_discharge[i,t] <= 0.2 * B_discharge[i,t] * SOC_max[i] / Sbase for i in range(N_n) for t in range(N_t)), 'discharging_power')

# M.addConstrs((P_charge[i,t] <= 0.2 * SOC_max[i] / Sbase for i in range(N_n) for t in range(N_t)), 'charging_power')
# M.addConstrs((P_discharge[i,t] <= 0.2 * SOC_max[i] / Sbase for i in range(N_n) for t in range(N_t)), 'discharging_power')
 
M.addConstrs((SOC[i, t] <= SOC_max[i] / Sbase for i in range(N_n) for t in range(N_t)), 'SOC_MAX')
M.addConstrs((SOC[i, t] >= SOC_min[i] / Sbase for i in range(N_n) for t in range(N_t)), 'SOC_MIN')
 
M.addConstrs((SOC[i, 0]  == SOC_ini[i] / Sbase + (P_charge[i, 0] * eta_charge - P_discharge[i, 0] / eta_discharge) for i in range(N_n)), name='SOC_1')
M.addConstrs((SOC[i, t]  == SOC[i, t-1] + (P_charge[i, t] * eta_charge - P_discharge[i, t] / eta_discharge) for i in range(N_n) for t in range(1, N_t)), 'SOC') # SOC indicates energy
M.addConstrs((SOC[i, 23] == SOC_ini[i] / Sbase for i in range(N_n)), name='SOC_24')
 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# objective function
M.setObjective((sum(structure["thermal_generator"][np.where(structure["thermal_generator"][:, 0] == g + 1)][0, 6] * P_TG[g, t] * P_TG[g, t] + 
                    structure["thermal_generator"][np.where(structure["thermal_generator"][:, 0] == g + 1)][0, 7] * P_TG[g, t] +
                    structure["thermal_generator"][np.where(structure["thermal_generator"][:, 0] == g + 1)][0, 8] for t in range(N_t) for g in range(N_g)) * Sbase +
               Penalty_load_curt * sum(Load_curt[i, t]for i in range(N_n) for t in range(N_t)) * Sbase +
               Penalty_PV_curt * sum(P_PV_curt[i, t] for i in range(N_n) for t in range(N_t)) * Sbase), grb.GRB.MINIMIZE)


# M.setObjective((sum(structure["thermal_generator"][np.where(structure["thermal_generator"][:, 0] == g + 1)][0, 6] * P_TG[g, t] for t in range(N_t) for g in range(N_g)) * Sbase +
#                Penalty_load_curt * sum(Load_curt[i, t]for i in range(N_n) for t in range(N_t)) * Sbase +
#                Penalty_PV_curt * sum(P_PV_curt[i, t] for i in range(N_n) for t in range(N_t)) * Sbase), grb.GRB.MINIMIZE)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# slove
M.Params.MIPGap = 0.005    # 0.5%
# M.Params.TimeLimit = 300  # 5 minutes
M.optimize()
# output results
print('Obj=', M.objVal)
M.write('M.lp')


# record results
battery_SOC  = np.zeros((2,  N_t+1))
source_P_PV  = np.zeros((3,  N_t+1))
source_P_TG  = np.zeros((5, N_t+1))
battery_P_c  = np.zeros((2,  N_t+1))
battery_P_d  = np.zeros((2,  N_t+1))
curtail_load = np.zeros((30, N_t+1))
curtail_PV   = np.zeros((3,  N_t+1))
P_ij = np.zeros((30,30, N_t+1))




for t in range(1,8761):
    battery_SOC[0, t] = SOC[18,t-1].X            * Sbase
    battery_SOC[1, t] = SOC[20,t-1].X            * Sbase
    battery_P_c[0, t] = P_charge[18, t-1].X      * Sbase
    battery_P_d[0, t] = P_discharge[18, t-1].X   * Sbase
    battery_P_c[1, t] = P_charge[20, t-1].X      * Sbase
    battery_P_d[1, t] = P_discharge[20, t-1].X   * Sbase
    source_P_PV[0, t] = P_PV[15, t-1].X          * Sbase
    source_P_PV[1, t] = P_PV[20, t-1].X          * Sbase
    source_P_PV[2, t] = P_PV[22, t-1].X          * Sbase

    curtail_PV[0, t] = P_PV_curt[15, t-1].X      * Sbase
    curtail_PV[1, t] = P_PV_curt[20, t-1].X      * Sbase
    curtail_PV[2, t] = P_PV_curt[22, t-1].X      * Sbase

    for i in range(5):
        source_P_TG[i, t] = P_TG[i, t-1].X       * Sbase    
    for i in range(30):
        curtail_load[i, t] = Load_curt[i, t-1].X * Sbase


total_penalty_load = np.sum(curtail_load) * Penalty_load_curt
total_penalty_PV = np.sum(curtail_PV) * Penalty_PV_curt

P_ij = np.zeros((30,30, N_t))
for t in range(8760):
    for i in range(30):
        for j in range(30):
            P_ij[i,j,t] = P[i,j,t].X * Sbase

# BUS21 21→2 10→21
# P_21_22_out = P_ij[21-1, 22-1, :]
# P_10_21_in = P_ij[10-1, 21-1, :]





# reserve:
# Load
total_load_30 = d_load*(21.7+2.4+7.6+94.2+22.8+30+5.8+11.2+6.2+8.2+3.5+9+3.2+2.2+17.5+3.2+8.7+3.5+2.4+10.6)
# dispatched solar
total_dispatched_PV = np.sum(source_P_PV, axis=0)
total_dispatched_PV = total_dispatched_PV[1:]
# dispatched charging
battery_charging = battery_P_c[:,1:]
total_dispatched_charging = np.sum(battery_charging, axis=0)
# dispatched discharging
battery_discharging = battery_P_d[:,1:]
total_dispatched_discharging = np.sum(battery_discharging, axis=0)
# actual solar
actual_solar_MW = power_PV_Jacumba[['SAM_gen']]
actual_solar_MW.index = power_PV_Jacumba['NSRDB_time']
actual_solar_MW = actual_solar_MW['2020-01-01':'2020-12-30']['SAM_gen']
actual_solar_MW = actual_solar_MW.values
actual_solar = actual_solar_MW*3


# dispatched netload
dispatched_netload = total_load_30 - total_dispatched_PV #+ total_dispatched_charging - total_dispatched_discharging
# actal netload
actual_netload = total_load_30 - actual_solar


# error
e = dispatched_netload - actual_netload
# array2dataframe
error = pd.DataFrame(e, columns=['error'])
error.index = pd.date_range(start='2020-01-01 00:00:00', end='2020-12-30 23:00:00', freq='1h',tz='UTC') 

# find error<0
under_forecast = error.where(error < 0)
# average resample daily
under_forecast_nan = under_forecast.dropna() 
under_forecast_average = under_forecast_nan.resample('1d').mean() 
# 80% quantile
under_forecast_average_abs = under_forecast_average.abs()
quantile_uf_80 = under_forecast_average_abs.quantile(0.8)



# find error>0
over_forecast = error.where(error > 0)
# average resample daily
over_forecast_nan = over_forecast.dropna()
over_forecast_average = over_forecast_nan.resample('1d').mean()
# 80% quantile
quantile_of_80 = over_forecast_average.quantile(0.8)

# underforecast economy $/MWh
average_reserve_cost = 79.95
uf_cost = average_reserve_cost*quantile_uf_80




























