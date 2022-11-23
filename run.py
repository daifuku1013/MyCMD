"""
run.py
a classical Molecular Dynamics simulation program
based on verlet list method, LJ potentials and velocity verlet algorithm
"""
# Zhao-Yang Liu, Tsinghua University, 2022.11
# My first try at molecular dynamics (❁´ω`❁)
# This is the main running module!


import json
import time
import scipy.constants as cn

from init import initiate_cell, initiate_atom
from calc import build_neighbor_list, calculate_force, update_position
from calc import calculate_potential_energy, calculate_kinetic_energy
from calc import Berendson_thermostat, calculate_rdf
from output import output_runlog, output_rdf, output_plot


#-------------------------------------param-------------------------------------
with open('input.json','r') as f:
    input_dict = json.load(f)               #从构型文件中读入参数
natom = input_dict['natom']                 #原子数量
atom_mass = input_dict['gmol']              #原子质量,g/mol
neighbor_n = input_dict['neighbor_n']       #近邻原子最大数量
rcut = input_dict['rcut']                   #LJ势的截断半径
neighbor_r = input_dict['neighbor_r']       #近邻表的额外截断半径
epsilon = input_dict['epsilon']             #LJ势的参数
sigma = input_dict['sigma']                 #LJ势的参数
dt = input_dict['time_step']                #模拟时间步长,ps
time_steps = input_dict['total_time_step']  #总模拟步数
steps_neighbor = input_dict['nstep_search'] #更新近邻表的步数
steps_output = input_dict['nstep_out']      #输出结果的步数
ensemble = input_dict['ensemble']           #系综
T0 = input_dict['initial_temperature']      #初始温度
tau = input_dict['tau']                     #Berendson热库控制温度耦合快慢的时间参数,ps
Tc = input_dict['constant_temperature']     #系综恒定温度
cal_rdf = input_dict['cal_rdf']             #是否计算径向分布函数(RDF)
rdf_rcut = input_dict['rdf_rcut']           #RDF最大半径
rdf_dr = input_dict['rdf_dr']               #RDF微元球壳厚度
rdf_start = input_dict['rdf_start']         #开始计算RDF的步数
rdf_stop = input_dict['rdf_stop']           #结束计算RDF的步数
rdf_interval = input_dict['rdf_interval']   #每几步计算rdf来取平均
#-------------------------------------------------------------------------------


if __name__ == "__main__":
    """
    - velocity verlet alogrithm
    * r(t+dt)=r(t)+v(t)dt+f(t)/2m*dt^2
    * v(t+dt)=v(t)+[f(t+dt)+f(t)]/2m*dt
    """
    print('---test---')
    time1 = time.time()

    for j in range(time_steps+1):   #主循环
        if j == 0:  #初始化（第0步）
            cell_size = initiate_cell() #初始化模拟区域
            atom_position, atom_velocity = initiate_atom()  #初始化原子坐标&速度
            nlist, list_atom = build_neighbor_list(natom, neighbor_n, rcut, neighbor_r, cell_size, atom_position)   #构建近邻表
            atom_force = calculate_force(natom, atom_position, atom_velocity, nlist, list_atom) #计算原子受力
            output_plot(j,atom_position)   #绘制原子的坐标分布
        else:
            if j % steps_neighbor == 0:     #每隔一定步数更新近邻表
                nlist, list_atom = build_neighbor_list(natom, neighbor_n, rcut, neighbor_r, cell_size, atom_position)
            if j > 1 and ensemble == 'NVT': #对于NVT系综，在两步MD之间施加Berendson热库
                atom_velocity = Berendson_thermostat(atom_velocity, Temperature)
            atom_position = update_position(cell_size, dt, atom_mass, atom_position, atom_velocity, atom_force)    #S2：计算坐标r(t+dt)并更新
            atom_velocity += (atom_force * cn.eV / cn.angstrom) / (2 * atom_mass * 1e-3 / cn.N_A) * (dt * cn.pico) / cn.angstrom * cn.pico  #S3:计算速度v(t+dt)第一步
            atom_force = calculate_force(natom, atom_position, atom_velocity, nlist, list_atom)     #S4:计算受力f(t+dt)并更新
            atom_velocity += (atom_force * cn.eV / cn.angstrom) / (2 * atom_mass * 1e-3 / cn.N_A) * (dt * cn.pico) / cn.angstrom * cn.pico  #S5:计算速度v(t+dt)并更新
            if cal_rdf and (j+1) >= rdf_start:
                rr, rdf = calculate_rdf(atom_position, nlist, list_atom, cell_size)
                if (j-rdf_start) % rdf_interval == 0:
                    output_rdf(j, rdf, rr)  #输出rdf

        potential_energy = calculate_potential_energy(natom, rcut, epsilon, sigma, atom_position, nlist, list_atom) #计算系统势能
        kinetic_energy, Temperature = calculate_kinetic_energy(natom, atom_mass, atom_velocity) #计算系统动能与温度
        total_energy = potential_energy + kinetic_energy    #计算系统总能量

        if j % steps_output == 0:   #每隔一定步数输出结果
            output_runlog(j, dt, [potential_energy, kinetic_energy, Temperature, total_energy])
            print(f'第{j}步, t={j*dt:.2f}ps, E={total_energy:.2f}eV, T={Temperature:.2f}K')

    time2 = time.time()
    print(f'总耗时{time2-time1:.2f}s')
    print('---over---')