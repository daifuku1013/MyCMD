"""
init.py
- initialize simulation cell (from geo.in)
- initialize atom position   (from geo.in)
- initialize atom velocity   (from geo.in/randomize)
"""
# Zhao-Yang Liu, Tsinghua University, 2022.11
# My first try at molecular dynamics (❁´ω`❁)
# This is the initialization module!


import json
import numpy as np 
import pandas as pd 
import scipy.constants as cn


#-------------------------------------param-------------------------------------
with open('input.json','r') as f:
    input_dict = json.load(f)   #从文件中读入参数
natom = input_dict['natom']                 #原子数量
geo_dir = input_dict['geo_dir']             #构型文件路径
atom_mass = input_dict['gmol']              #原子质量, g/mol
T0 = input_dict['initial_temperature']      #初始温度, K
read_v = input_dict['read_v']               #是否直接从构型文件读入速度
#-------------------------------------------------------------------------------


def initiate_cell():
    """
    - initialize simulation cell (from geo.in)
    return cell_size: np.array((lx,ly,lz)), A
    """
    cell_param = pd.read_csv(geo_dir, sep=' ',skiprows=1, nrows=3, names=['x','y','z'])   #从构型文件开头读取模拟区域尺寸
    cell_size = np.array((cell_param.loc[0]['x'],cell_param.loc[1]['y'], cell_param.loc[2]['z']), dtype=np.float64)     #dataframe->np.array
    return cell_size

def initiate_atom():
    """
    - initialize atom position (from geo.in)
    - initialize atom velocity (from geo.in/randomize)
    return atom_position: np.array, size=natom*3, A
           atom_velocity: np.array, size=natom*3, A/ps
    """
    read_position = pd.read_csv(geo_dir,sep='\t',skiprows=6,nrows=natom, names=['atomic','x','y','z'])[['x','y','z']]  #从构型文件读取所有原子的坐标,A
    atom_position = np.array(read_position, dtype=np.float64)  #dataframe->np.array
    if read_v:  #直接从构型文件读入速度,A/ps
        read_velocity = pd.read_csv(geo_dir,sep='\t',skiprows=872, nrows=natom,names=['atomic','x','y','z'])[['x','y','z']]
        atom_velocity = np.array(read_velocity, dtype=np.float64)   #daraframe->np.array
    else:       #随机初始化速度，其总动能满足初始温度T0
        atom_velocity = np.random.rand(natom,3) - 0.5   #采用(-0.5,0.5)的均匀分布随机初始化每个原子的速度
        atom_velocity -= np.average(atom_velocity)      #先整体平移速度，使质心速度为0
        Ek = 0.5 * (atom_mass / 1e3 / cn.N_A) * np.sum(np.linalg.norm(atom_velocity,axis=1) ** 2)  #计算动能
        scaling_factor = 3 * natom / 2 * cn.k * T0 / Ek #放缩因子
        atom_velocity *= np.sqrt(scaling_factor)        #再整体缩放速度大小，使总动能与想要的温度匹配
        atom_velocity /= (cn.angstrom / cn.pico)        #从计算中的国际单位制转换为金属单位制,A/ps
    return atom_position, atom_velocity

def test_init():
    """
    - test of init.py
    """
    print('cell_size:\n',initiate_cell())
    print('atom position & atom velocity:\n', initiate_atom())


if __name__=='__main__':
    test_init()