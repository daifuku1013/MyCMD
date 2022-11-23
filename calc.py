"""
calc.py
- build up neighbor list
- calculate force and energy
- Berendson thermostat for NVT
- calculate RDF
"""
# Zhao-Yang Liu, Tsinghua University, 2022.11
# My first try at molecular dynamics (❁´ω`❁)
# This is the calculation module!


import json
import numpy as np 
import scipy.constants as cn
from numba import njit

from init import initiate_cell


#-------------------------------------param-------------------------------------
with open('input.json','r') as f:
    input_dict = json.load(f)   #从文件中读入参数
natom = input_dict['natom']                 #原子数量
atom_mass = input_dict['gmol']              #原子质量,g/mol
neighbor_n = input_dict['neighbor_n']       #近邻原子最大数量
rcut = input_dict['rcut']                   #LJ势的截断半径
neighbor_r = input_dict['neighbor_r']       #近邻表的额外截断半径
epsilon = input_dict['epsilon']             #LJ势的参数
sigma = input_dict['sigma']                 #LJ势的参数
dt = input_dict['time_step']                #模拟时间步长,ps
T0 = input_dict['initial_temperature']      #初始温度
tau = input_dict['tau']                     #Berendson热库控制温度耦合快慢的时间参数,ps
Tc = input_dict['constant_temperature']     #系综恒定温度
rdf_rcut = input_dict['rdf_rcut']           #RDF最大半径
rdf_dr = input_dict['rdf_dr']               #RDF微元球壳厚度

cell_size = initiate_cell()                 #模拟区域的尺寸
#-------------------------------------------------------------------------------


@njit
def in_range(r_ij, cell_size):
    """
    - periodic boundary condition (calculate the minimum distance between two atoms)
    return r_ij: np.array((dx,dy,dz)), A
           -lx/2 < dx < lx/2
           -ly/2 < dy < ly/2
           -lz/2 < dz < lz/2
    """
    for a, l in enumerate(cell_size):     #对x/y/z依次应用周期性边界条件
        r_ij[a] -= np.floor((r_ij[a] + l / 2) / l) * l
    return r_ij   #返回两原子间的最小距离与坐标差

@njit
def build_neighbor_list(natom, neighbor_n, rcut, neighbor_r, cell_size, atom_position):
    """
    - build up neighbor list by Verlet list method
    - return nlist: np.array, size=natom*1
             list_atom: np.array, size=natom*neighbor_n
    """
    nlist = np.zeros((natom), dtype=np.int32)                   #储存每个原子的近邻原子数量
    list_atom = np.zeros((natom, neighbor_n), dtype=np.int32)   #储存近邻原子编号的列表
    rcut = neighbor_r + rcut    #近邻表截断半径=LJ势截断半径+近邻表额外截断半径
    for i in range(natom):
        for j in range(i+1, natom):
            r_ij = atom_position[j] - atom_position[i]  #计算两原子距离
            r_ij = in_range(r_ij, cell_size)            #应用周期性边界条件
            if abs(r_ij[0]) > rcut:     #先排除一些明显不是近邻的组合
                continue
            elif abs(r_ij[1]) > rcut:   #同上
                continue
            elif abs(r_ij[2]) > rcut:   #同上上
                continue
            elif np.linalg.norm(r_ij) <= rcut:   #当两个原子间的距离小于近邻表截断半径时，认为两原子为近邻
                list_atom[i][nlist[i]] = j  #把j原子编号储存在i原子近邻表中
                list_atom[j][nlist[j]] = i  #把i原子编号储存在j原子近邻表中
                nlist[i] += 1               #i原子近邻原子数+1
                nlist[j] += 1               #j原子近邻原子数+1
    return nlist, list_atom

@njit
def calculate_force(natom, atom_position, atom_velocity, nlist, list_atom):
    """
    - calculate the atomic force in LJ force field
    * Lennard-Jones(LJ) potentials: V(r) = 4 * epsilon * [(sigma/r)^12-(sigma/r)^6]
    * From potential energy to force: F = -nabla V
    > I.Daan Frenkel，Understanding Molecular Simulations (1996) P67-69
    return atom_force: np.array, size=natom*3, eV/A
    """
    atom_force = np.zeros((natom, 3), dtype=np.float64) #储存每个原子受力
    for i in range(natom):  #遍历每个原子
        for j in range(nlist[i]):   #每个原子受力=其近邻原子与其相互作用力的累积
            jj = list_atom[i][j]    #读取近邻原子编号
            r_ij = atom_position[i] - atom_position[jj] #两原子坐标差
            r_ij = in_range(r_ij, cell_size)    #应用周期性边界条件
            if np.linalg.norm(r_ij) < rcut:     #如果两原子坐标差小于LJ势截断半径，则考虑相互作用力
                Cij = 4 * epsilon * (12 * (sigma / np.linalg.norm(r_ij)) ** 12 - 6 * (sigma / np.linalg.norm(r_ij)) ** 6) 
                atom_force[i] += Cij * r_ij / np.linalg.norm(r_ij) ** 2
    return atom_force

@njit
def calculate_potential_energy(natom, rcut, epsilon, sigma, atom_position, nlist, list_atom):
    """
    - calculate the total potential energy in LJ force field
    return total_energy: float, eV
    """ 
    potential_energy = np.zeros((natom, natom), dtype=np.float64)   #两两原子间的势能
    for i in range(natom):
        for j in range(nlist[i]):
            jj = list_atom[i][j]
            r_ij = atom_position[i] - atom_position[jj]
            r_ij = in_range(r_ij, cell_size)
            d = np.linalg.norm(r_ij)
            u_cut = 4 * epsilon * ((sigma / rcut) ** 12 - (sigma / rcut) ** 6)      #截断半径处的势能
            if d < rcut:
                u_ij = 4 * epsilon * ((sigma / d) ** 12 - (sigma / d) ** 6)     
                potential_energy[i][jj] = u_ij - u_cut
    total_energy = 0    #计算系统总势能
    for i in range(natom):
        for j in range(i+1, natom):
            total_energy += potential_energy[i][j]
    return total_energy

# @njit
def calculate_kinetic_energy(natom, atom_mass, atom_velocity):
    """
    - calculate the total kinetic energy and temperature
    return kinetic_energy: float, eV
           Temperature: float, K
    """
    kinetic_energy = 0.5 * (atom_mass / 1e3 / cn.N_A) * np.sum((np.linalg.norm(atom_velocity,axis=1) * cn.angstrom / cn.pico) ** 2) / cn.eV     # E_k=sum(1/2*m*v^2), eV
    #用np.linalg.norm()函数中的沿axis求模，numba就会不能识别，但np.sum()带来的速度增益更显著。因此这个函数不用numba加速
    Temperature = 2 * (kinetic_energy * cn.eV) / (3 * natom * cn.k )   #E_k=3nkT/2->T=2E_k/(3nk)
    return kinetic_energy, Temperature

@njit
def update_position(cell_size, dt, atom_mass, atom_position, atom_velocity, atom_force):
    """
    - update position from velocity and force
    * r(t+dt)=r(t)+v(t)dt+f(t)/2m*dt^2
    return atom_position: np.array, size=natom*3, A
           0 < x < lx
           0 < y < ly
           0 < z < lz
    """
    atom_position += atom_velocity * dt + (atom_force * cn.eV / cn.angstrom) / (2 * atom_mass * 1e-3 / cn.N_A) * (dt * cn.pico) ** 2 / cn.angstrom #r(t+dt)=r(t)+v(t)dt+f(t)/2m*dt^2
    for i in range(len(atom_position)):
        for a, l in enumerate(cell_size): #应用周期性边界条件
            atom_position[i][a] -= np.floor(atom_position[i][a] / l) * l
    return atom_position

@njit
def Berendson_thermostat(atom_velocity, Temperature):
    """
    - Berendson thermostat for NVT ensemble
    * Couple the current symstem temperature with the setting constant temperature 
    * and use the "scaling factor" tau to describe the speed of temperature change
    """
    scaling_factor = np.sqrt(1 + dt / tau * (Tc / Temperature - 1))   #计算速度缩放系数 
    return atom_velocity * scaling_factor   #对温度进行放缩

@njit
def calculate_rdf(atom_position, nlist, list_atom, cell_size):
    """
    - calculate RDF
    * Radial Distribution Function(RDF): relative probability of occurrence of other particles at specific distance
    * g(r) = n(r)/(rho*V) = n(r)/(4*pi*r^2*rho*dr)
    * rho = natom / V --- average particle density in the system
    return rr: np.array, A
           gr: np.array, size=len(rr)
    """
    rr = np.arange(rdf_dr,rdf_rcut+rdf_dr,rdf_dr)   #离散化径向坐标
    nr = np.zeros(len(rr),dtype=np.int32)  #储存相应球壳位置内的原子数
    gr = np.zeros(len(rr),dtype=np.float64)  #径向分布函数
    for i in range(natom):  #累积所有原子的径向分布函数
        for j in range(nlist[i]):
            jj = list_atom[i][j]
            r_ij = atom_position[i] - atom_position[jj]
            r_ij = in_range(r_ij, cell_size)
            num = int(np.linalg.norm(r_ij) / rdf_dr) #判断这个原子在第几层球壳里
            if num > 0 and num < len(rr): #从r>dr开始统计
                nr[num] += 1    #对应球壳内的粒子数增加1
    V = cell_size[0] * cell_size[1] * cell_size[2]  #体系总体积
    rho = natom / V
    gr = nr / (4 * cn.pi * rr ** 2 * rho * rdf_dr) / natom  #对所有原子的径向分布函数作平均
    return rr, gr

def test_calc():
    pass

if __name__=='__main__':
    test_calc()