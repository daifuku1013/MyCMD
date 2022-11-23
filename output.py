"""
init.py
- initialize simulation cell (from geo.in)
- initialize atom position   (from geo.in)
- initialize atom velocity   (from geo.in/randomize)
"""
# Zhao-Yang Liu, Tsinghua University, 2022.11
# My first try at molecular dynamics (❁´ω`❁)
# This is the output module!


import json
import datetime


#-------------------------------------param-------------------------------------
with open('input.json','r') as f:
    input_dict = json.load(f)               #从构型文件中读入参数
rdf_start = input_dict['rdf_start']         #开始计算RDF的步数
#-------------------------------------------------------------------------------


def output_runlog(j, dt, results):
    """
    输出该时刻总势能，总动能，温度，总能量
    """
    if j == 0:
        with open('run.log','a+') as f:
            f.writelines('%'+'date '+ str(datetime.datetime.now()).split('.')[0].replace(":", "-")+'\n')
            f.writelines(f'%step time(ps) total_Eng(eV) potential_Eng(eV) kinetic_Eng(eV) Temperature(K)\n')
    with open('run.log','a+') as f:
        f.writelines('{} {:.2f} {:.12f} {:.12f} {:.12f} {:.6f}'.format(j, j*dt, results[3], results[0], results[1], results[2])+'\n')

def output_rdf(j, rdf, rr):
    """
    输出该时刻径向分布函数
    """
    if j == rdf_start:
        with open('rdf.log','w') as f:
            f.writelines('%'+'date '+ str(datetime.datetime.now()).split('.')[0].replace(":", "-")+'\n')
            f.writelines(f'%time step={j}\n')
            for i in range(len(rdf)):
                f.writelines('{} {}\n'.format(rr[i], rdf[i]))
    else:
        with open('rdf.log','a+') as f:
            f.writelines(f'%time step={j}\n')
            for i in range(len(rdf)):
                f.writelines('{} {}\n'.format(rr[i], rdf[i]))

def output_plot(j,atom_position):
    """
    绘制该时刻粒子分布图
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
    plt.style.use('science')
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(atom_position)):
        ax.scatter(atom_position[i][0],atom_position[i][1],atom_position[i][2],marker='o',s=3,c='blue')
    # plt.show()
    plt.savefig(f'position-{j}.png')

# def output_anime():
#     import cv2
#     size = (640,480)#这个是图片的尺寸，一定要和要用的图片size一致
#     #完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
#     videowrite = cv2.VideoWriter(r'test.mp4',-1,10,size)#20是帧数，size是图片尺寸
#     img_array=[]
#     for filename in [r'{}.png'.format(2 * i) for i in range(50)]:#这个循环是为了读取所有要用的图片文件
#         img = cv2.imread(filename)
#         if img is None:
#             print(filename + " is error!")
#             continue
#         img_array.append(img)
#     for i in range(50):#把读取的图片文件写进去
#         videowrite.write(img_array[i])
#     videowrite.release()
#     print('end!')