# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 09:27:20 2018

@author: lx
"""

import numpy as np # linaly是关于线性代数的内容
from scipy.linalg import expm,rq
class Camera:
    def __init__(self,P):
        self.P = P  #P = K(R T)
        self.K = None #内部参数
        self.R = None #旋转矩阵
        self.t = None #平移矩阵
        self.c = None #物理视网膜原点的位置
    def project(self,X): 
        '''P是一个3*4的矩阵，X是一个4*n的矩阵，n表示需要计算投影点的数量。'''
        x = np.dot(self.P,X)#两个矩阵进行点乘,获取world中的点在image中的位置
        #print("X",X.shape,"x ",x[2].shape)
        for i in range(3):
            """-----------为什么只选取了x的前3个元素，为什么选取x[2]作为基准点？
            ----答：这里不是前三个元素，x[2]表示x的第三行元素"""
            x[i]/=x[2]     #x = PX/z
        return x 
    
    '''对映射矩阵P进行分解'''
    def factor(self):
        ''' Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
            and R upper triangular.'''
        self.k,self.R = rq(self.P[:,:3])
    def rotation_matrix(self,a):
        "创建一个绕向量a旋转的三维旋转矩阵---------不明白 expm到底做了什么工作"
        R = np.eye(4)
        R[:3,:3] = expm([[0,-a[2],a[1]],[a[2],0,a[0]],[-a[1],a[0],0]])
        return R