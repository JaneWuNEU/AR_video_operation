# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 09:27:20 2018

@author: lx
"""
from Camera import Camera # python3需要从py文件中导入模块
import numpy as np
from matplotlib import pyplot as plot

ObjectData_path = "F:/project/dataset/oxford/house/3D/house.p3d"
#载入world points
points = np.loadtxt(ObjectData_path).T # 把原始的N*3转换为3*n
points = np.vstack((points,np.ones(points.shape[1])))#把非齐次向量转化为齐次向量，这一行的代码是在矩阵points下增加一行

P = np.hstack((np.eye(3),np.array([0,0,-10]).reshape(3,1)))#np.eye表示生成一个对角阵,直接设定了amera和world之间的旋转矩阵（R T）
cam = Camera(P)
x = cam.project(points)

#绘制投影
plot.figure()
plot.plot(x[0],x[1],"k.") #为甚只选择了前两个点
plot.show()
