# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:23:57 2018

@author: neu_1
"""
import sys
sys.path.append("F:/project/python")
from utils import WRFile
import numpy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
import random as ran
from sklearn.svm import SVC
class ClusterData:
    def __init__(self):
        self.wrFile = WRFile()
    '''
    timestamp                      second,new_timestamp
    ["2016-11-17 06:08:42.692"]===>[42,2016-11-17 06:08:42]
    ["2016-11-17 06:08:42"]===>[42,2016-11-17 06:08:42]
    '''
    def getSecond_timestamp(self,timestamp):
        loc = timestamp.rindex(":")
        try:
            dot_loc = timestamp.rindex(".")
            new_item = timestamp[:dot_loc]
            second = timestamp[loc+1:dot_loc]
        except ValueError:
            new_item = timestamp
            second = timestamp[loc+1:] 
        return {"second":second,"timestamp":new_item}
    
    def transformQtoXYZ(self,q_x,q_y,q_z,q_w):
        x = round(2*q_x*q_z+2*q_y*q_w,2)
        y = round(2*q_y*q_z-2*q_x*q_w,2)
        z = round(1-2*q_x*q_x - 2*q_y*q_y,2)
        return [x,y,z]
    
    def mergeDataInSeconds(self,num,video_num):
        path = "F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/"+str(num)+"/VD_1.xlsx"
        #readDataFromExcel(self,filePath,cols=1,sheet_name = "1",min_col =1,max_col = 1)
        #readDataFromExcel2(self,filePath,rows = 1,sheet_name = "1",min_row =1,max_row = 1):
        result = self.wrFile.readDataFromExcel2(path,rows = 2,sheet_name = "Sheet",max_row = 0)
        #Timestamp	PlaybackTime	UnitQuaternion.x	UnitQuaternion.y	UnitQuaternion.z	UnitQuaternion.w	HmdPosition.x	HmdPosition.y	HmdPosition.z  
        max_rows = len(result) #最大行-1
        min_rows = 1
        count = 1
        sum_q_x = float(result[1][2])
        sum_q_y = float(result[1][3])
        sum_q_z = float(result[1][4])
        sum_q_w = float(result[1][5])
        
        time_xyz = []
    
        for r in range(min_rows+1,max_rows):
            item = result[r]
            item = numpy.array(item[2:], dtype="float64")
            current_time = self.getSecond_timestamp(result[r][0])
            last_time = self.getSecond_timestamp(result[r-1][0])
            if(current_time["second"]==last_time["second"]):
                #print("same")
                sum_q_x = sum_q_x+item[0]
                sum_q_y = sum_q_y+item[1]
                sum_q_z = sum_q_z+item[2]
                sum_q_w = sum_q_w+item[3]
                count = count+1
            else:
                
                average_q_x = round(sum_q_x/count,2)
                average_q_y = round(sum_q_y/count,2)
                average_q_z = round(sum_q_z/count,2)
                average_q_w = round(sum_q_w/count,2)
                xyz = self.transformQtoXYZ(average_q_x,average_q_y,average_q_z,average_q_w)
                #print(xyz)
                count = 1
                sum_q_x = item[0]
                sum_q_y = item[1]
                sum_q_z = item[2]
                sum_q_w = item[3]
                temp = [current_time["timestamp"]]
                temp.extend(xyz)
                time_xyz.append(temp)
                
        #结算最后一秒
        average_q_x = round(sum_q_x/count,2)
        average_q_y = round(sum_q_y/count,2)
        average_q_z = round(sum_q_z/count,2)
        average_q_w = round(sum_q_w/count,2)
        xyz = self.transformQtoXYZ(average_q_x,average_q_y,average_q_z,average_q_w)
        timestamp =[self.getSecond_timestamp(result[max_rows-1][0])["timestamp"]]
        timestamp.extend(xyz)
        time_xyz.append(timestamp)
        #(self,data,filePath,describe ="no description",cols = 0,sheet_name = "1")#
        rows = self.wrFile.writeDataIntoExcel(data = time_xyz,filePath ="F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/"+str(num)+"/u"+str(num)+"_v"+str(video_num)+"_second.xlsx" )
        return rows  
    '''
    data = 48*204*3 
     48 stands for the total number of users
     204 represents there are 204 seconds of this video
     3 describe 3 dimension data needs to be recorded
    
    '''
    def mergeDataFromUsers(self,video):
        data = numpy.zeros((48,204,3))
        #print(data)
        '''=====store the data of users in a multi-dimension matrix Data====='''
        for user in range(1,48+1):
            filePath = "F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/"+str(user)+"/u"+str(user)+"_v"+str(video)+"_second.xlsx" #u1_v1_second.xlsx
            
            user_data = numpy.array(self.wrFile.readDataFromExcel2(filePath))
            #print(user_data.shape)
            user_rows = len(user_data)
            row_standard = 204
            if(user_rows>=row_standard):
                data[user-1,0:row_standard] = user_data[:row_standard,1:]
            else:#user_rows<row_standard needs to be added some rows
                data[user-1,:user_rows] = user_data[:,1:]
                for i in range(1,row_standard-user_rows+1):
                    data[user-1,user_rows-1+i,:] = user_data[user_rows-1,1:]
        '''=====================================''' 
        #print(data[18,:,:])         
        second_data = data[:,0,:]
        for r in range(1,204):
            #if r==18:
                #print(data[:,r,:])
            second_data = numpy.hstack((second_data,data[:,r,:]))
        print(second_data.shape)
        self.wrFile.writeDataIntoExcel(data = second_data,filePath = "F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/second.xlsx")
   
    def predictMotion():
        pass
    def getClassifer(self,Bmax):
        #1.randomly select 43 samples of the data, and 
        #2.depend on samples to train classfier
        filePath = "F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/cluster_data.xlsx"
        data = numpy.array(self.wrFile.readDataFromExcel2(filePath = filePath,max_row  = 48 ))
        accuracy = numpy.zeros(203).reshape((203,1))
        for col in range(0,203):# 应该从后往前选
            X = data[:,col*4:(col+1)*4]
            if col>=199:
                label = numpy.array(data[:,col*4+3+1*4]).reshape((48,1))
            else:
                label = numpy.array(data[:,col*4+3+Bmax*4]).reshape((48,1))
            # get samples
            whole_index = range(0,48)
            sample_index = ran.sample(whole_index,43)
            test_index = list(set(whole_index)-set(sample_index))            
            sample_data = X[sample_index][:,:3]
            sample_label = label[sample_index]
            
            test_data = X[test_index][:,:3]
            test_label = label[test_index]
            
            #carry out training
            try:
              clf = SVC(gamma = 'auto')
              clf.fit(sample_data,sample_label)
              accuracy_temp = clf.score(test_data,test_label)
              accuracy[col] = accuracy_temp
            except ValueError:
                accuracy[col] = 1
        #print(accuracy.shape)
        filePath = "F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/prediction_accuracy.xlsx"
        self.wrFile.writeDataIntoExcel(data = accuracy,filePath = filePath)
        
            

            
        
   
    
    def clusterData(self,filePath):
        data = numpy.array(self.wrFile.readDataFromExcel2(filePath = filePath,max_row  = 48 ))
        #X = data[:,:3]
        #cluster_result = numpy.zeros((204,3))
        label_result = None
        for col in range(0,204):
            X = data[:,col*3:(col+1)*3]
            #samples = numpy.sample(range(0,48),43)
            #test = set(range(0,48))-set(samples)
            
            # 计算向量间的欧氏距离
            #print(" col ",col)
            #dist =  DistanceMetric.get_metric('euclidean')
            #result = dist.pairwise(X)
            #print(result)
            
            # 进行聚类
            eps = 0.6
            clustering = DBSCAN(eps=eps,min_samples=5).fit(X)        
            labels = numpy.array(clustering.labels_).reshape((48,1))
            #print(clustering.labels_)
            
            # 记录聚类结果
            X = numpy.hstack((X,labels))  
            if col==0:
                label_result = X
            else:
                label_result = numpy.hstack((label_result,X))
        filePath = "F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/cluster_statistic.xlsx"
        #self.wrFile.writeDataIntoExcel(data = label_result,filePath = filePath)
        #print(label_result.shape)
        #print(label_result[:,:12])
        #return cluster_pointer    
        
cluster = ClusterData()
num = 1
rows_record = []
filePath = "F:\\project\\dataset\\vr\\Formated_Data\\Experiment_1/second.xlsx"
cluster.getClassifer(Bmax = 5)
#cluster.clusterData(filePath)

