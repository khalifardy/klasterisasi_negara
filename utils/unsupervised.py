"""
Kumpulan algoritma unsupervised learning
"""
import numpy as np
import seaborn as sns
from .matematika.distance import euclidian_distance,kuadrat_jarak


class Kmeans:

    def __init__(self,cluster:int=2,inital_centroid=None,limit_loop:int=300,konvergen=None,random_state:int=2):
        self.__cluster = cluster
        self.__inital_centroid=inital_centroid
        self.__limit_loop=limit_loop
        self.__historycentroid = {}
        self.__konvergen = konvergen
        self.__random_state = random_state
        self.__cluster_fit = None
        self.__inersia = None
        self.__x = None
        self.__label = None
    
    @property
    def inersia(self):
        #ini hanya sebuah dekorator
        pass

    @property
    def label(self):
        #inin hanya sbeuah dekorator
        pass
    
    def sethistory_centroid(self,loop,list_centroid):
        self.__historycentroid[loop]=list_centroid
    
    def history_centroid(self)->dict:
        return self.__historycentroid
    
    def fit_predict(self,x:np.ndarray)->np.ndarray:
        self.__x = x
        self.fit(x)
        predict = [None for _ in range(len(x))]
        for key3 in self.__cluster_fit.keys():
            for index in self.__cluster_fit[key3]:
                predict[index] = key3
        self.__label = np.array(predict)
        return np.array(predict)

    def fit(self,x:np.ndarray):
        self.__x = x
        np.random.seed(self.__random_state)
        if self.__inital_centroid== None:
            self.__inital_centroid= {}
            for i in range(self.__cluster):
                #centroid = np.random.uniform(x.min(),high = x.max(),size=len(x[0]) )
                centroid = x[np.random.choice(x.shape[0], 1, replace=False)]

                self.__inital_centroid[i] = list(centroid)        
        loop = 0 
        konv = 0
        while loop < self.__limit_loop:
            cluster={}
            for i in range(self.__cluster):
                cluster[i]=[]
            self.sethistory_centroid(loop,self.__inital_centroid)
            for index,value in enumerate(x):
                cls = None
                jrk = None
                for key in self.__inital_centroid.keys():
                    jarak = euclidian_distance(value,np.array(self.__inital_centroid[key]))
                    if jrk == None or jarak <= jrk:
                        jrk = jarak
                        cls = key
                cluster[cls].append(index)
            update_centroid = {}

            for key2 in cluster.keys():
                if len(cluster[key2])!=0:
                    init = x[cluster[key2][0]].copy()
                    for rec in cluster[key2][1:]:
                        init += x[rec].copy()
                    init = init/len(cluster[key2])
                else:
                    init = list( x[np.random.choice(x.shape[0], 1, replace=False)])

                update_centroid[key2] = list(init)
            
            if konv == self.__konvergen:
                break

            if update_centroid == self.__inital_centroid:
                konv +=1
            else:
                self.__inital_centroid = update_centroid
            loop += 1
        
        self.__cluster_fit = cluster

    @inersia.getter
    def inersia_(self):
        total = 0
        
        for key in self.__cluster_fit.keys():
            centroid = self.__inital_centroid[key]
            for rec in self.__cluster_fit[key]:
                total += kuadrat_jarak(self.__x[rec],np.array(centroid))
        self.__inersia = total
        return total

    @label.getter
    def label__(self):
        return self.__label 



           
                

            

            




    

    
