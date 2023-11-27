import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Visualisasi:
    """
    class untuk visualisasi data
    dengan konstruktor dataframe pandas
    """
    def __init__(self,data):
        self.__data = data
    
    def setData(self,data):
        self.__data = data
    
    def getData(self):
        return self.__data
    
    def distribusi(self):
        num_kolom = len(self.__data.columns)
        num_rows = (num_kolom + 4)//5
        num_rows = min(15, num_rows)

        fig,axes = plt.subplots(num_rows, 5, figsize=(30, 20))
    
        for i, kolom in enumerate(self.__data.columns):
            row = i //5
            col = i % 5
            sns.histplot(self.__data[kolom], kde=True, ax=axes[row, col])
            #axes[row,col].set_title(kolom)
    
        for i in range(num_kolom, num_rows * 5):
            row = i // 5
            col = i % 5
            fig.delaxes(axes[row, col])
    
        fig.suptitle('Distribusi untuk Setiap Kolom', y=1.02)


        plt.tight_layout()
        plt.show()

    def box_plot(self,kolom):
         #fungsi boxplot
        sns.set(rc={'figure.figsize':(38,10)})
        melted_data = pd.melt(self.__data, value_vars=kolom, var_name="variabel", value_name="value")
        ax = sns.boxplot(x="variabel",y="value",data=melted_data)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.show()
    
    def scatter(self,kolom):
        num_kolom = len(self.__data.columns)
        num_rows = (num_kolom + 4)//5
        num_rows = min(15, num_rows)

        fig,axes = plt.subplots(num_rows, 5, figsize=(30, 20))

        for i, colu in enumerate(self.__data.columns):
            row = i // 5
            col = i % 5
            axes[row, col].scatter(x=self.__data[colu], y=self.__data[kolom])
            axes[row, col].set_xlabel(colu)

        for i in range(num_kolom, num_rows * 5):
            row = i // 5
            col = i % 5
            fig.delaxes(axes[row, col])
    
        fig.suptitle(f'Scatter plot untuk Setiap Kolom terhadap {kolom}', y=1.02)


        plt.tight_layout()
        return plt
    
class Skalasisasi:
    """
    kelas untuk skalasisasi
    """
    def __init__(self,data):
        self.__data = data.copy()
    
    def minmax_scaler(self):
        kolom = self.__data.columns
        data = self.__data.copy()
        for col in kolom:
            maksimal = data[col].max()
            minimum = data[col].min()
            delta = maksimal - minimum
            data[kolom] = data[kolom].apply(lambda x : (x-minimum)/delta)
        
        return data
    
    def standar_scaler(self):
        kolom =self.__data.columns
        data = self.__data.copy()
        for col in kolom:
            mean = data[col].mean()
            std = data[col].std()
            data[kolom] = data[kolom].apply(lambda x : (x-mean)/std)
        
        return data
            