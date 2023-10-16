#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:22:41 2023

@author: kuailegugu
"""

#project

# import some packages

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import helper_functions

# change some settings

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 8)
plt.rcParams['figure.figsize'] = (10,8)

# indicate paths to files

import os 
home = os.environ["HOME"]
path_to_data = f"{home}/shared/project/lyrics.csv"
path_to_dictionary = f"{home}/shared/project/words.csv"

#question 1.1

unchanged = 0
for i in range(len(stemming['Stem'])):
    if stemming['Stem'][i] == stemming['Word'][i]:
        unchanged+=1
unchanged_fraction = unchanged/len(stemming)
unchanged_fraction

q1_1 = 0.7216639871382636


#question 1.2
stemming.loc[stemming["Word"] == 'message',['Word','Stem']]

q1_2 = 'messag'

#question 1.3
stemming.loc[stemming['Stem'] == 'singl',['Word','Stem']]
q1_3 = 'single'

#question 1.4
max_stemming_dif = 0
for i in range(len(stemming)):
    if stemming['length difference'][i]>max_stemming_dif:
        max_stemming_word = stemming['Word'][i]
max_stemming_word
q1_4 = sincere

#question 2
#training set & testing set 
#training只要80%的总的数据

training_proportion = 0.8

number_songs = len(lyrics)
number_training = int(number_songs * training_proportion)

lyrics_shuffled = lyrics.sample(frac = 1, random_state = 42)#随机打乱
#replace = True表示有放回抽样，frac=0.5表示随机抽取50%，random_state表示种子，保证数据可重复实现

training_set = lyrics_shuffled.iloc[:number_training]#取training的数量
testing_set = lyrics_shuffled.iloc[number_training:]#取剩下的部分，保证两个set的数据不重复

training_set = training_set.reset_index()
testing_set = testing_set.reset_index()

country_training = 0
for i in range(len(training_set)):
    if training_set['Genre'][i] == 'Country':
        country_training+=1
proporation_country_training = country_training/len(training_set)
'proporation_country_training'+'='+ str(proporation_country_training)

country_testing = 0
for i in range(len(testing_set)):
    if testing_set['Genre'][i] == 'Country':
        country_testing+=1
proporation_country_testing = country_testing/len(testing_set)
'proporation_country_testing'+'='+ str(proporation_country_testing)

proportion_country_training = 0.4943820224719101
proportion_country_testing = 0.5508982035928144


#question 3
in_your_eyes = training_set.loc[training_set["Title"] == "In Your Eyes",["like","love"]] 
sangria_wine = testing_set.loc[testing_set["Title"] == "Sangria Wine",["like","love"]]
#取出like 和love的值
distance = np.sum((sangria_wine.values - in_your_eyes.values)**2)#直接两个value相减就会得到对应值相减的平方的和
distance = np.sqrt(distance) #计算两点之间平面上的直线距离

def distance_two_songs(row_1, row_2, words):
    coordinates_1 = row_1[words]
    coordinates_2 = row_2[words]

    distance = np.sum((coordinates_1.values-coordinates_2.values)**2)
    distance = np.sqrt(distance)
    
    # YOUR CODE HERE
    
    return distance

in_your_eyes = training_set.loc[training_set["Title"] == "In Your Eyes",:] 
sangria_wine = testing_set.loc[testing_set["Title"] == "Sangria Wine",:]
words = ['like','love','the']

distance_two_songs(in_your_eyes,sangria_wine,words)

#question 3.2
words = ["like", "love", "the"]
row = testing_set.loc[testing_set["Title"] == "Sangria Wine", :]
#这个row是title = sangria wine的一整行
distance = helper_functions.compute_distances(row, training_set, words)
#得到的是一个array(为什么有这么多？)

closest_song = training_set_with_distance.loc[training_set_with_distance['distance'] == training_set_with_distance['distance'].min(),'Title']
closest_song

q2_2_1 = "I'm In Love"

#2_2_2
training_set_with_distance.loc[training_set_with_distance['Title']== "I'm In Love",'Genre']
    
q2_2_2 = 'Hip-hop'

#question 4
training_set_with_distance_top_15 = training_set_with_distance.sort_values("distance", ascending = True).head(15)
#顺序排列，获取最近的15个song
#4.1
count_country_nearest_neighbors = 0
count_hiphop_nearest_neighbors = 0
for i in training_set_with_distance_top_15['Genre']:
    if i == 'Country':
        count_country_nearest_neighbors+=1
    else:
        count_hiphop_nearest_neighbors+=1
        
count_country_nearest_neighbors = 8
count_hiphop_nearest_neighbors = 7

q4_1 = 'Country'

#4.2
def compute_mode(column, table):
    return table[column].mode().values[0]
#这个table的指定column，取众数，取第一个
compute_mode('Genre',training_set_with_distance_top_31)
q4_2 = 'Country'

#question 5
#注意每次操作的时候最好都要.copy()一下
predictions = []

# iterate through the rows of testing_set
for idx, row in testing_set.iterrows():
    #compute distance from a song to the songs in training_set
    #计算testing和training集里的距离
    distance = helper_functions.compute_distances(row, training_set_with_distance, words)
    training_set_with_distance["distance"] = distance
    #加一列distance
    
    # sort the songs in traing_set by distance
    training_set_with_distance_top_k = training_set_with_distance.sort_values("distance", ascending = True).head(k)
    #返回前k个nearest song
    # determine mode 
    prediction = compute_mode("Genre", training_set_with_distance_top_k)
    #用之前定义的函数计算k个中的众数
    # record the prediction
    predictions.append(prediction)
    #返回到前面定义过的dict里
#这个迭代是对testing里的每一首歌给出一个定义，返回到字典里
count_country_testing = 0
count_hiphop_testing = 0
for i in predictions:
    if i == 'Country':
        count_country_testing += 1
    else:
        count_hiphop_testing += 1
count_country_testing
count_hiphop_testing
count_country_testing = 196
count_hiphop_testing = 138

#5.2
testing_set_with_predictions = testing_set.copy()
testing_set_with_predictions["predictions"] = predictions
testing_set_with_predictions.head(3)



























