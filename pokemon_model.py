import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('pokemon.csv')

def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df, df_dummy], axis=1)
        df = df.drop(i, axis=1)
    return df

def train_test_splitter(DataFrame, column):
    '''
    Splits the dataframe into a train and test dataset if the
    column is not equal to 1 or is equal to 1
    '''
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    # remove column from both datasets
    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return df_train, df_test

def label_delineator(df_train, df_test, label):
    '''
    Removes the label column and extracts the dataframe's values into an array.
    Puts the label's values into an array.
    '''
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    
    test_data = df_test.drop(label, axis=1).values
    test_labels = df_test[label].values

    return train_data, train_labels, test_data, test_labels

def data_normalizer(train_data, test_data):
    '''
    Normalizes the data by scaling the values between 0 and 1
    '''
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)
    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)
    
    return train_data, test_data

# formatting data needed for model
df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]
df['isLegendary'] = df['isLegendary'].astype(int)

'''
Creating dummy variables because
1. Some categories are not numerical.
2. Converting multiple categories into numbers implies that they are on a scale.
3. The categories for "Type_1" should all be boolean (0 or 1).
'''
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])

'''
Split the data to train the model and another to test it
train: Generation > 1
test: Generation = 1
'''
df_train, df_test = train_test_splitter(df, 'Generation')

'''
Separate isLegendary label from train and test DataFrames, convert
datasets into array with only its values
'''
train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')

'''
Normalize the data sets
'''
train_data, test_data = data_normalizer(train_data, test_data)

'''
Creating the model
'''
length = train_data.shape[1]
model = keras.Sequential()
model.add(keras.layers.Dense(500, activation='relu', input_shape=[length,]))
model.add(keras.layers.Dense(2, activation='softmax'))
