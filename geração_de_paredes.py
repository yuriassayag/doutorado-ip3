# -*- coding: utf-8 -*-
"""Geração de paredes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-A_WEWKf4N_r--et9yv1A2zkDvs3gvVu

### Declaração das bibliotecas
"""

import math
import json
import numpy as np
import pandas as pd

aps_positions = {'WAP1' : [4.3 , 5.0 ],
                 'WAP2' : [8.0 , 6.0 ],
                 'WAP3' : [16.5, 3.0 ],
                 'WAP4' : [23.0, 3.5 ],
                 'WAP5' : [33.0, 5.0 ],
                 'WAP6' : [38.0, 4.3 ],
                 'WAP7' : [39.0, 11.2],
                 'WAP8' : [33.5, 12.0],
                 'WAP9' : [28.5, 12.0],
                 'WAP10': [20.0, 12.3],
                 'WAP11': [1.5 , 11.4],
                 'WAP12': [4.0 , 17.5],
                 'WAP13': [13.5, 10.0],
                 'WAP14': [17.8, 9.0 ],
                 'WAP15': [32.3, 9.0 ]}

list_aps = list(aps_positions.keys())

"""#### Abre o json com as informações das salas (dimensão)"""

with open('rooms.json', 'r') as json_file:
    rooms = json.load(json_file)

"""#### Calcula a quantidade de paredes por intersecção de retas."""

class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def interceptPoint(self, x1, y1, x2, y2):
        p0_x = self.x1
        p0_y = self.y1
        p1_x = self.x2
        p1_y = self.y2

        p2_x = x1
        p2_y = y1
        p3_x = x2
        p3_y = y2
        
        s1_x = p1_x - p0_x
        s1_y = p1_y - p0_y
        s2_x = p3_x - p2_x
        s2_y = p3_y - p2_y

        s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
        t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

        if (s >= 0) and (s <= 1) and (t >= 0) and (t <= 1):
            intX = p0_x + (t * s1_x)
            intY = p0_y + (t * s1_y)
            
            return [intX, intY];

        return -1

"""#### Calcula o tamanho de todas as paredes do cenário (top, bottom, right, left)"""

walls = []

for r in rooms['rooms']:
    room = rooms['rooms'][r]
    
    if room['type'] == 'room':
        xWidth = room['x']+room['width']
        yHeight = room['y']+room['height']

        walls.append(Wall(room['x'], room['y'], xWidth   , room['y'] )); #top
        walls.append(Wall(xWidth   , room['y'], xWidth   , yHeight   )); #right
        walls.append(Wall(xWidth   , yHeight  , room['x'], yHeight   )); #bottom
        walls.append(Wall(room['x'], yHeight  , room['x'], room['y'] )); #left

"""#### Realiza o cálculo de paredes entre dois pontos usando as funções acima. O primeiro ponto é a posição do beacon, o segundo é  a posição do ponto."""

def intercept(x1, y1, x2, y2):

    if (x1 == x2) and (y1 == y2):
        return []

    allInterceptions = []
    
    for wall in walls:
        intercept = wall.interceptPoint(x1, y1, x2, y2)

        if intercept != -1:
            allInterceptions.append(intercept);
    
    result = []

    for testPoint in allInterceptions:
        foundClose = 'false'

        for resultPoint in result:
            dist = math.sqrt(math.pow(testPoint[0] - resultPoint[0], 2) + math.pow(testPoint[1]-resultPoint[1], 2))

            if dist < 5:
                foundClose = 'true'
                break

        if foundClose == 'false':
            result.append(testPoint)

    return result

#TESTE

p1 = [397,416]
p2 = [2645,607]

res = len(intercept(p1[0],p1[1],p2[0],p2[1]))
print('Paredes:', res)

df = pd.read_csv('db_yuri_training5_semsala42_1-todos-aps-maxValue.csv')
df = df[(df["DEVICE"] == '2055') | (df["DEVICE"] == '121B') | (df["DEVICE"] == '20B5')].reset_index(drop=True)

labels = df["LABEL"].unique()
labels = list(labels)

temp, x, y = '', '', ''

dic = {"LABEL": [], 
       "X": [],
       "Y": [],
       "WAP1": [],
       "WAP2": [],
       "WAP3": [],
       "WAP4": [],
       "WAP5": [],
       "WAP6": [],
       "WAP7": [],
       "WAP8": [],
       "WAP9": [],
       "WAP10": [],
       "WAP11": [],
       "WAP12": [],
       "WAP13": [],
       "WAP14": [],
       "WAP15": []}

for i in range(len(labels)):
    temp = df[df["LABEL"] == labels[i]].iloc[0]
    point = [temp["X"],temp["Y"]]
    #print('LABEL:', temp["LABEL"], 'POINT:', point)

    dic["LABEL"].append(temp["LABEL"])
    dic["X"].append(point[0])
    dic["Y"].append(point[1])

    for i in range(len(list_aps)):
        ap_pos = aps_positions[list_aps[i]]
        paredes = len(intercept(point[0]*100,point[1]*100,ap_pos[0]*100,ap_pos[1]*100))
        #print(list_aps[i], 'POS:', ap_pos, 'WALLS:', paredes)

        dic[list_aps[i]].append(paredes)

print(dic)

df_out = pd.DataFrame.from_dict(dic)
df_out.to_csv('walls_values-ofc.csv', encoding='utf-8', index=True)
