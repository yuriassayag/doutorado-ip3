import math
import random
import json
import numpy as np
import pandas as pd
import matplotlib.patheffects as PathEffects
from matplotlib         import pyplot as plt
#from pyswarm            import pso
from matplotlib.patches import Circle

from time import sleep, time
from multiprocessing.pool import ThreadPool, Pool
import multiprocessing
from tqdm import tqdm

#%matplotlib inline

# Map Informations
x_under_lim = 1
x_upper_lim = 44
y_under_lim = 1
y_upper_lim = 15

aps_positions = {'WAP11': [1.5 , 11.4],
                 'WAP5' : [33.0, 5.0 ],
                 'WAP3' : [16.5, 3.0 ],
                 'WAP1' : [4.3 , 5.0 ],
                 'WAP9' : [28.5, 12.0],
                 'WAP12': [4.0 , 17.5],
                 'WAP8' : [33.5, 12.0],
                 'WAP10': [20.0, 12.3],
                 'WAP6' : [38.0, 4.3 ],
                 'WAP7' : [39.0, 11.2],
                 'WAP4' : [23.0, 3.5 ],
                 'WAP2' : [8.0 , 6.0 ],
                 'WAP14': [17.8, 9.0 ],
                 'WAP15': [32.3, 9.0 ],
                 'WAP13': [13.5, 10.0]}

list_aps = list(aps_positions.keys())

#Constantes Log-Distance

#N_list = [2.9, 3.1, 3.3 , 3.5, 3.7, 3.9, 4.1, 4.2, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.6, 6.1]
#PL0_list = [40, 45, 50, 55, 60, 65]

N   = 3.5
PL0 = 60
WALL_LOSS = 3
d0  = 1
STD_DEV = 1
TXPOWER = 0

#PSO Infomations
dimension = 2
population = 50
generation = 10
fitness_criterion = 10e-8

def euclidian_distance(P1, P2):
    X1, Y1 = P1[0], P1[1]
    X2, Y2 = P2[0], P2[1]
    return round(math.sqrt((X2-X1)**2 + (Y2-Y1)**2),1)

def toPixels(x, y):
    return (x*100, y*100)

def generateMap(particles, costs, realPosition, currentRound):
    low_cost = 100000
    low_position = [0,0]
    
    perfPixels = toPixels(realPosition[0], realPosition[1])
    
    plt.clf()
    #plt.close()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)                    
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    img = plt.imread("map.png")
    plt.imshow(img)
    
    txt = plt.text(perfPixels[0], perfPixels[1], "✖", weight='bold', fontsize=8, color="white", verticalalignment='center', horizontalalignment='center', zorder=5)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    groups      = {}
    
    for i in range(len(particles)):
        pos = particles[i]
        posPixels = toPixels(pos[0], pos[1])
        label     = str(pos[0]) + str(pos[1])
        
        cost = costs[i]
        
        if cost < low_cost:
            low_cost = cost
            low_position = pos
        
        if label not in groups:
            groups[label] = [1, posPixels[0], posPixels[1], cost]
        else:
            groups[label][0] += 1
            
        circ = Circle((posPixels[0] + np.random.normal(0, 5), posPixels[1] + np.random.normal(0, 5)), radius=10, alpha=0.5, color="red", zorder=10)
        plt.gca().add_patch(circ)
    
    for group in groups.values():
        plt.annotate("Cost: " + str(round(group[3], 1)), xy=(group[1]+5, group[2]-5), size=2.7, va="center", ha="left", xytext=(group[1] + 50, group[2] - 80), zorder=8,
              bbox=dict(boxstyle="square", facecolor="#ffffffcc", edgecolor="#aaaaaa88", linewidth=0.4),
              arrowprops=dict(arrowstyle="-", antialiased=True, color="#444444", connectionstyle="arc3,rad=-0.2", linewidth=0.15))
        
    #Plot Low Position
    lowPosPixels = toPixels(low_position[0], low_position[1])
    circ = Circle((lowPosPixels[0], lowPosPixels[1]), radius=10, color="#2ca05aff", zorder=12)
    plt.gca().add_patch(circ)
    
    imgFilename = "sample-" + str(currentRound) + ".png"
    plt.savefig(imgFilename, dpi=300, bbox_inches="tight", pad_inches=0)
    
def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=0.5, c=0.5):
    # Initialise new velocity array
    num_particle = len(particle)
    new_velocity = np.array([0.0 for i in range(num_particle)])
    
    # Randomly generate r1, r2 and inertia weight from normal distribution
    r1 = random.uniform(0,max)
    r2 = random.uniform(0,max)
    w = random.uniform(w_min,max)
    c1 = c
    c2 = c
    
    # Calculate new velocity
    for i in range(num_particle):
        new_velocity[i] = round((w*velocity[i] + c1*r1*(pbest[i]-particle[i])+c2*r2*(gbest[i]-particle[i])),0)
    
    return new_velocity

def update_position(particle, velocity):
    # Move particles by adding velocity
    new_particle = particle + velocity
    
    return new_particle

def pso_2d(population, dimension, generation, fitness_criterion, real_position, row):
    
    seed = time()
    random.seed(seed)
    #print('seed:', seed)
    
    # Population
    particles = [[round(random.uniform(x_under_lim, x_upper_lim),1), round(random.uniform(y_under_lim, y_upper_lim),1)] for i in range(population)]
    
    # Particle's best position
    pbest_position = particles
    
    # Fitness
    pbest_fitness = [fitness_function([p[0],p[1]], row) for p in particles]
    
    # Index of the best particle
    gbest_index = np.argmin(pbest_fitness)
    
    # Global best particle position
    gbest_position = pbest_position[gbest_index]
    
    # Velocity (starting from 0 speed)
    velocity = [[0.0 for j in range(dimension)] for i in range(population)]

    # Loop for the number of generation
    for t in range(generation):
        # Stop if the average fitness value reached a predefined success criterion
        if np.average(pbest_fitness) <= fitness_criterion:
            break
        else:
            for n in range(population):
                # Update the velocity of each particle
                velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
                
                # Move the particles to new position
                particles[n] = update_position(particles[n], velocity[n])

        # Calculate the fitness value
        pbest_fitness = [fitness_function([p[0],p[1]], row) for p in particles]
        
        # Find the index of the best particle
        gbest_index = np.argmin(pbest_fitness)
        
        # Update the position of the best particle
        gbest_position = pbest_position[gbest_index]
        
        ##print('Round', t+1, ', Best Position:', gbest_position, ', Cost:', pbest_fitness[gbest_index])
        ##generateMap(particles, pbest_fitness, real_position, t)

    # Print the results
    #print('Global Best Position: ', gbest_position)
    #print('Best Fitness Value: ', min(pbest_fitness))
    #print('Average Particle Best Fitness Value: ', np.average(pbest_fitness))
    #print('Number of Generation: ', t)
    
    return gbest_position

df = pd.read_csv('db_yuri_training5_semsala42_1-todos-aps-maxValue.csv')
#df = pd.read_csv('/content/drive/MyDrive/UFAM/Doutorado/Doutorado-Artigo3/db_yuri_training5_semsala42_1-todos-aps-maxValue.csv')
df = df[(df["DEVICE"] == '2055') | (df["DEVICE"] == '121B') | (df["DEVICE"] == '20B5')].reset_index(drop=True)

#df_walls = pd.read_csv('/content/drive/MyDrive/UFAM/Doutorado/Doutorado-Artigo3/walls_values.csv')
df_walls = pd.read_csv('walls_values.csv')

def sample_dfwall_low_dist(particle_position):
    low = 100000
    label = ''

    for i in df_walls.itertuples():
        point_sample = [i.X, i.Y]
        dist = euclidian_distance(particle_position, point_sample)

        if dist < low:
            label = i.LABEL
            low = dist

    return label

def attenuation():
    u = 0
    v = 0
    
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    
    normal = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
    return normal * STD_DEV

# Cada particula tem uma posição [x,y] e pegamos essa posição para obter as distâncias pros respectivos waps
# e através da distância obter o RSSI entre eles. No final será retornado um dicionário com o RSSI para todos os WAPS.
def particle_RSSI(particle_position):
    particle_sample = {}

    #label = sample_dfwall_low_dist(particle_position)
    
    for i in range(len(list_aps)):
        wap_position = aps_positions[list_aps[i]]
        dist = euclidian_distance(wap_position, particle_position)
        
        #walls_qtd = df_walls[df_walls["LABEL"] == label][list_aps[i]].iloc[0]
        #print(label, list_aps[i], walls_qtd)
        walls_qtd=0
        
        if dist < 0.1:
            RSSI = -PL0
        else:
            distLoss = PL0 + 10 * N * math.log10(dist/d0)
            wallLoss = walls_qtd * WALL_LOSS
            pathLoss = distLoss + wallLoss
            
            #RSSI = round((TXPOWER - ( pathLoss)) + attenuation() ,0)
            RSSI = round((TXPOWER - ( pathLoss)) ,0)
        
        if RSSI < -95: RSSI = -105
        particle_sample[list_aps[i]] = RSSI

    return particle_sample


#RMSD entre os RSSI da particula e do sample
def fitness_function(particle_position, row):
    particula = particle_RSSI(particle_position)
    error = 0
    
    for i in range(len(list_aps)):
        #print(row, list_aps[i], error)
        rssi_particula = particula[list_aps[i]]
        rssi_ap        = getattr(row, list_aps[i])
        
        erro_atual = math.pow((rssi_particula - rssi_ap) , 2)
        erro_atual = math.sqrt(erro_atual)
        error += erro_atual
        #print(row, list_aps[i], 'error:', math.pow((particula[list_aps[i]] - getattr(row, list_aps[i])) , 2), error)
    
    return round(error,2)

def error_by_room(room):
    error_by_room = []
    room_df     = df[df['ROOM_ID'] == room]
    room_points = room_df['LABEL'].unique()
    
    for point in room_points:
        point_df = room_df[room_df['LABEL'] == point]
        
        for row in point_df.itertuples():
            real_position = [row.X, row.Y]

            estimated_position = pso_2d(population, dimension, generation, fitness_criterion, real_position, row)
            estimated_error = euclidian_distance(real_position, estimated_position)
            
            error_by_room.append(estimated_error)
            
    return_error = np.mean(error_by_room)
    print('Sala:', room, 'Error:', return_error)
    return return_error

#-----------------------------------------------------------------------------------------
list_rooms = list(df['ROOM_ID'].unique())

inicio_processo = time()

subprocessos = []
pool = Pool(15)

for room in tqdm(list_rooms):
#for room in list_rooms:
    #resultado_paralelo = error_by_room(room)
    resultado_paralelo = pool.apply_async(error_by_room, (room, ))
    subprocessos.append(resultado_paralelo)

lista_api_paralela = [result.get(timeout=120) for result in tqdm(subprocessos)]
print('\n\nMEDIA:', np.mean(lista_api_paralela))

fim_processo = time()
processamento_paralelo = fim_processo - inicio_processo
#print('Processamento paralelo:', round( (processamento_paralelo), 1 ), 'segundos')
