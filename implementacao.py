import math
import random
import json
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.patheffects as PathEffects
from matplotlib         import pyplot as plt
from matplotlib.patches import Circle
from time import sleep, time
from multiprocessing.pool import ThreadPool, Pool
from tqdm import tqdm
#from pyswarm            import pso
#%matplotlib inline

################################# INFORMAÇOES DO MAPA #############################
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

################################# LOG-DISTANCE PARAM #############################
d0  = 1
STD_DEV = 1
TXPOWER = 0

################################# PARTICLE SWARM OPTIMAZATION PARAM ##############
dimension = 5
fitness_criterion = 10e-8
#population = 50
#generation = 15

population = 100
generation = 15
best_particle = {}

################################# FUNÇOES GLOBAIS ################################
def euclidian_distance(P1, P2):
    X1, Y1 = P1[0], P1[1]
    X2, Y2 = P2[0], P2[1]
    return round(math.sqrt((X2-X1)**2 + (Y2-Y1)**2),2)

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
        pos = particles[i][0]
        posPixels = toPixels(pos[0], pos[1])
        label     = str(pos[0]) + str(pos[1])
        p_pl0 = particles[i][1]
        p_n   = particles[i][2]
        p_wall= particles[i][3]

        cost = costs[i]
        
        if cost < low_cost:
            low_cost = cost
            low_position = pos
        
        if label not in groups:
            groups[label] = [i, posPixels[0], posPixels[1], cost, p_pl0, p_n]
        else:
                groups[label][0] += 1
            
        circ = Circle((posPixels[0] + np.random.normal(0, 5), posPixels[1] + np.random.normal(0, 5)), radius=10, alpha=0.5, color="red", zorder=10)
        plt.gca().add_patch(circ)
    
    for group in groups.values():
        plt.annotate("P" + str(group[0]) + ', Cost:' + str(round(group[3], 1)) + '\nPL0:' + str(round(group[4], 1)) + ', N:' +str(round(group[5], 1)) , xy=(group[1]+5, group[2]-5), size=2.7, va="center", ha="left", xytext=(group[1] + 50, group[2] - 80), zorder=8,
              bbox=dict(boxstyle="square", facecolor="#ffffffcc", edgecolor="#aaaaaa88", linewidth=0.4),
              arrowprops=dict(arrowstyle="-", antialiased=True, color="#444444", connectionstyle="arc3,rad=-0.2", linewidth=0.15))
        
    #Plot Low Position
    lowPosPixels = toPixels(low_position[0], low_position[1])
    circ = Circle((lowPosPixels[0], lowPosPixels[1]), radius=10, color="#2ca05aff", zorder=12)
    plt.gca().add_patch(circ)
    
    imgFilename = "testes/sample-" + str(currentRound) + ".png"
    #plt.title("PL0, N, WALL_LOSS caminhando em difereçao ao melhor da rodada anterior", fontsize = 10)
    plt.savefig(imgFilename, dpi=300, bbox_inches="tight", pad_inches=0)
    
def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, w_max=0.5, c=0.5):
    # Initialise new velocity array
    num_pos = len(particle[0])
    num_params = len(particle)

    new_velocity = np.array([0.0 for i in range(len(particle)+1)])

    # Randomly generate r1, r2 and inertia weight from normal distribution
    r1 = random.uniform(0,w_max)
    r2 = random.uniform(0,w_max)
    w = random.uniform(w_min,w_max)
    c1 = c
    c2 = c

    # Calculate new velocity
    for i in range(num_pos):
        new_velocity[i] = (w*velocity[i] + c1*r1*(pbest[0][i]-particle[0][i])+c2*r2*(gbest[0][i]-particle[0][i]))

    # Calculate new velocity
    #for i in range(1, num_params):
    #    new_velocity[i] = (w*velocity[i+1] + c1*r1*(pbest[i]-particle[i])+c2*r2*(gbest[i]-particle[i]))
    if particle[1] < gbest[1]:
        new_velocity[2] = 2
    elif particle[1] > gbest[1]:
        new_velocity[2] = -2
    else:
        new_velocity[2] = 0

    if particle[2] < gbest[2]:
        new_velocity[3] = 0.3
    elif particle[2] > gbest[2]:
        new_velocity[3] = -0.3
    else:
        new_velocity[3] = 0
    #print(new_velocity)

    if particle[3] < gbest[3]:
        new_velocity[4] = 1
    elif particle[3] > gbest[3]:
        new_velocity[4] = -1
    else:
        new_velocity[4] = 0

    return new_velocity

def update_position(particle, velocity):
    # Move particles by adding velocity
    new_particle = [  [particle[0][0] + velocity[0], particle[0][1] + velocity[1]  ],
                       particle[1] + velocity[2], particle[2] + velocity[3], particle[3] + velocity[4]]
    #new_particle = particle + velocity
    #print('p', particle, 'v' ,velocity, 'new', new_particle, '\n')
    
    return new_particle

##################### PARTICLE SWARM OPTIMAZATION ALGORITHM ######################
def pso_2d(population, dimension, generation, fitness_criterion, real_position, row):
    seed = time()
    #seed = 1672159026.9478798
    print('seed:', seed)
    random.seed(seed)
    
    # Population (N, PL0, WALL_LOSS random between respective interval)
    particles = {i: [[round(random.uniform(x_under_lim, x_upper_lim),1), round(random.uniform(y_under_lim, y_upper_lim),1)], 
                    round(random.uniform(50, 60),0), round(random.uniform(3.5, 5),1), round(random.uniform(1, 5),0)] for i in range(population)}
    
    #print(particles)
    # Particle's best position
    pbest_position = particles
    
    # Fitness
    pbest_fitness = [fitness_function(particles[p], row) for p in particles]
    
    # Index of the best particle
    gbest_index = np.argmin(pbest_fitness)
    
    # Global best particle position
    gbest_position = pbest_position[gbest_index]
    
    # Velocity (starting from 0 speed)
    velocity = [[0.0 for j in range(dimension)] for i in range(population)]

    # Loop for the number of generation
    for t in range(generation):
        #print('------------------ round',t)
        # Stop if the average fitness value reached a predefined success criterion
        if np.average(pbest_fitness) <= fitness_criterion:
            break
        else:
            for n in range(population):
                # Update the velocity of each particle

                #print('P=', particles[n], '| V=', velocity[n], '| PB=', pbest_position[n], '| GB=',gbest_position)
                velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
                #print('Velocity[n]:', velocity[n])
                # Move the particles to new position
                particles[n] = update_position(particles[n], velocity[n])
        
        #print('----------- round', t+1)

        # Calculate the fitness value
        pbest_fitness = [fitness_function(particles[p], row) for p in particles]
        
        # Find the index of the best particle
        gbest_index = np.argmin(pbest_fitness)
        
        # Update the position of the best particle
        gbest_position = pbest_position[gbest_index]

        #best_particle = particle_RSSI(pbest_position[gbest_index])
        
        generateMap(particles, pbest_fitness, real_position, t)
    
    return gbest_position


####################### FUNÇOES PARA CADA SAMPLE DA DATABASE ###################

# Busca qual o label do json de labels que e mais proximo da particula atual para contar a quantidade de paredes.
# Recebe a particula como parametro e retorna o index da label mais proxima no dataframe.
def sample_dfwall_low_dist(particle_position):
    low = 100000
    index = 10000

    for k, v in points_walls.items():
        point_sample = [v[1], v[2]]
        dist = euclidian_distance(point_sample, particle_position)

        if dist < low:
            low = dist
            index = v[0]

    return index

# Atenuacao do RSSI como uma variavel normal.
def attenuation():
    u = 0
    v = 0
    
    u = random.uniform(0, 1)
    v = random.uniform(0, 1)
    
    normal = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
    return normal * STD_DEV

# Cada particula tem uma posição [x,y] e pegamos essa posição para obter as distâncias pros respectivos waps
# E através da distância obter o RSSI entre eles. No final será retornado um dicionário com o RSSI para todos os WAPS.
def particle_RSSI(particle_position):
    particle_sample = {}

    #label_index = sample_dfwall_low_dist(particle_position[0])
    
    PL0 = particle_position[1]
    N   = particle_position[2]
    WALL_LOSS = particle_position[3]

    for i in range(len(list_aps)):
        wap_position = aps_positions[list_aps[i]]
        dist = euclidian_distance(wap_position, particle_position[0])

        walls_qtd=0
        #walls_qtd = df_walls.at[label_index, list_aps[i]]

        if dist < 0.1:
            RSSI = -PL0
        else:
            distLoss = PL0 + 10 * N * math.log10(dist/d0)
            wallLoss = walls_qtd * WALL_LOSS
            pathLoss = distLoss + wallLoss
            
            #RSSI = round((TXPOWER - ( pathLoss)) + attenuation() ,0)
            RSSI = round((TXPOWER - ( pathLoss)),0)
        
        if RSSI < -95: RSSI = -105
        
        #particle_sample[list_aps[i]] = [RSSI, PL0, N, WALL_LOSS]
        particle_sample[list_aps[i]] = RSSI
    
    return particle_sample

# RMSD entre os RSSI da particula e do sample
def fitness_function(particle_position, row):
    particula = particle_RSSI(particle_position)
    error = 0
    
    for i in range(len(list_aps)):
        #print(list_aps[i], math.sqrt(math.pow((particula[list_aps[i]][0] - getattr(row, list_aps[i])) , 2)))
        error += math.sqrt(math.pow((particula[list_aps[i]] - getattr(row, list_aps[i])) , 2))
    
    error = error
    #print(particle_position, error, '\n')
    return round(error,5)

################################# ARQUIVOS #############################
df = pd.read_csv('db_yuri_training5_semsala42_1-todos-aps-maxValue.csv')
#df = pd.read_csv('/content/drive/MyDrive/UFAM/Doutorado/Doutorado-Artigo3/db_yuri_training5_semsala42_1-todos-aps-maxValue.csv')
df = df[(df["DEVICE"] == '2055') | (df["DEVICE"] == '121B') | (df["DEVICE"] == '20B5')].reset_index(drop=True)

df = df.sample(n=1, random_state=1)

#df_walls = pd.read_csv('/content/drive/MyDrive/UFAM/Doutorado/Doutorado-Artigo3/walls_values.csv')
df_walls = pd.read_csv('walls_values.csv')

with open('walls.json', 'r') as json_file:
    points_walls = json.load(json_file)
#########################################################################

def error_by_room(room):
    error_by_room = []
    room_df     = df[df['ROOM_ID'] == room]
    room_points = room_df['LABEL'].unique()

    for point in room_points:
        point_df = room_df[room_df['LABEL'] == point]

        for row in point_df.itertuples():
            real_position = [row.X, row.Y]

            estimated_position = pso_2d(population, dimension, generation, fitness_criterion, real_position, row)[0]
            estimated_error = euclidian_distance(real_position, estimated_position)

            error_by_room.append(estimated_error)

    return_error = np.mean(error_by_room)
    print('Sala:', room, 'Error:', round(return_error,3), 'm')
    return return_error

list_rooms = list(df['ROOM_ID'].unique())

inicio_processo = time()

subprocessos = []
pool = Pool(15)

for room in tqdm(list_rooms):
#for room in list_rooms:
    #resultado_paralelo = error_by_room(room)
    resultado_paralelo = pool.apply_async(error_by_room, (room, ))
    subprocessos.append(resultado_paralelo)

lista_api_paralela = [result.get(timeout=300) for result in tqdm(subprocessos)]
print('\n\nMEDIA:', np.mean(lista_api_paralela))
#print('\n\nMEDIA:', np.mean(subprocessos))

fim_processo = time()
processamento_paralelo = fim_processo - inicio_processo
print('Processamento paralelo:', round( (processamento_paralelo), 1 ), 'segundos')
