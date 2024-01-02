import numpy as np
import random
import copy
from string import ascii_uppercase
import enchant
import matplotlib.pyplot as plt

#US english dictionary
suggestor = enchant.Dict('en_US')

def skeleton(x):
    """Define skeleton (grid) of the crossword based on hard-coding
    
    1 represents white block which can be filled with letter
    0 represents black block which can't be filled with letter
    :param x: number pointing available skeleton on this project. dtype: int
    """
    if x == 1:
        skeleton = [[1,1,1,1,0,1,1,1,1,1,1,1,1,1,1],
                    [0,1,0,1,0,1,0,0,1,0,1,0,1,0,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [0,1,0,1,0,1,0,0,1,0,1,0,1,0,1],
                    [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1],
                    [0,1,0,1,0,1,0,1,0,1,0,1,0,0,1],
                    [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
                    [0,0,0,1,0,1,0,1,0,1,0,1,0,0,0],
                    [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1],
                    [1,0,0,1,0,1,0,1,0,1,0,1,0,1,0],
                    [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1],
                    [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0],
                    [1,1,1,1,1,1,1,1,1,1,0,1,1,1,1]]
    elif x == 2:
        skeleton = [[1,1,1,1,1,0,1,1,1,1,0,0,1,1,1],
                    [1,1,1,1,1,0,1,1,1,1,0,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [0,0,0,1,1,1,1,1,0,1,1,1,1,1,1],
                    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
                    [1,1,1,1,1,1,0,1,1,1,1,1,0,0,0],
                    [1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,0,1,1,1,1,0,1,1,1,1],
                    [0,0,0,1,1,1,1,1,0,1,1,1,1,1,1],
                    [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],
                    [1,1,1,1,1,1,0,1,1,1,1,1,0,0,0],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
                    [1,1,1,0,0,1,1,1,1,0,1,1,1,1,1]]
    
    elif x == 3:
        skeleton = [[0,1,1,1,1,1,1,1,1,1,0],
                    [1,0,1,0,1,0,1,0,1,0,1],
                    [1,1,1,1,1,0,1,1,1,1,1],
                    [1,0,1,0,1,1,1,0,1,0,1],
                    [1,1,1,1,1,0,1,1,1,1,1],
                    [1,0,0,1,0,0,0,1,0,0,1],
                    [1,1,1,1,1,0,1,1,1,1,1],
                    [1,0,1,0,1,1,1,0,1,0,1],
                    [1,1,1,1,1,0,1,1,1,1,1],
                    [1,0,1,0,1,0,1,0,1,0,1],
                    [0,1,1,1,1,1,1,1,1,1,0]]
    return np.asarray(skeleton)
       
def crossover(parent_1, parent_2):
    """Cross-over process of genetic algorithm
    
    Parents are duplicated into their own offspring. 
    A word in each offspring is randomly selected to be interchanged based on its position.
    
    :param parent_1: first parent. dtype: class Individual instance
    :param parent_2: second parent. dtype: class Individual instance
    
    return first offspring and second offspring
    """
    offs_1 = copy.deepcopy(parent_1)
    offs_2 = copy.deepcopy(parent_2)
    a = random.randint(1,len(parent_1.word_list)-1)
    b = random.randint(1,len(parent_2.word_list)-1)
    idx = list(population[0].word_list.keys())
    key_1 = idx[a]
    key_2 = idx[b]
    word_1 = offs_1.word_list[key_1]
    word_2 = offs_2.word_list[key_2]
    offs_1.wordlist = offs_1.update_wordlist(key_2, word_2)
    offs_2.wordlist = offs_2.update_wordlist(key_1, word_1)
    offs_1.fitness = offs_1.compute_fitness()
    offs_2.fitness = offs_2.compute_fitness()
    return offs_1, offs_2               
 

class Individual(object):
    """Individual or chromosome
    
    :param skeleton: skeleton used for the crossword. dtype: numpy array
    :param grid: grid filled with numbers representing letters. dtype: numpy array
    :param word_list: words formed by the grid. dtype: dictionary
    
    """
    
    def __init__(self, skeleton, grid=[], word_list={}, fitness=0):
        self.skeleton = skeleton
        self.grid = grid
        self.word_list = word_list
        self.fitness = fitness
    
    def init_grid(self):
        """Initialize grid filled with random integer representing letter
        
        return crossword grid of individual. dtype: numpy array
        """
        self.grid = self.skeleton.copy()
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i,j] == 1:
                    self.grid[i,j] = random.randint(65,90)
        return self.grid
        
    def init_wordlist(self):
        """Initialize word_list
        
        Defining dictionary with key ((x,y), direction, length) and value word
        : (x,y) is index in the grid of the first letter
        : direction is {0,1} represents across or down
        : length is the word length
        : word is string of letters formed by decoding the grid based on the key
        
        return word list of individual. dtype: dictionary     
        """
        body = self.skeleton.copy()
        a = np.zeros((1, len(body[0])))
        body = np.concatenate((a, body, a), axis=0)
        b = np.zeros((len(body),1))
        body = np.concatenate((b, body, b), axis=1)
    
        n_row = len(body) - 2
        n_col = len(body[0]) - 2   
        
        wlength = {}
        for i in range(1,n_row+1):
            for j in range(1,n_col+1):
                if body[i,j] == 1:
                    loc = (i-1, j-1)
                    if body[i,j-1] == 0:
                        wlen = 0
                        while body[i,j] != 0:
                            wlen += 1
                            j += 1
                        if wlen > 1:
                            wlength[loc] = wlen
        
        hlength = {}
        for j in range(1,n_col+1):
            for i in range(1,n_row+1):
                if body[i,j] == 1:
                    loc = (i-1, j-1)
                    if body[i-1,j] == 0:
                        hlen = 0
                        while body[i,j] != 0:
                            hlen += 1
                            i += 1
                        if hlen > 1:
                            hlength[loc] = hlen    
        self.word_list = {}
        for loc in wlength:
            i, j = loc
            key = (loc, 0, wlength[loc])
            self.word_list[key] = None        
        for loc in hlength:
            i, j = loc
            key = (loc, 1, hlength[loc])
            self.word_list[key] = None
            
        self.word_list = self.decode_grid()
        return self.word_list
    
    def decode_grid(self):
        """Get list of words formed by the grid
        
        return word_list represented as dictionary
        """
        for key, _ in self.word_list.items():
            word = []
            i,j = key[0]
            if key[1] == 0:
                for l in range(key[2]):
                    word.append(chr(self.grid[i,j]))
                    j += 1
            elif key[1] == 1:
                for l in range(key[2]):
                    word.append(chr(self.grid[i,j]))
                    i += 1
            word = ''.join(word)
            self.word_list[key] = word
        return self.word_list        
    
    def update_wordlist(self, key, new_word):
        """Update current grid and word_list given a new word
        
        :param key: referring a word location in word_list
        :param new_word: word replacing current word on a location (key)
        
        return updated grid and word_list
        """
        self.word_list[key] = new_word
        i,j = key[0]
        for l in range(len(new_word)):
            self.grid[i,j] = ord(new_word[l])
            if key[1] == 0:
                j += 1
            elif key[1] == 1:
                i += 1
        self.word_list = self.decode_grid()
        return self.grid, self.word_list
        
    def compute_fitness(self):
        """Compute fitness of Individual
        
        penalty is added 1 if word is not found in dictionary
        fitness = 1/(penalty + 0.00000000001)
        objective = max(fitness) = min(penalty)
        
        return fitness value
        """
        penalty = 0
        for key in self.word_list:
            word = self.word_list[key]
            if suggestor.check(word)==False:
                penalty += 1
        self.fitness = 1/(penalty + 0.00000000001)
        return self.fitness
               
        
#%%
# parameter configuration

n_individual = 10
n_generation = 100
p_c = 1
p_m = 0.1

# select skeleton
s = skeleton(1) 
individual = Individual(s)

# generate population
population = []
for i in range(n_individual):
    x = Individual(s, grid=individual.init_grid(), word_list=individual.init_wordlist())
    x.word_list = x.decode_grid()
    x.fitness = x.compute_fitness()
    population.append(x)
key = list(population[0].word_list.keys())

n_words = len(population[0].word_list)

# print
print ("Algorithm is running for", n_generation, "generations")
print ("Number of individual generated is", n_individual)
print ("Crossover probability:", p_c)
print ("Mutation probability:", p_m)
print ("Total number of words in the grid:", n_words)

#%%
list_penalty = []

for generation in range(n_generation):

    # cross-over
    
    n_parent = int(p_c* n_individual)
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    parent = population[:n_parent]
    
    offs = []
    for i in range(0,n_parent,2):
        parent_1 = parent[i]
        parent_2 = parent[i+1]
        offs_1, offs_2 = crossover(parent_1, parent_2)
        offs.append(offs_1)
        offs.append(offs_2)
        
    # mutation
        
    n_mutation = int(p_m*n_words)
    
    for p in range(len(offs)):
        selected = random.sample(range(0,n_words),n_mutation)
        letter="'"
        for q in selected:
            selected_word = offs[p].word_list[key[q]]
            selected_key = key[q]
            suggested = suggestor.suggest(selected_word)
            if suggested != []:           
                for r in range(len(suggested)):
                    if len(selected_word) == len(suggested[r]) and any(letter in word and len(word) > 1 for word in suggested[r].split()) == False:
                        suggestor.add_to_session(suggested[r])
                        #print suggested[r]
                        offs[p].wordlist = offs[p].update_wordlist(selected_key,suggested[r])[1]
                        offs[p].fitness = offs[p].compute_fitness()
                        break
            else:
                new_word = list(selected_word)
                new_word[1] = random.choice(ascii_uppercase)
                new_word = ''.join(new_word)
                offs[p].wordlist = offs[p].update_wordlist(selected_key,new_word)
                offs[p].fitness = offs[p].compute_fitness()
    
    
    # generation survivor -- elitism
    population.extend(offs)
    population.sort(key=lambda x: x.fitness, reverse=True)
    population = population[:n_individual]
    
    # get the best individual
    best = population[0]
    chrv = np.vectorize(chr)
    cw = chrv(best.grid)
    
    # print penalty
    list_penalty.append(int(1/best.fitness))
    print ("Generation:", generation+1, "\t Wrong words:", int((1/best.fitness)-0.00000000001))


#%% 

#print the best grid  
print ("BEST INDIVIDUAL")
print (cw)

# plot penalty by generation
plt.plot(list_penalty)
plt.xlabel('Generation')
plt.ylabel('Penalty')
plt.title('Penalty Plot')
plt.show() 
       