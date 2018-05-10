"""==============================================================
 COURSE:	CSC 635, Homework 4
 PROGRAMMER:	Junya Zhao and Rohan Saha
 Trace:         \\trace\Class\CSC-535-635\001\Rohan2728\HW4
 DATE:	        5/1/2018
 DESCRIPTION:   To implement a Genetic Algorithm, for jump it game
 FILES:	        hw4.py
 DATASET:       input1.txt and input2.txt
 =============================================================="""

# ---------------------------------Imports--------------------------------------
import itertools as it
import random

# ------------------------------------------------------------------------------


# ---------------------------------Variables------------------------------------
global cost, path

cost = []  # global table to cache results - cost[i] stores minimum cost of playing the game starting at cell i
path = []  # global table to store path leading to cheapest cost

# ------------------------------------------------------------------------------

# ---------------------------------Class----------------------------------------


class GeneticAlgorithm:
    probCrossover = 0.0
    probMutation = 0.0
    maxGeneration = 0
    board = []
    population = []
    fitness = []

    def __init__(self, gameBoard, nMaxGeneration, pCrossover, pMutation):
        self.population = []
        self.fitness = []
        self.board = gameBoard
        self.probCrossover = pCrossover
        self.probMutation = pMutation
        self.maxGeneration = nMaxGeneration

    '''Initializes Genetic Algorithm'''
    def geneticAlgorithm(self):
        self.createPopulation()
        list(map(lambda c: self.fitness.append(self.generateFitness(c)), self.population))
        for i in range(self.maxGeneration):
            '''Selection Roulette Wheel'''
            parent1 = self.selectionRouletteWheel()
            parent2 = self.selectionRouletteWheel()

            child1, child2 = self.crossover(parent1, parent2) # Do crossover
            mChild1, mChild2 = self.mutation(child1, child2) # Do mutation

            child1Fitness = self.generateFitness(mChild1)
            child2Fitness = self.generateFitness(mChild2)

            # Replaces the least fittest individuals from the gene pool
            if child1Fitness > self.generateFitness(parent1) or child1Fitness > self.generateFitness(parent2):
                sP1 = min(self.fitness)
                sP1_index = self.fitness.index(sP1)
                if self.population.count(mChild1) == 0:
                    self.population[sP1_index] = mChild1
                    self.fitness[sP1_index] = child1Fitness
                
            if child2Fitness > self.generateFitness(parent1) or child2Fitness > self.generateFitness(parent2):
                sp2 = min(self.fitness)
                sP2_index = self.fitness.index(sp2)                
                if self.population.count(mChild2) == 0:
                    self.population[sP2_index] = mChild2
                    self.fitness[sP2_index] = child2Fitness
            
    '''Roulette Wheel selection function'''
    def selectionRouletteWheel(self):
        totalSum = sum(self.fitness)
        rWheel = list(map(lambda f: f/totalSum, self.fitness))
        s = 0
        pick = random.random()
        for i, r in enumerate(rWheel):
            s += r
            if pick <= s:
                parent = self.population[i]
                break

        return parent

    '''Single point crossover function'''
    def crossover(self, parent1, parent2):
        child1, child2 = [], []
        counter = 0
        if random.random() <= self.probCrossover:
            while counter < len(parent1):
                c1, c2 = self.doCrossover(parent1, parent2)
                counter += 1
                if self.pairwiseTesting(c1) and self.pairwiseTesting(c2):
                    child1 = c1
                    child2 = c2
                    break
            if counter >= len(parent1):
                child1 = parent1
                child2 = parent2
        else:
            child1 = parent1
            child2 = parent2

        return child1, child2

    def doCrossover(self, parent1, parent2):
        crossOverPoint = int(len(parent1)/2)
        c1 = parent1[:crossOverPoint] + parent2[crossOverPoint:]
        c2 = parent2[:crossOverPoint] + parent1[crossOverPoint:]
        return c1, c2

    '''Random index mutation function'''
    def mutation(self, child1, child2):
        mChild1, mChild2 = [], []
        counter = 0
        if random.random() <= self.probMutation:
            while counter < len(child1):
                mC1, mC2 = self.doMutation(child1, child2)
                counter += 1
                if self.pairwiseTesting(mC1) and self.pairwiseTesting(mC2):
                    mChild1 = mC1
                    mChild2 = mC2
                    break
            if counter >= len(child1):
                mChild1 = child1
                mChild2 = child2
        else:
            mChild1 = child1
            mChild2 = child2

        return mChild1, mChild2

    def doMutation(self, child1, child2):
        mChild1, mChild2 = child1[:], child2[:]
        index = random.randint(0, len(child1) - 2)
        if child1[index] == 0:
            mChild1[index] = 1
        else:
            mChild1[index] = 0
        if child2[index] == 0:
            mChild2[index] = 1
        else:
            mChild2[index] = 0
        return mChild1, mChild2

    '''Evalution function to get the fitness value of each genome'''
    def generateFitness(self, chromosome):
        cost = 0
        for index, value in enumerate(chromosome):
            if value == 1:
                cost += self.board[index]
        return 1/cost

    '''Generates candidate solutions or the gene pool'''
    def createPopulation(self):        
        n = len(self.board)
        for i in range(n*3):
            pop = [random.randint(0, 1)]
            for j in range(n-2):
                if pop[-1] == 0:
                    pop.append(1)
                else:
                    pop.append(random.randint(0, 1))
            pop.append(1)
            if self.population.count(pop) == 0:
                self.population.append(pop)

    '''Function to validate that genome doesnot have simultaneous 0 values''' 
    def pairwiseTesting(self, candSol):
        isSame = True
        x, y = it.tee(candSol)
        next(y, None)
        for i, j in zip(x, y):
            if i == 0 and j == 0:
                isSame = False
                break
        return isSame

    '''Gets the max fittest value from the gene pool, afer max generation is reached'''
    def getFittestSolution(self):
        max_fittest = max(self.fitness)
        fittest = self.population[self.fitness.index(max_fittest)]
        max_fittest = round(1 / max_fittest)
        print("Minimum Cost (fitness): ", max_fittest)
        path_contents = "0"
        print("path showing indices of visited cells:", end=" ")
        print(0, end="")
        for i in range(len(fittest)):
            if fittest[i] == 1:
                print(" ->", i + 1, end="")
                path_contents += " -> " + str(self.board[i])
        print("\npath showing contents of visited cells: ", path_contents)

        return max_fittest


# ------------------------------------------------------------------------------

# ---------------------------------Functions------------------------------------
"""
Dynamic Programming solution to the jump-It problem
The solution finds the cheapest cost to play the game along with the path leading
to the cheapest cost
"""
def jumpIt(board):
    # Bottom up dynamic programming implementation
    # board - list with cost associated with visiting each cell
    # return minimum total cost of playing game starting at cell 0

    n = len(board)
    cost[n - 1] = board[n - 1]  # cost if starting at last cell
    path[n - 1] = -1  # special marker indicating end of path "destination/last cell reached"
    cost[n - 2] = board[n - 2] + board[n - 1]  # cost if starting at cell before last cell
    path[n - 2] = n - 1  # from cell before last, move into last cell
    # now fill the rest of the table
    for i in range(n - 3, -1, -1):
        # cost[i] = board[i] + min(cost[i+1], cost[i+2])
        if cost[i + 1] < cost[i + 2]:  # case it is cheaper to move to adjacent cell
            cost[i] = board[i] + cost[i + 1]
            path[i] = i + 1  # so from cell i, one moves to adjacent cell
        else:
            cost[i] = board[i] + cost[i + 2]
            path[i] = i + 2  # so from cell i, one jumps over cell
    return cost[0]


def displayPath(board):
    # Display path leading to cheapest cost - method displays indices of cells visited
    # path - global list where path[i] indicates the cell to move to from cell i
    cell = 0  # start path at cell 0
    print("path showing indices of visited cells:", end=" ")
    print(0, end="")
    path_contents = "0"  # cost of starting/1st cell is 0; used for easier tracing
    while path[cell] != -1:  # -1 indicates that destination/last cell has been reached
        print(" ->", path[cell], end="")
        cell = path[cell]
        path_contents += " -> " + str(board[cell])
    print()
    print("path showing contents of visited cells:", path_contents)


'''Calculates the overall accuracy of Genetic algorithm against Dynamic Programming'''
def Accuracy():
    correct = 0
    for dp_cost, ga_cost in accuracyResult:        
        if dp_cost == ga_cost :
            correct += 1
    overall_accuracy = (correct / len(accuracyResult)) * 100
    print("GA Overall Accuracy: ", overall_accuracy, "%")


# ------------------------------------------------------------------------------

# ---------------------------------Program Main---------------------------------

if __name__ == "__main__":
    # f = open("input1.txt", "r")  # input1.txt
    f = open("input2.txt", "r")  # input2.txt
    
    accuracyResult = []
    for line in f:
        lyst = line.split()  # tokenize input line, it also removes EOL marker
        lyst = list(map(int, lyst))

        # DP solution
        cost = [0] * len(lyst)  # create the cache table
        path = cost[:]  # create a table for path that is identical to path
        min_cost = jumpIt(lyst)

        print("game board:", lyst)
        print("_______________________________________________________")
        print("DP Solution")
        print("minimum cost: ", min_cost)
        displayPath(lyst)
        print("_______________________________________________________")
        print("GA Solution")
        maxGeneration = 700
        probCrossover = 0.8
        probMutation = 0.8
        ga = GeneticAlgorithm(lyst[1:], maxGeneration, probCrossover, probMutation)
        ga.geneticAlgorithm()
        max_fittest = ga.getFittestSolution()
        print("===========================================================================================")
        print()
        accuracyResult.append((min_cost, max_fittest))
    Accuracy()
# ---------------------------------End of Program-------------------------------
