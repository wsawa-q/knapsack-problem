# Knapsack Genetic Algorithm Documentation

## User Documentation

### Overview
This Haskell program implements a genetic algorithm to solve the Knapsack Problem. The Knapsack Problem involves selecting a subset of items, each with a weight and value, to maximize the total value without exceeding a specified weight capacity.

### Running the Program
To run the program, follow these steps:

1. **Compile the Haskell code:**
    ```sh
    ghc knapsack.hs
    ```

2. **Execute the compiled program with the required arguments:**
    ```sh
    ./knapsack <filename> <popSize> <numGenerations> <groupSize> <mutationRate>
    ```

    - `<filename>`: Path to the file containing knapsack problem data.
    - `<popSize>`: Size of the population.
    - `<numGenerations>`: Number of generations to evolve.
    - `<groupSize>`: Size of the group for natural selection.
    - `<mutationRate>`: Mutation rate for the genetic algorithm.

    **Example command:**
    ```sh
    ./knapsack dataset1.txt 250 80 10 0.1
    ```

### Dataset File Structure
The dataset files (`dataset1.txt`, `dataset2.txt`, `dataset3.txt`) have the same structure. The first line specifies the maximum bag capacity, followed by lines with pairs of numbers representing the value and weight of each item.

**Example:**
50
60 10  
100 20  
120 30  
30 5  
90 15  
70 8  
80 12  
40 7  
50 9  
20 3  

- The first line `50` is the maximum bag capacity.
- Each subsequent line contains two numbers: the value and weight of an item.

## Developer Documentation

### Overview of the Knapsack Problem
The Knapsack Problem is a classic optimization problem where the goal is to select items with given weights and values to maximize the total value without exceeding the capacity of the knapsack. The genetic algorithm is a heuristic search method used to find approximate solutions to optimization problems by mimicking the process of natural selection.

### Main Function
The main function parses command-line arguments, loads the dataset, and runs the genetic algorithm to find the optimal set of items.

```haskell
main :: IO ()
main = do
  args <- getArgs
  let [filename, popSizeStr, numGenerationsStr, groupSizeStr, mutationRateStr] = args
  let popSize = read popSizeStr :: Int
  let numGenerations = read numGenerationsStr :: Int
  let groupSize = read groupSizeStr :: Int
  let mutationRate = read mutationRateStr :: Double
  (capacity, items) <- loadItems filename
  winner <- geneticAlgorithm popSize numGenerations groupSize mutationRate capacity items
  let itemIndices = [idx | (idx, gene) <- zip [1..] (fst winner), gene]
  putStrLn $ "Items: " ++ show itemIndices
  putStrLn $ "Fitness: " ++ show (snd winner)

## Genetic Algorithm
The genetic algorithm evolves a population of candidate solutions (genomes) over a specified number of generations to find the best individual.

### Important Functions

- `loadItems`: Reads the file containing the knapsack problem data and returns the capacity and a list of items (value, weight).
    ```haskell
    loadItems :: String -> IO (Int, [(Int, Int)])
    ```

- `createInitialPopulation`: Generates an initial population of random genomes and calculates their fitness.
    ```haskell
    createInitialPopulation :: Int -> Int -> Int -> [(Int, Int)] -> IO Population
    ```

- `fitness`: Calculates the fitness of a genome based on the total value and weight of included items considering the knapsack capacity.
    ```haskell
    fitness :: Genome -> Int -> [(Int, Int)] -> Int
    ```

- `naturalSelection`: Selects the best individual from a randomly chosen subset of the population.
    ```haskell
    naturalSelection :: Int -> Population -> IO Individual
    ```

- `blendGenomes`: Creates a new genome by combining genes from two parent genomes.
    ```haskell
    blendGenomes :: Genome -> Genome -> IO Genome
    ```

- `mutateGenome`: Mutates the genome by flipping genes with a given mutation rate.
    ```haskell
    mutateGenome :: Double -> Genome -> IO Genome
    ```

- `geneticAlgorithm`: Runs the genetic algorithm evolving the population over a specified number of generations to find the best individual.
    ```haskell
    geneticAlgorithm :: Int -> Int -> Int -> Double -> Int -> [(Int, Int)] -> IO Individual
    ```

### Algorithm Description
The genetic algorithm follows these steps:

1. **Initialization:** Generate an initial population of random solutions (genomes).
2. **Selection:** Select the best individuals from the population based on their fitness.
3. **Crossover:** Create new solutions by combining genes from selected parents.
4. **Mutation:** Introduce random changes to some genes to maintain diversity.
5. **Evaluation:** Calculate the fitness of new solutions.
6. **Replacement:** Form a new population by replacing the least fit individuals.

Repeat steps 2-6 for a specified number of generations or until convergence.

By following this documentation, users and developers can understand how to use and extend the genetic algorithm implementation for solving the Knapsack Problem.

