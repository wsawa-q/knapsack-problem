import Data.List ( maximumBy )
import System.Random ( randomIO, randomRIO )
import Control.Monad ( replicateM ) 
import Data.Ord ( comparing )
import System.Environment ( getArgs )

type Genome = [Bool]
type Individual = (Genome, Int)
type Population = [Individual]

-- Reads containing of the file like the capacity and items (value and weight) and returns the capacity and a list of items.
loadItems :: String -> IO (Int, [(Int, Int)])
loadItems filename = do
    contents <- readFile filename
    let (capacityLine:itemsLines) = lines contents
    let capacity = read capacityLine :: Int
    let items = map parseItem itemsLines
    return (capacity, items)

-- Parses a line of the file to extract the value and weight of an item.
parseItem :: String -> (Int, Int)
parseItem line = let [value, weight] = map read (words line) in (value, weight)

-- Generates an initial population of random genomes and calculates their fitness.
createInitialPopulation :: Int -> Int -> Int -> [(Int, Int)] -> IO Population
createInitialPopulation populationSize genomeLength capacity items = do
    genomes <- generateGenomes populationSize genomeLength
    let population = buildPopulation genomes capacity items
    return population

-- Generates a list of random genomes of given size.
generateGenomes :: Int -> Int -> IO [Genome]
generateGenomes populationSize genomeLength = replicateM populationSize generateGenome
  where
    generateGenome = replicateM genomeLength randomGene
    randomGene = (< 0.01) <$> (randomRIO (0.0, 1.0) :: IO Double)

-- Builds a population by pairing each genome with its fitness value.
buildPopulation :: [Genome] -> Int -> [(Int, Int)] -> Population
buildPopulation genomes capacity items = map (\genome -> (genome, fitness genome capacity items)) genomes

-- Selects the best individual from a randomly chosen subset of the population.
naturalSelection :: Int -> Population -> IO Individual
naturalSelection groupSize population = do
    competitors <- getRandomCompetitors groupSize population
    return $ findBestCompetitor competitors

-- Randomly selects a specified number of individuals from the population.
getRandomCompetitors :: Int -> Population -> IO [Individual]
getRandomCompetitors n population = replicateM n selectRandomIndividual
  where
    selectRandomIndividual = do
        idx <- randomIndex (length population)
        return $ population !! idx

randomIndex :: Int -> IO Int
randomIndex upperBound = randomRIO (0, upperBound - 1)

-- Finds the individual with the highest fitness value from a list of individuals.
findBestCompetitor :: [Individual] -> Individual
findBestCompetitor = maximumBy (comparing snd)

-- Creates a new genome by combining genes from two parent genomes.
blendGenomes :: Genome -> Genome -> IO Genome
blendGenomes genome1 genome2 = mapM selectBit (zip genome1 genome2)

-- Selects a gene from one of the parents randomly.
selectBit :: (Bool, Bool) -> IO Bool
selectBit (bit1, bit2) = do
    useFirstBit <- randomIO
    return $ if useFirstBit then bit1 else bit2

-- Mutates the genome by flipping genes with a given mutation rate.
mutateGenome :: Double -> Genome -> IO Genome
mutateGenome mutationRate genome = mapM (mutateGene mutationRate) genome

-- Determines whether to mutate a gene based on the mutation rate
mutateGene :: Double -> Bool -> IO Bool
mutateGene mutationRate gene = do
    shouldMutate <- mutateWithRate mutationRate
    return $ if shouldMutate then not gene else gene

-- Generates a random value to decide if mutation should occur.
mutateWithRate :: Double -> IO Bool
mutateWithRate mutationRate = do
    rand <- randomRIO (0.0, 1.0)
    return (rand < mutationRate)

-- Calculates the fitness of a genome based on the total value and weight of included items, considering the knapsack capacity.
fitness :: Genome -> Int -> [(Int, Int)] -> Int
fitness genome capacity items =
    if totalWeight > capacity then 0
    else totalValue
  where
    totalWeight = sum [if gene then weight else 0 | (gene, (_, weight)) <- zip genome items]
    totalValue = sum [if gene then value else 0 | (gene, (value, _)) <- zip genome items]

-- Runs the genetic algorithm, evolving the population over a specified number of generations to find the best individual.
geneticAlgorithm :: Int -> Int -> Int -> Double -> Int -> [(Int, Int)] -> IO Individual
geneticAlgorithm popSize numGenerations groupSize mutationRate capacity items = do
    let genomeSize = length items
    pop <- createInitialPopulation popSize genomeSize capacity items
    evolve pop numGenerations
  where
    evolve pop 0 = return $ maximumBy (comparing snd) pop
    evolve pop gen = do
        parent1 <- naturalSelection groupSize pop
        parent2 <- naturalSelection groupSize pop
        child <- blendGenomes (fst parent1) (fst parent2)
        mutant <- mutateGenome mutationRate child
        let mutantFitness = fitness mutant capacity items
        let newPop = (mutant, mutantFitness) : pop
        evolve newPop (gen - 1)

-- Main, parses command line arguments, loads items from a file, and runs the genetic algorithm.
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

{- 
To run the program, use the following command:

1. run ghc knapsack.hs — to create an object and use program;
2. run  ./knapsack <filename> <popSize> <numGenerations> <groupSize> <mutationRate> where:
  - <filename>: path to the file containing knapsack problem data;
  - <popSize>: size of the population;
  - <numGenerations>: number of generations to evolve;
  - <groupSize>: size of the group for natural selection;
  - <mutationRate>: mutation rate for the genetic algorithm.

for example: run ./knapsack dataset1.txt 250 80 10 0.1
-}