import numpy as np
import pandas as pd
import random
from time import process_time
import statistics


# solução é uma lista de 8 elementos, posição é uma coluna, valor é a linha
def generate_random_solution():
    return [random.randint(0, 7) for i in range(0, 8)]


# codificação decimal para codificação binária
def to_binary(decimal_solution):
    return [bin(position)[2:].zfill(3) for position in decimal_solution]

# codificação binária para codificação decimal
def to_decimal(binary_solution):
    return [int(position, 2) for position in binary_solution]


# gerar população de k indivíduos já em binário
def generate_population(k):
    return [to_binary(generate_random_solution()) for i in range(0, k)]


# fitness é a quantidade de pares de rainhas se atacando
def calculate_fitness(solution):
    collisions = 0
    
    # iterando sobre cada rainha
    for column in range(0, 8):
        line = solution[column]

        # contar colisões com a rainha atual na mesma linha
        collisions += solution.count(line) - 1
        
        # contar colisões com a rainha atual na diagonal superior esquerda
        for i, j in zip(range(line-1, -1, -1), range(column-1, -1, -1)):
            if solution[j] == i:
                collisions += 1

        # contar colisões com a rainha atual na diagonal inferior esquerda
        for i, j in zip(range(line+1, 8, +1), range(column-1, -1, -1)):
            if solution[j] == i:
                collisions += 1
        
        # contar colisões com a rainha atual na diagonal superior direita
        for i, j in zip(range(line-1, -1, -1), range(column+1, 8, +1)):
            if solution[j] == i:
                collisions += 1
        
        # contar colisões com a rainha atual na diagonal inferior direita
        for i, j in zip(range(line+1, 8, +1), range(column+1, 8, +1)):
            if solution[j] == i:
                collisions += 1

    return collisions//2


# estratégia da roleta, quanto maior o fitness maior a chance de ser escolhido
def select_parents(population, fitnesses):
    sum_fitnesses = sum(fitnesses)
    # distr. de prob. para as fitness
    distribution = [fit/sum_fitnesses for fit in fitnesses]

    # 2 indexes escolhidos dentre os k indivíduos, sem repetição, de acordo com a distribuição
    indexes = np.random.choice(len(population), 2, p=distribution, replace=False)
    
    return [population[i] for i in indexes]


# estratégia do ponto de corte, o crossover tem uma prob. cross_rate de acontecer
def crossover(parents, cross_rate):
    # se houver cruzamento, trocar material genético
    if random.uniform(0, 1) <= cross_rate:
        children = []
        
        # ponto de corte, aleatório entre 1 e 7
        cutoff = random.randint(1, 7)
        
        # troca de material genético
        child_1 = parents[0][0:cutoff] + parents[1][cutoff:]
        child_2 = parents[1][0:cutoff] + parents[0][cutoff:]
        
        children.append(child_1)
        children.append(child_2)
        
        return children
        
    return parents # se o cruzamento não for possível, os pais se tornam filhos


# estratégia do bit flip, onde um dos 24 bits de um filho é trocado
def mutate(child, mutate_rate):
    # se a mutação for possível, alterar material genético por bit flip
    if random.uniform(0, 1) <= mutate_rate:
        mutated_child = []
        
        # unificar todo o material genético em uma lista de 24 bits
        unified_genes = list("".join(child))
        bit = random.randint(0, 23) # bit a ser alterado
        
        # troca do bit
        if unified_genes[bit] == "0":
            unified_genes[bit] = "1"
        else:
            unified_genes[bit] = "0"
        
        # junta os bits em uma única string
        mutated_string = "".join(unified_genes)
        
        # divide os bits de 3 em 3, obtendo a forma original
        for i in range(0, 24, 3):
            mutated_child.append(mutated_string[i:i+3])
        
        return mutated_child

    return child # se a mutação não for possível, o filho não se altera


# estratégia elitista, onde os k indivíduos com as menores fitness sobrevivem
def select_survivors(map_fitness_to_individual, k):
    sorted_map = sorted(map_fitness_to_individual) # ordenando as fitness em ordem crescente
    return sorted_map[:k] # retornando os k sobreviventes


# tam. da população k, a tx. de crossover cross_rate, a tx. de mutação mutate_rate e máximo de gerações gen
def genetic_algorithm(k, cross_rate, mutate_rate, gen):
    max_gen = gen # usar a original para tirar a diferença
    population = generate_population(k)
    fitnesses = [calculate_fitness(to_decimal(individual)) for individual in population] # fitness para cada indivíduo
    best_solution = min(zip(fitnesses, population)) # 2-tupla que armazena a menor fitness e o melhor indivíduo da população
    
    while best_solution[0] > 0 and gen > 0:
        # seleção dos pais
        parents = select_parents(population, fitnesses)
        
        # cruzamento
        children = crossover(parents, cross_rate)

        # mutação e avaliação dos filhos
        mutated_children = [mutate(child, mutate_rate) for child in children]
        fitnesses_children = [calculate_fitness(to_decimal(individual)) for individual in mutated_children]
        
        # adicionando os filhos na população e suas fitness na lista de fitnesses
        population.extend(mutated_children)
        fitnesses.extend(fitnesses_children)
        
        # mapeando fitness aos respectivos individuos, k + 2 tuplas
        map_fitness_to_individual = zip(fitnesses, population)

        # selecionando as k melhores tuplas
        survivors = select_survivors(map_fitness_to_individual, k)
        
        # obtendo a melhor solução até o momento
        best_solution = min(survivors)
        
        # desfazendo o mapeamento
        fitnesses, population = [list(tup) for tup in zip(*survivors)]
        
        gen -= 1
    
    # retorna a melhor solução, sua fitness e a geração de parada
    return [best_solution[1], best_solution[0], max_gen-gen]

execution = []
solutions = [] # soluções para cada uma das 50 execuções
fitnesses = [] # fitness para cada uma das 50 execuções
generations = [] # geração de parada para cada uma das 50 execuções
runtimes = [] # tempos de execução para cada uma das 50 execuções


# 50 execuções
for i in range(1, 51):
    execution.append(i) # número da execução
    start = process_time() # tempo da CPU antes da execução
    result = genetic_algorithm(20, 0.8, 0.03, 1000) # solução, fitness e geração de parada
    end = process_time() # tempo da CPU após a execução
    
    runtimes.append((end - start)*1000)
    solutions.append(result[0])
    fitnesses.append(result[1])
    generations.append(result[2])
    
    
df = pd.DataFrame({ "Runtime": execution,
                    "Best Solution": solutions,
                    "Fitness": fitnesses,
                    "Runtime (ms)": runtimes,
                    "Generation": generations})


# estatísticas das fitness
fitnesses_mean = statistics.mean(fitnesses)
fitnesses_stdev = statistics.stdev(fitnesses)
print("Fitnesses")
print("Mean: ", fitnesses_mean)
print("StDev: ± ", fitnesses_stdev)

# estatísticas dos tempos de execução
runtimes_mean = statistics.mean(runtimes)
runtimes_stdev = statistics.stdev(runtimes)
print("\nRuntimes")
print("Mean: ", runtimes_mean, )
print("StDev: ± ", runtimes_stdev)

# estatísticas das gerações
generations_mean = statistics.mean(generations)
generations_stdev = statistics.stdev(generations)
print("\nGenerations")
print("Mean: ", generations_mean, )
print("StDev: ± ", generations_stdev)

        
# 5 melhores soluções encontradas

# mapeando fitness às respectivas soluções
map_fitness_to_solutions = zip(fitnesses, solutions)

# ordenando as fitness em ordem crescente
sorted_map = sorted(map_fitness_to_solutions)

# obtendo as 5 melhores soluções
top_five_solutions = sorted_map[:5]

for solution in top_five_solutions:
    print('-'*34)
    print("Solution: ", solution[1])
    print("Fitness: ", solution[0])
