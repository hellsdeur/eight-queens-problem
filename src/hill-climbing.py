import numpy as np
import matplotlib.pyplot as plt
import random
from time import process_time
import statistics


# solução é uma lista de 8 elementos, posição é uma coluna, valor é a linha
def generate_random_solution():
    return [random.randint(0, 7) for i in range(0, 8)]


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

    # metade das colisões = pares de rainhas atacantes
    return collisions//2


# max_clm = maximum consecutive lateral movements
def stochastic_hill_climbing(max_clm):
    clm = max_clm # clm variável
    iterations = 0 # número de iterações
    
    # solução inicial
    best_solution = generate_random_solution()
    best_fitness = calculate_fitness(best_solution)

    # enquanto o fitness mínimo não seja alcançado e ainda restem clms
    while best_fitness > 0 and clm > 0:
        
        # gera uma nova solução e calcula o fitness
        new_solution = generate_random_solution()
        new_fitness = calculate_fitness(new_solution)
        
        # se o novo fitness for menor, ele é tomado como o melhor
        if new_fitness < best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness
            clm = max_clm # reseta o contador de movimentos laterais
        else:
            clm -= 1 # realiza movimento lateral
            
        iterations += 1

    # retorna a melhor solução, a fitness e o número de iterações até a parada
    return [best_solution, best_fitness, iterations]


# execução e análise dos resultados
solutions = [] # soluções para cada uma das 50 execuções
fitnesses = [] # fitness para cada uma das 50 execuções
iterations = [] # quantidades de iterações para cada uma das 50 execuções
runtimes = [] # tempos de execução para cada uma das 50 execuções

# 50 execuções
for i in range(0, 50):
    start = process_time() # tempo da CPU antes da execução
    result = stochastic_hill_climbing(1000) # solução, fitness e iterações
    end = process_time() # tempo da CPU após a execução
    
    runtimes.append(end - start)
    solutions.append(result[0])
    fitnesses.append(result[1])
    iterations.append(result[2])


# estatísticas das iterações
iterations_mean = statistics.mean(iterations)
iterations_stdev = statistics.stdev(iterations)
print("Iterations")
print("Mean: ", iterations_mean)
print("StDev: ± ", iterations_stdev)

# estatísticas dos tempos de execução
runtimes_mean = statistics.mean(runtimes)
runtimes_stdev = statistics.stdev(runtimes)
print("\nRuntimes")
print("Mean: ", runtimes_mean, )
print("StDev: ± ", runtimes_stdev)


# plotando gráfico "execução x número de iterações necessárias"
x = np.linspace(1, 50, 50)
plt.figure(figsize=(20,5))
plt.grid()
plt.axis([0, 50, 0, max(iterations)])
plt.plot(x, iterations)
plt.xlabel("Execution")
plt.ylabel("Minimum iterations needed")
plt.show()


# plotando gráfico "execução x tempo de execução (em segundos)"
x = np.linspace(1, 50, 50)
plt.figure(figsize=(20,5))
plt.grid()
plt.axis([0, 50, 0, max(runtimes)])
plt.plot(x, runtimes)
plt.xlabel("Execution")
plt.ylabel("Runtime (seconds)")
plt.show()

# mapeando fitness às respectivas soluções
map_fitness_to_solutions = zip(fitnesses, solutions)

# ordenando as fitness em ordem crescente
sorted_map = sorted(map_fitness_to_solutions)

# obtendo as 5 melhores soluções
top_five_solutions = sorted_map[:5]

# soluções, fitness e representação visual das 5 melhores soluções
for solution in top_five_solutions:
    print('-'*34)
    print("Solution: ", solution[1])
    print("Fitness: ", solution[0])