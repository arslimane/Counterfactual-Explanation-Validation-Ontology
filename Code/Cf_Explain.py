from keras.models import load_model
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.signal import savgol_filter
import topsispy as tp
from IPython.display import clear_output
from tensorflow.keras.losses import MAE
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from mpl_toolkits.mplot3d import Axes3D


class ExplainCf:
    def __init__(self,query_instance,model,population,Total_cf,direction,feature_name,weights=[0.3,0.6,0.1],sign=[1, -1, 1],crossprob=0.8):
        self.query_instance=query_instance
        self.model=model
        self.weights=weights
        self.sign=sign
        self.crossprob=crossprob
        self.mutationprob=1-crossprob
        self.population=population
        self.Total_cf=Total_cf
        self.feature_name=feature_name
        self.direction=direction
    def fitness_func(self,solution,dnr):
        d =  np.linalg.norm(self.query_instance-solution)
        pr1=self.model.predict(np.reshape(self.query_instance,(1,np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])),verbose=0)
        pr2=self.model.predict(np.reshape(solution,(1,np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])),verbose=0)
        if self.direction == "greater":
            valid_target = np.all([a > b for a, b in zip([pr2], [pr1])])
        elif self.direction == "less":
            valid_target = np.all([a < b for a, b in zip([pr2], [pr1])])
        else:
            valid_target = False  # Invalid direction
        y= np.linalg.norm(pr1-pr2) if valid_target else 0
        if (y==0):
            fitness=np.infty
        else:
            if (dnr==0):
                fitness=np.infty
            else:
                fitness =(d/y)+(1/dnr)
        
        return fitness,d,y,dnr
    def nearest_neighbor_distance(self,p,c,j):
        min_d=np.inf
        index=0
        for i in range(len(p)):
            if(j!=i):
                d =  np.linalg.norm(p[i]-c)
                if(min_d>d):
                    index=i
                    min_d=d
        return min_d,index
    def filtering(self,data):
        dt=[]
        while(len(data)!=0):
            dt.append(data[0])
            d=data[1:]
            df=[]
            for i in range(len(d)):
                if(np.linalg.norm(d[i]-dt[-1])!=0):
                    df.append(d[i])
            data=df
        return dt
    def selection(self,data,sign,weights):
        d=[]
        f=[]
        pro=[]
        validity=[]
        diversity=[]
        for i in tqdm(range(len(data)), desc="selection..."):
            d.append(np.reshape(data[i],(np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])))
            min_d=self.nearest_neighbor_distance(data,data[i],i)
            pro.append(self.fitness_func(data[i],min_d[0])[1])
            validity.append(self.fitness_func(data[i],min_d[0])[2])
            diversity.append(self.fitness_func(data[i],min_d[0])[3])
            f.append(self.fitness_func(data[i],min_d[0])[0])
        matrix=np.transpose([validity,pro,diversity])   #validity(maximize),proximity(minimize),diversity(maximize)
        rank=tp.topsis(matrix, weights, sign)
        df=pd.DataFrame({"data":d,"f":rank[1]})
        df=df.sort_values(by=['f'],ascending=False)
        t=df.head(n=self.Total_cf)
        t=np.array(t['data'])
        t2=np.zeros((self.Total_cf,np.shape(self.query_instance)[0],np.shape(self.query_instance)[1]))
        for i in range(self.Total_cf):
            for j in range(np.shape(self.query_instance)[0]):
                for k in range(np.shape(self.query_instance)[1]):
                    t2[i][j][k]=t[i][j][k]
        return t2
    
    def generate(self,data):
        new_generation = []
        if(random.random()<self.crossprob):
            for l in tqdm(range(self.Total_cf), desc="generating..."):
                pi1 = random.randint(0, len(data) - 1)
                pi2 = random.randint(0, len(data) - 1)
                p1 = data[pi1]
                p2 = data[pi2]
                # Generate a random feature number 
                b=random.randint(0,np.shape(self.query_instance)[0]-1)
                f= [0,1,2]
                e=random.randint(b,min(b+20,np.shape(self.query_instance)[0]))
                for i in f:
                    for j in range(b,e):
                        k = p1[j][i]
                        p1[j][i] = p2[j][i]
                        p2[j][i] = k
                
                new_generation.append(np.reshape(p1,(np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])))
            
                
                new_generation.append(np.reshape(p2,(np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])))
            
            
        return new_generation
    
    def mutation(self,data,):
        muta = []
        if random.random() < self.mutationprob:
            for j in tqdm(range(5), desc="mutation..."):
                individual = data[random.randint(0, len(data) - 1)]
                individual = np.reshape(individual, (np.shape(self.query_instance)[0], np.shape(self.query_instance)[1]))
                feature = random.choice(range(np.shape(self.query_instance)[1])) 
                x = individual[feature]
                xl = 0
                xu = 1
                rand = random.random()

                # Randomly determine whether to push towards lower or upper boundary
                mutation_amount = random.uniform(-0.1, 0.1)  # Allow both decrease and increase
                x = np.clip(x + mutation_amount, 0, 1)  # Ensure values stay within [0, 1]
        
                individual[feature] = x
                muta.append(np.reshape(individual, (np.shape(self.query_instance)[0], np.shape(self.query_instance)[1])))
            

        return muta
    def evaluate(self,data):
        c=[]
        m=np.inf
        index=0
        for i in tqdm(range(len(data)), desc="evaluating..."):
            min_d=self.nearest_neighbor_distance(data,data[i],i)
            f=self.fitness_func(data[i],min_d[0])[0]
            if(f<m):
                m=f
                index=i
        return m,index
    def is_convergent(self,array, tolerance=1e-6):
        a=0
        for i in range(1, np.shape(array)[0]):
            if abs(array[i] - array[i-1]) > tolerance:
                a=0
            else:
                a=a+1
        if(a>=50):
            return True
        else :
            return False
        
    def conterfactual_generation_GENO_TOPSIS(self,max_iteration=50):
        r=self.population
        a=[1,2,3]
        iteration=0
        index=0
        metric=0
        best_population=r
        previous_metric=np.inf
    
        try:
            while(not self.is_convergent(a) and iteration<=max_iteration ):
                f=self.selection(r,self.sign,self.weights)
                
                r=self.generate(f)
            
                if(len(r)!=0):
                    r=np.concatenate((f, r), axis=0)
                else:r=f
                m=self.mutation(r)
                if(len(m)!=0):
                    r=np.concatenate((r, m), axis=0)
                d=[]
                f=[]
                pro=[]
                validity=[]
                diversity=[]
                for i in tqdm(range(len(r)), desc="filtering..."):
                    d.append(r[i])
                    min_d=self.nearest_neighbor_distance(r,r[i],i)
                    pro.append(self.fitness_func(r[i],min_d[0])[1])
                    validity.append(self.fitness_func(r[i],min_d[0])[2])
                    diversity.append(self.fitness_func(r[i],min_d[0])[3])
                    f.append(self.fitness_func(r[i],min_d[0])[0])
                    
                matrix=np.transpose([validity,pro,diversity])  
                rank=tp.topsis(matrix, self.weights, self.sign)
                df=pd.DataFrame({"data":d,"f":rank[1]})
                df=df.sort_values(by=['f'],ascending=False)
                r=np.array(df['data'])
                r=self.filtering(r)
                metric,index=self.evaluate(r)
                if(metric<=previous_metric):
                    previous_metric=metric
                    best_population=r
                r=best_population    
                if(metric<=previous_metric):
                    a.append(metric)
                else:a.append(previous_metric)
                iteration+=1
                clear_output(wait=True)
        except(KeyboardInterrupt):
            return best_population,index,a


        return best_population,index,a
    
    def Visualize_fitness(self,a):
        ax=plt.figure(figsize=(10, 5))
        plt.plot(a[3:])
        plt.xlabel('Generation')
        plt.ylabel('best fitness value')
        plt.grid(color='#2A3459')
        ax.set_facecolor('white')
        plt.title('Evolution of fitness values over time')
    def VisualizeDifferencePerTime(self,cefs,index):
        original_series=self.query_instance
        counterfactual_series=cefs[index]
        ab= [np.abs(original_series[:, i]-counterfactual_series[:, i])/np.shape(self.query_instance)[0] for i in range(np.shape(self.query_instance)[1])]
        ax=plt.figure(figsize=(10, 5))
        for i in range(np.shape(self.query_instance)[1]):
            plt.plot(ab[i], label=self.feature_name[i]+' Difference per time point')
        plt.xlabel('Time point')
        plt.ylabel('Difference')
        plt.grid(color='#2A3459')
        plt.legend()
        ax.set_facecolor('white')
        #plt.savefig('C:\\Users\\sarbaoui01\\OneDrive - INSA Strasbourg\\Desktop\\Counterfactuals\\Vcurrent\\cf_best_fD.pdf')
        plt.show()

    def VisualizeMAEPerFeature(self,cefs,index):
        original_series=self.query_instance
        counterfactual_series=cefs[index]
        abs_per_feature = [np.sum(np.abs(original_series[:, i]-counterfactual_series[:, i]))/np.shape(self.query_instance)[0] for i in range(np.shape(self.query_instance)[1])]
        ax=plt.figure(figsize=(10, 5))
        plt.bar(self.feature_name, abs_per_feature, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('MAE')
        plt.grid(color='#2A3459')
        plt.title('MAE per Feature Comparison')
        ax.set_facecolor('white')
        #plt.savefig('C:\\Users\\sarbaoui01\\OneDrive - INSA Strasbourg\\Desktop\\Counterfactuals\\Vcurrent\\cf_best_mae_perF.pdf')
        plt.show()

    def compare(self,x2,time,f):
        names=self.feature_name
        feature=np.where(np.array(names) ==f)[0][0]
        d1=[self.query_instance[i][feature] for i in range(time,np.shape(self.query_instance)[0])]
        d2=[x2[i][feature] for i in range(time,np.shape(self.query_instance)[0])]
        plt.style.use("classic")
        colors = [
            '#08F7FE',  # teal/cyan
            '#FE53BB',  # pink
            '#F5D300',  # yellow
            '#00ff41',  # matrix green
        ]
        df = pd.DataFrame({'Original_data': list(d1),
                        'Counterfactual_data': list(d2)})
        fig, ax = plt.subplots(figsize=(10,5))
        df.plot(marker='.', color=colors, ax=ax)
        ax.grid(color='#2A3459')
        plt.xlabel('Time step', fontweight = 'bold', fontsize='large')
        plt.ylabel('Values', fontweight = 'bold', fontsize='large')
        plt.title('Comparision between the original and counterfactual values of feature '+f)
        plt.legend(loc='lower right')
        fig.set_facecolor('white')
        #plt.savefig('C:\\Users\\sarbaoui01\\OneDrive - INSA Strasbourg\\Desktop\\Counterfactuals\\Vcurrent\\cf_best_'+name+f+'.pdf')
        plt.show()

    def VisualizeComparision(self,cefs,index,startTime=0):
        for i in range(np.shape(self.query_instance)[1]):
            self.compare(cefs[index],startTime,self.feature_name[i])

    def VisualizePrediction(self,cefs,index,outPutName):
        pr1=self.model.predict(np.reshape(cefs[index],(1,np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])),verbose=0)
        pr2=self.model.predict(np.reshape(self.query_instance,(1,np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])),verbose=0)
        ax=plt.figure(figsize=(10, 5))
        plt.plot(pr1[0],label="Counterfactual")
        plt.plot(pr2[0],label="Original")
        #plt.plot(Y1[410])
        plt.xlabel('Time step')
        plt.ylabel(outPutName)
        plt.grid(color='#2A3459')
        plt.title(outPutName+' values for the counterfactual and the original')
        plt.legend()
        ax.set_facecolor('white')

    def VisualizeDistribution(self,cefs):
        proximity_values = []
        validity_values = []
        diversity_values = []
        r=cefs
        for i in tqdm(range(len(r)), desc="analysing..."):
                min_d=self.nearest_neighbor_distance(r,r[i],i)
                proximity_values.append(self.fitness_func(r[i],min_d[0])[1])
                validity_values.append(self.fitness_func(r[i],min_d[0])[2])
                diversity_values.append(self.fitness_func(r[i],min_d[0])[3])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the point (x, y, z)
        ax.scatter(proximity_values, validity_values,  diversity_values, color='darkblue', s=30)  # s is the size of the point

        for i in range(len(proximity_values)):
            ax.text(proximity_values[i], validity_values[i],  diversity_values[i], '%d' % i, color='black', fontsize=10)
        # Set labels for axes
        ax.set_xlabel('Proximity')
        ax.set_ylabel('Validity')
        ax.set_zlabel('Diversity')
        fig.set_facecolor('white')

        # Set the title
        ax.set_title('Proximity vs Validity vs Diversity')
        #plt.savefig('C:\\Users\\sarbaoui01\\OneDrive - INSA Strasbourg\\Desktop\\Counterfactuals\\Vcurrent\\GA_analysing.pdf')

        # Display the plot
        plt.show()
                

#-------------------------------------------------------------------------------
# -----------------------------------------------------------------------------                
    def init_individual(self,icls, points, index):
        individual = icls(points[index])
        return individual

# Custom function to initialize the population from the points array
    def init_population(self,pcls, ind_init, points):
        return pcls(ind_init(creator.Individual, points, i) for i in range(len(points)))
    
    def custom_crossover(self,ind1, ind2):
        if(np.ndim(ind1)==2 and np.ndim(ind2)==2):
            cxpoints = random.randint(1, min(np.shape(ind1)[0],np.shape(ind2)[0]) - 1)
            cxpointe =random.randint(cxpoints, min(np.shape(ind1)[0],np.shape(ind2)[0]) - 1)
            for i in range(np.shape(self.query_instance)[1]):
                ind1[cxpoints:cxpointe,i], ind2[cxpoints:cxpointe,i] = ind2[cxpoints:cxpointe,i], ind1[cxpoints:cxpointe,i]
        return ind1, ind2


    def evaluate2(self,individual):
        individual = np.array(individual)
        
        # Proximity (distance to the input data)
        proximity = np.linalg.norm(individual - self.query_instance)
        
        # Validity (model prediction difference)
        original_pred = self.model.predict(np.reshape(self.query_instance,(1,np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])),verbose=0)[0]
        ctf_pred = self.model.predict(np.reshape(individual,(1,np.shape(self.query_instance)[0],np.shape(self.query_instance)[1])),verbose=0)[0]
        validity = np.linalg.norm(original_pred - ctf_pred)
        
        # Diversity (distance to the nearest neighbor in the population)
        
        if (len(self.population) > 1):
            reshaped_population = np.array([ind for ind in self.population if not np.array_equal(ind, individual)])
            reshaped_individual = individual
            distances = np.linalg.norm([reshaped_individual]-reshaped_population)
            diversity = np.min(distances)
        else:
            diversity = 0  # No other individuals to compare with

        return proximity, -validity, -diversity

    def mutPolynomialBounded(self,individual, eta, low, up, indpb):
        feature=random.choice(range(np.shape(self.query_instance)[1]))
        if random.random() <= self.mutationprob:
                x = individual[:,feature]
                xl = 0
                xu = 1
                delta1 = x
                delta2 = xu - x 
                rand = random.random()
                mut_pow = 1.0 / (eta + 1)

                if rand <= 0.5:
                    # Mutate towards the lower boundary
                    delta = (xl - x) * random.uniform(0, 0.5)  # Add a fraction towards 0
                else:
                    # Mutate towards the upper boundary
                    delta = (xu - x) * random.uniform(0, 0.5)  # Add a fraction towards 1

                # Update the selected feature
                # Randomly determine whether to push towards lower or upper boundary
                mutation_amount = random.uniform(-0.1, 0.1)  # Allow both decrease and increase
                x = np.clip(x + mutation_amount, 0, 1)  # Ensure values stay within [0, 1]
            
                #x = min(max(x, xl), xu)
                individual[:,feature] = x
        return individual,



    def dominates(self,fitness1, fitness2):
        """
        Check if fitness1 dominates fitness2 (all objectives are equal or better).
        """
        return all(f1 <= f2 for f1, f2 in zip(fitness1, fitness2)) and any(f1 < f2 for f1, f2 in zip(fitness1, fitness2))
    def sortNondominated(self,individuals, n):
        pareto_fronts = []
        dom_count = {i: 0 for i in range(n)}
        dominated_set = {i: [] for i in range(n)}
        fronts = []
        # Step 1: Dominance comparison
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(individuals[i].fitness.values, individuals[j].fitness.values):
                        dominated_set[i].append(j)
                    elif self.dominates(individuals[j].fitness.values, individuals[i].fitness.values):
                        dom_count[i] += 1
        
        # Step 2: Building fronts
        current_front = []
        for i in range(n):
            if dom_count[i] == 0:
                individuals[i].fitness.rank = 0
                current_front.append(i)
                
        pareto_fronts.append(current_front[:])
        front_idx = 0
        
        while len(pareto_fronts[front_idx]) > 0:
            next_front = []
            for i in pareto_fronts[front_idx]:
                for j in dominated_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        individuals[j].fitness.rank = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            pareto_fronts.append(next_front[:])
    
        # Step 3: Sorting within fronts (using crowding distance as secondary criterion if needed)
        for front in pareto_fronts[:-1]:
            front.sort(key=lambda x: individuals[x].fitness.values)  # Sort by fitness values
            
        return pareto_fronts[:-1]  # Exclude the last empty front
    
    def customSelectNSGA2(self,individuals,n):
        # Assign non-dominated ranks and crowding distances
        
        pareto_fronts =self.sortNondominated(individuals, len(individuals))
        
        selected = []
        current_rank = 0
        while len(selected) < n:
            if current_rank < len(pareto_fronts):
                front = pareto_fronts[current_rank]
                if len(selected) + len(front) <= n:
                    selected.extend(front)
                else:
                    # Sort by crowding distance within the front
                    sorted_front = sorted(front, key=lambda x: np.sum(x.fitness.crowding_dist), reverse=True)
                    remaining = n- len(selected)
                    selected.extend(sorted_front[:remaining])
            else:
                break
            current_rank += 1
        
        return selected


    def conterfactual_generation_NSGA2(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        toolbox.register("individual", self.init_individual, creator.Individual, self.population)
        toolbox.register("population", self.init_population, list, self.init_individual, self.population)
        toolbox.register("attr_float", np.random.uniform, 0,1, np.shape(self.query_instance)[0])
        toolbox.register("mate", self.custom_crossover)
        toolbox.register("mutate", self.mutPolynomialBounded, eta=1.0, low=0, up=1, indpb=self.mutationprob)
        toolbox.register("select", self.customSelectNSGA2)
        toolbox.register("evaluate", self.evaluate2)
        population = toolbox.population()
        ngen = 50
        cxpb = self.crossprob
        mutpb = self.mutationprob

        for gen in tqdm(range(ngen), desc="generating..."):
            # Select parents for crossover
            
            parents = toolbox.select(population, len(population))
        
            p=[]
            for i in parents:
                p.append(population[i])
            parents=p
            # Clone the selected individuals
            offspring = [toolbox.clone(ind) for ind in parents]
            
            # Apply crossover and mutation to offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    #del child1.fitness.values, child2.fitness.values
            
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the fitness of the offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the current population with the offspring
            population[:] = offspring
            clear_output(wait=True)

        # Extracting the Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=False)[0]
        return pareto_front
            