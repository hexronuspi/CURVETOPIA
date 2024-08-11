import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import PIL
import random
import pickle
import imageio.v2 as imageio

# define acctivation function params
tansig = lambda n: 2 / (1 + np.exp(-2 * n)) - 1
sigmoid = lambda n: 1 / (1 + np.exp(-n))
hardlim = lambda n: np.where(n >= 0, 1, 0)
purelin = lambda n: n
relu = lambda n: np.maximum(0, n)
square_error = lambda x, y: np.sum(0.5 * (x - y)**2)
sig_prime = lambda z: sigmoid(z) * (1 - sigmoid(z))
relu_prime = lambda z: np.where(z > 0, 1, 0)
softmax = lambda n: np.exp(n - np.max(n, axis=0, keepdims=True)) / np.sum(np.exp(n - np.max(n, axis=0, keepdims=True)), axis=0)
softmax_prime = lambda n: softmax(n) * (1 - softmax(n))
cross_entropy = lambda x, y: -np.sum(x * np.log(y + 1e-8)) 

from random import randint
import copy

class GAKill(Exception):
    def __init__(self, message):
        self.message = message

class Gene:
    fitness = 0
    score = 0
    genotype = []
    cursor = 0

    def __init__(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def evaluate(self, data, targets):
        pass

    def mutate(self, rate):
        gen_len = len(self.genotype)

        idx = np.random.randint(0, gen_len, size=int(round(rate * gen_len)))
        self.genotype[idx] += 0.1 * (2 * np.random.random_sample(len(idx)) - 1)

    def read_genotype(self, delta):
        chunk = self.genotype[self.cursor:self.cursor + delta]
        self.cursor += delta
        return chunk

class GeneticAlgorithm:
    popsize = 0
    error = 1
    epoch = 0
    armageddon = 0

    def __init__(self, epochs, mutation_rate, data, targets, obj, args):
        self.obj = obj
        self.args = args
        self.mutation_rate = mutation_rate
        self.training_data = data
        self.targets = targets
        self.armageddon = epochs

    def populate(self, size):
        self.population = [self.obj(self.args) for _ in range(size)]
        self.popsize = size

    def singleton(self):
        return self.obj(self.args, build=False)

    def evaluate(self):
        for gene in self.population:
            gene.evaluate(self.training_data, self.targets)

        self.population = sorted(self.population, key=lambda gene: gene.fitness)
        self.error = 1 - self.fittest().fitness 

    def crossover(self):
        population = [self.breed(self.roulette(2)) for _ in range(self.popsize)]
        self.population = population

    def breed(self, parents):
        offspring = self.singleton()
        length = parents[0].genotype.size - 1
        cuts = [randint(0, length // 2), randint(length // 2, length)]

        offspring.genotype = np.concatenate((parents[0].genotype[:cuts[0]],
                                              parents[1].genotype[cuts[0]:cuts[1]],
                                              parents[0].genotype[cuts[1]:]))

        offspring.mutate(self.mutation_rate)
        offspring.decode()

        return offspring

    def roulette(self, n):
        choice = self.population[-self.popsize // 2:]
        fitnesses = np.array([x.fitness for x in choice])
        fitnesses /= np.sum(fitnesses)  

        return np.random.choice(choice, n, p=fitnesses)

    def fittest(self):
        return copy.deepcopy(self.population[-1])

    def evolve(self):
        return self.epoch < self.armageddon

class NeuralNet(Gene):
    errors = []
    test_accuracies = []
    train_accuracies = []
    alpha_max = 0.8
    alpha_min = 0.1
    decay_speed = 100

    def __init__(self, args, build=True):
        self.biases = []
        self.weights = []
        self.skeleton = args
        if build:
            self.build(self.skeleton)
            self.encode()

    def build(self, skeleton):
        for i, width in enumerate(skeleton[1:], start=1):
            weights = (2 * np.random.random((width, skeleton[i-1])) - 1)
            biases = (2 * np.random.random(width) - 1)
            self.weights.append(weights)
            self.biases.append(biases)

        self.n = len(self.weights) + 1

    def feed_forward(self, activation):
        zs = []
        activations = [activation]
        z = activation

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        activations.append(softmax(z))
        return activations, zs

    def backpropagate(self, activation, target):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activations, zs = self.feed_forward(activation)
        self.errors[-1] += square_error(target, activations[-1])
        if np.argmax(target) == np.argmax(activations[-1]):
            self.train_accuracies[-1] += 1

        delta = softmax_prime(zs[-1]) * (activations[-1] - target)

        nabla_w[-1] = np.outer(delta, activations[-2])
        nabla_b[-1] = delta

        for i in range(2, self.n):
            delta = np.dot(self.weights[-i+1].T, delta) * sig_prime(zs[-i])

            nabla_w[-i] = np.outer(delta, activations[-i-1])
            nabla_b[-i] = delta

        return nabla_w, nabla_b

    def gradient_descent(self, training_data, targets, epochs, test_data=None, vis=False):
        m = len(training_data)

        for i in range(epochs):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            self.errors.append(0) 
            self.train_accuracies.append(0)

            for tag, img in training_data:
                target = np.array([int(x in tag) for x in targets])
                delta_nabla_w, delta_nabla_b = self.backpropagate(img, target)

                for j in range(self.n - 1):
                    nabla_w[j] += delta_nabla_w[j]
                    nabla_b[j] += delta_nabla_b[j]

            self.weights = [w - (self.learning_rate(i) / m) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (self.learning_rate(i) / m) * nb for b, nb in zip(self.biases, nabla_b)]

            if test_data:
                self.test_accuracies.append(self.validate(targets, test_data))

            self.errors[-1] /= m  
            self.train_accuracies[-1] /= float(m)
            print(f"Epoch: {i} error: {self.errors[-1]} accuracy: {self.test_accuracies[-1]} train_accuracy: {self.train_accuracies[-1]}")

        if vis:
            plt.figure(1)
            plt.plot(range(epochs), self.errors)
            plt.xlabel('Time (Epochs)')
            plt.ylabel('Error')

            plt.figure(2)
            plt.plot(range(epochs), self.train_accuracies, 'g')
            plt.plot(range(epochs), self.test_accuracies, 'r')
            plt.xlabel('Time (Epochs)')
            plt.ylabel('Accuracy')

            plt.show()

    def validate(self, targets, test_data):
        accuracy = 0.0
        for tag, img in test_data:
            target = np.array([int(x in tag) for x in targets])
            activations, zs = self.feed_forward(img)

            if np.argmax(target) == np.argmax(activations[-1]):
                accuracy += 1

        return accuracy / len(test_data)

    def learning_rate(self, i):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-i / self.decay_speed)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.weights, self.biases = pickle.load(f)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump((self.weights, self.biases), f)

    def encode(self):
        genotype = np.array([]) 
        for w, b in zip(self.weights, self.biases):
            genotype = np.concatenate((genotype, w.flatten(), b))

        self.genotype = genotype

    def decode(self):
        self.weights = []
        self.biases = []
        for i, width in enumerate(self.skeleton[1:], start=1):
            d = self.skeleton[i-1] * width
            weights = self.read_genotype(d).reshape(width, self.skeleton[i-1])
            biases = self.read_genotype(width)
            self.weights.append(weights)
            self.biases.append(biases)

        self.cursor = 0
        self.n = len(self.weights) + 1

    def evaluate(self, training_data, targets):
        error = 0

        for tag, img in training_data:
            target = np.array([int(x in tag) for x in targets])
            activations, zs = self.feed_forward(img)
            error += square_error(activations[-1], target)

        self.fitness = 1 - error / len(training_data)

def read_data(path):
    data = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for f in filenames:
            try:
                img_path = os.path.join(dirpath, f)
                img = imageio.imread(img_path, mode='L')
                img = np.ravel(img) / 255.0 
                label = os.path.basename(dirpath)
                data.append((label, img))
            except Exception as e:
                print(f"Error reading file {img_path}: {e}")
    if not data:
        print("Warning: No data found in the specified directory.")
    return data

def train_ga(num_ga_epochs, num_sgd_epochs, vis_flag):
    training_data = read_data('/content/dataset/train/')
    test_data = read_data('/content/dataset/test/')
    random.shuffle(training_data)
    img_len = len(training_data[0][1])
    ga = GeneticAlgorithm(epochs=int(num_ga_epochs),
                          mutation_rate=0.01,
                          data=training_data,
                          targets=targets,
                          obj=NeuralNet,
                          args=[img_len, 10, 4, 7])

    print("Creating population...")
    ga.populate(200)
    print("Initiating GA heuristic approach...")
    errors = []
    while ga.evolve():
        try:
            ga.evaluate()
            ga.crossover()
            ga.epoch += 1

            errors.append(ga.error)
            print("error: " + str(ga.error))
        except GAKill as e:
            print(e.message)
            break

    vis = bool(int(vis_flag))
    if vis:
        fig = plt.figure()
        plt.plot(range(ga.epoch), errors)
        plt.xlabel('Time (Epochs)')
        plt.ylabel('Error')
        plt.show()

    print("--------------------------------------------------------------\n")

    nn = ga.fittest()
    if num_sgd_epochs:
        print("Initiating Gradient Descent optimization...")
        try:
            nn.gradient_descent(training_data, targets, int(num_sgd_epochs), test_data=test_data, vis=vis)
        except GAKill as e:
            print(e.message)

    nn.save("neuralnet.pkt")
    print("Done!")

def validate():
    test_data = read_data('/content/dataset/test/')
    nn = NeuralNet([], build=False)
    nn.load("neuralnet.pkt")
    accuracy = nn.validate(targets, test_data)
    print("Accuracy: " + str(accuracy))

def predict(image_path):
    img = Image.open(image_path).convert('L') 
    img = img.resize((100, 100)) 
    img = np.array(img).flatten()  
    img = img / 255.0 
    
    nn = NeuralNet([], build=False)
    nn.load("neuralnet.pkt")

    activations, zs = nn.feed_forward(img)
    print(targets[np.argmax(activations[-1])])


targets = np.array(['circle', 'ellipse', 'rectangle', 'regularPolygon', 'roundedRectangle', 'star', 'straightLine'])
train_ga(num_ga_epochs=10, num_sgd_epochs=5, vis_flag=1)
validate()


predict('test_image_path.png')
