import torch
from ViT import VisionTransformer
import numpy as np
import torch.nn as nn
from math import ceil, exp
from random import choice, shuffle

class CSO(object):
    def __init__(self):
        self.__Gbest = []

    def _set_Gbest(self, Gbest):
        self.__Gbest = Gbest

    def get_Gbest(self):
        """Return the best position of algorithm (return type: list)"""

        return list(self.__Gbest)

    def save_agent_states(self, file_path, agent_states):
        torch.save(agent_states, file_path)

    def load_agent_states(self, file_path, model):   
        checkpoint = torch.load(file_path) 
        model.load_state_dict(checkpoint)
        return model

    def update_model_weights_roosters(self, model, sigma, lb, ub):
        for param in model.parameters():
            param.data = param.data * (1 + torch.rand_like(param.data) * sigma)
            param.data = torch.clamp(param.data, lb, ub)
        return model

    def update_model_weights_hines(self, model, i, r1, r2, s1, s2, lb, ub):
        model = self.load_agent_states(F'agents_{i}.pth', model)
        model_1 = self.load_agent_states(F'pbest_{i}.pth', model)
        model_2 = self.load_agent_states(F'pbest_{r1}.pth', model)
        model_3 = self.load_agent_states(F'pbest_{r2}.pth', model)

        for param, param_1, param_2, param_3 in zip(model.parameters(), model_1.parameters(), model_2.parameters(), model_3.parameters()):
            param.data = param_1.data + s1 * torch.rand_like(param_1.data) * (
                            param_2.data - param_1.data) + s2 * torch.rand_like(param_1.data)*(
                            param_3.data - param_1.data)

            param.data = torch.clamp(param.data, lb, ub)
        return model

    def update_model_weights_chicks(self, model, i1, i2, FL, lb, ub):
        model = self.load_agent_states(F'agents_{i1}.pth', model)
        model_1 = self.load_agent_states(F'pbest_{i1}.pth', model)
        model_2 = self.load_agent_states(F'pbest_{i2}.pth', model)

        for param, param_1, param_2 in zip(model.parameters(), model_1.parameters(), model_2.parameters()):
            param.data = param_1.data * FL * (param_2.data - param_1.data)
            param.data = torch.clamp(param.data, lb, ub)
        return model
    
    def loss(self, model, name, i):
        criterion = nn.CrossEntropyLoss()
        model = self.load_agent_states(F'{name}_{i}.pth', model)
        outputs = model(self.images, include_head=True)
        loss = criterion(outputs, self.labels)
        return loss

    def optimize(self, config, conv_stem_configs, n, train_dataset, lb, ub, iteration, G=5, FL=0.5):
        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param G: after what time relationship will be upgraded (default
        value is 5)
        :param FL: parameter, which means that the chick would follow its
        mother to forage for food (0 < FL < 2. Default value is 0.5)
        """

        rn = ceil(0.15 * n)
        hn = ceil(0.7 * n)
        cn = n - rn - hn
        mn = ceil(0.2 * n)
        self.__agents = []
        for i in range(n):
            model = VisionTransformer(
                            config, 
                            config["image_size"], 
                            config["patch_size"], 
                            config["n_embd"], 
                            config["num_classes"],
                            conv_stem_configs).to("cuda")
            self.save_agent_states(F'agents_{i}.pth',model.state_dict())
            self.__agents.append(i)

        pbest = []
        for i in range(n):
            model = VisionTransformer(
                            config, 
                            config["image_size"], 
                            config["patch_size"], 
                            config["n_embd"], 
                            config["num_classes"],
                            conv_stem_configs).to("cuda")
            self.save_agent_states(F'pbest_{i}.pth',model.state_dict())
            pbest.append(i)

        batch = next(iter(train_dataset))
        self.images, self.labels = batch
        self.images = self.images.to("cuda")
        self.labels = self.labels.to("cuda")

        fitness = [self.loss(model, "agents", x).cpu().detach().numpy() for x in self.__agents]
        pfit = fitness

        Pbest = self.__agents[np.array(fitness).argmin()]
        Gbest = Pbest

        for t in range(iteration):
            batch = next(iter(train_dataset))
            self.images, self.labels = batch
            self.images = self.images.to("cuda")
            self.labels = self.labels.to("cuda")

            if t % G == 0:
                chickens = self.__update_relationship(n, model, rn, hn,
                                                      cn, mn)
                roosters, hines, chicks = chickens

            for i in roosters:
                k = choice(roosters)
                while k == i:
                    k = choice(roosters)

                if pfit[i] <= pfit[k]:
                    sigma = 1
                else:
                    sigma = exp((pfit[k] - pfit[i]) / (abs(pfit[i]) + 0.01))

                model = self.load_agent_states(F'pbest_{pbest[i]}.pth', model)
                model = self.update_model_weights_roosters(model, sigma, lb, ub)
                self.save_agent_states(F'agents_{i}.pth',model.state_dict())

            for i in hines:
                r1 = i[1]
                r2 = choice([choice(roosters), choice(hines)[0]])
                while r2 == r1:
                    r2 = choice([choice(roosters), choice(hines)[0]])

                s1 = exp((pfit[i[0]] - pfit[r1]) / (abs(pfit[i[0]]) + 0.01))

                try:
                    s2 = exp(pfit[r2] - pfit[i[0]])
                except OverflowError:
                    s2 = float('inf')

                model = self.update_model_weights_hines(model, i[0], r1, r2, s1, s2, lb, ub)
                self.save_agent_states(F'agents_{i[0]}.pth',model.state_dict())

            for i in chicks:
                model = self.update_model_weights_chicks(model, i[0], i[1], FL, lb, ub)
                self.save_agent_states(F'agents_{i[0]}.pth',model.state_dict())

            fitness = []
            for x in self.__agents:
                fit = self.loss(model, "agents", x)
                if str(fit) == 'nan':
                    model = self.load_agent_states(F'agents_{x}.pth', model)
                    for param in model.parameters():
                        param.data = torch.rand_like(param.data)
                        param.data = torch.clamp(param.data, lb, ub)
                    self.save_agent_states(F'agents_{x}.pth',model.state_dict())
                fitness.append(fit.cpu().detach().numpy())

            for i in range(n):
                if fitness[i] < pfit[i]:
                    pfit[i] = fitness[i]
                    pbest[i] = self.__agents[i]
            print("Best fitness: ", fitness[np.array(fitness).argmin()])
            Pbest = self.__agents[np.array(fitness).argmin()]
            if self.loss(model, "pbest", Pbest) < self.loss(model, "agents", Gbest):
                Gbest = Pbest

        self._set_Gbest(Gbest)

    def __update_relationship(self, n, model, rn, hn, cn, mn):
        fitness = [(self.loss(model, "agents", self.__agents[i]), i) for i in range(n)]
        fitness.sort()

        chickens = [i[1] for i in fitness]
        roosters = chickens[:rn]
        hines = chickens[rn:-cn]
        chicks = chickens[-cn:]

        shuffle(hines)
        mothers = hines[:mn]

        for i in range(cn):
            chicks[i] = chicks[i], choice(mothers)

        for i in range(hn):
            hines[i] = hines[i], choice(roosters)

        return roosters, hines, chicks
