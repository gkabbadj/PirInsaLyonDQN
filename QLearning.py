import random
import matplotlib.pyplot as plt
#from math import *

#Ce programme implémente SARSA et le Q learning sur le jeu de la falaise
#Présenté dans le livre de Sutton et Barto

class QLearning:

    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.1
        self.epsilon = 0.1
        self.min_epsilon = 0.01
        self.decay_rate = 1
        self.reward = 0
        self.index = 36
        self.qTable = []
        self.qTable2 = []
        self.score = 0
        i = 0
        while i < 4*48:
            self.qTable.append(0)
            self.qTable2.append(0)
            i = i+1

    def cliff(self):
        if self.index < 47 and self.index > 36:
            return True

    def possibles(self):
        a = []
        if self.index % 12 != 11:
            a.append(0)
        if self.index % 12 != 0:
            a.append(1)
        if self.index > 11:
            a.append(2)
        if self.index < 36:
            a.append(3)
        return a

    def val_opt1(self,alpha):
        a = 4*self.index
        maxi = self.possibles()[0]
        if alpha == 1:
            for i in self.possibles():
                val = self.qTable[a+i]
                if val >= self.qTable[a+maxi]:
                    maxi = i
        if alpha == 2:
            for i in self.possibles():
                val = self.qTable2[a+i]
                if val >= self.qTable2[a+maxi]:
                    maxi = i
        return maxi

    def val_opt2(self):
        a = 4*self.index
        maxi = self.qTable[a+self.possibles()[0]]
        for i in self.possibles():
            val = self.qTable[a+i]
            if val >= maxi:
                maxi = val
        #print("Le choix pris est "+str(maxi))
        return maxi

    def hasard(self, tab):
        return tab[random.randrange(len(tab))]
    
    def mouvement(self, decision):
        #mouvement à effectuer
        if decision == 0:  # droite
            self.index += 1
        elif decision == 1:  # gauche
            self.index -= 1
        elif decision == 2:  # haut
            self.index -= 12
        elif decision == 3:  # bas
            self.index += 12

        #calcul de la récompense
        if self.cliff():
            self.reward = -100
        elif self.index == 47:
            self.reward = 0
        else:
            self.reward = -1

    def jouer1(self): #Qlearning
        add = 0
        tab = []
        for i in range(1000):
            a = 0
            self.index = 36
            self.score = 0
            fini = False
            self.reward = 0
            while not fini:
                prev = self.index
                #print(prev)
                if random.random() < self.epsilon:
                    #print(self.possibles())
                    action = self.hasard(self.possibles())
                    self.mouvement(action)
                else:
                    action = self.val_opt1(1)
                    self.mouvement(action)
                self.qTable[prev*4+action] += self.alpha * (self.reward + self.gamma * self.val_opt2() - self.qTable[prev*4+action])
                self.score += self.reward
                retour = self.cliff()
                if retour or self.index == 47:
                    self.index = 36
                    #fini = True
                    #print("L'épisode "+str(i)+" s'est fini à l'instant "+str(a)+" et au score de "+str(self.score))
                if a == 20:
                    fini = True
                    print("Avec QL, l'épisode " + str(i) + " s'est fini au score de " + str(self.score))
                    add += self.score
                    if i % 10 == 0:
                        add = add/10
                        tab.append(add)
                        add = 0
                a = a+1
        return tab
            #if self.epsilon > self.min_epsilon:
                #self.epsilon *= self.decay_rate
               # print(self.epsilon)


    def jouer2(self):   #SARSA
        tab = []
        add = 0
        for i in range(1000):
            a = 0
            self.index = 36
            self.score = 0
            fini = False
            self.reward = 0
            if random.random() < self.epsilon:
                action = self.hasard(self.possibles())
            else:
                action = self.val_opt1(2)
            while not fini:
                prev = self.index
                self.mouvement(action)
                # print(prev)
                if random.random() < self.epsilon:
                    # print(self.possibles())
                    action1 = self.hasard(self.possibles())
                    #self.mouvement(action)
                else:
                    action1 = self.val_opt1(2)
                    #self.mouvement(action)
                self.qTable2[prev * 4 + action] += self.alpha * (
                            self.reward + self.gamma * self.qTable2[self.index*4+action1] - self.qTable2[prev * 4 + action])
                self.score += self.reward
                retour = self.cliff()
                if retour or self.index == 47:
                    self.index = 36
                    # fini = True
                    #print("L'épisode "+str(i)+" s'est fini à l'instant "+str(a)+" et au score de "+str(self.score))
                if a == 20:
                    fini = True
                    print("Avec Sarsa, l'épisode " + str(i) + " s'est fini au score de " + str(self.score))
                    add += self.score
                    if i % 10 == 0:
                        add = add/10
                        tab.append(add)
                        add = 0
                action = action1
                a = a + 1
        return tab



#print(monde)
if __name__ == "__main__":
    Test = QLearning()
    Tableau1 = Test.jouer1()
    Test1 = QLearning()
    Tableau2 = Test1.jouer2()
    plt.plot(range(0,1000,10),Tableau1,'r--',range(0,1000,10),Tableau2,'b--')
    plt.ylabel("Comparaison de Sarsa et de Q-learning")
    plt.show()


