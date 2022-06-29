import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

#This .py file provides the LinUCB code in order to implement it with batches. This reduce the waiting time on a live data implementation.

class LinUCB_Arm():
    
    def __init__(self, Arm_index, Dimension, Alpha):
        self.Arm_index = Arm_index
        self.Alpha = Alpha
        self.A = np.identity(Dimension)
        self.b = np.zeros([Dimension, 1])
        
    def Compute_UCB(self, Features):
        A_inv = np.linalg.inv(self.A)
        
        # Theta computation
        self.Theta_a = np.dot(A_inv, self.b)
        
        # Reshape features
        x = Features.reshape([-1, 1])
        
        # Compute UCB
        UCB = np.dot(self.Theta_a.T, x) +  self.Alpha * np.sqrt(np.dot(x.T, np.dot(A_inv,x)))
        
        return UCB
    
    def Update_Reward(self, reward, Features):
        # Reshape features
        x = Features.reshape([-1,1])
        
        # Update A 
        self.A += np.dot(x, x.T)
        
        # Update b
        self.b += reward * x

class ComputeUCB():
    
    def __init__(self, K_arms, Dimension, Alpha):
        self.K_arms = K_arms
        self.LinUCB_Arms = [LinUCB_Arm(Arm_index = i, Dimension = Dimension, Alpha = Alpha) for i in range(K_arms)]
        
    def Select_Arm(self, features):
        # Initiate ucb to be surpassed by first arm.
        Highest_ucb = -100000000
        
        # Track candidate arms if we need to broke ties arbitrary
        Candidate_arms = []
        
        # Compute UCB for each arm (game experience) given "x" context
        for Arm_index in range(self.K_arms):
            Arm_ucb = self.LinUCB_Arms[Arm_index].Compute_UCB(features)
            if Arm_ucb > Highest_ucb:
                Highest_ucb = Arm_ucb
                Candidate_arms = [Arm_index]

            # Add candidate when we have ties
            if Arm_ucb == Highest_ucb:
                Candidate_arms.append(Arm_index)
        
        # Chose the best arm among all candidates randomly
        Best_Arm = np.random.choice(Candidate_arms)
        
        return Best_Arm

def LinUCB_Algorithm(Data, Alpha, Features, Reward_Feature, batches):
    #Extract number of Arms given the dataset.
    K_arms = np.unique(Data.ab_test_experience_id).shape[0]
    
    #Create UCB class (arms included)
    ComputeUCB_object = ComputeUCB(K_arms = K_arms, Dimension = len(Features), Alpha = Alpha)
    
    '''
    Initiate following metrics/markers:
    - Check how many times the best arm is the applied for the user
    - Cumulative reward we get
    - Times each arm is selected
    - Rewards obtained for each arm
    - Path the algorithm takes (which arm is selected at each step)
    '''
    Times_GameExperienceIsBestArm = 0
    Cumulative_Reward = 0
    Aligned_Reward = []
    Arms_Chosen = [1] * K_arms
    Arms_Rewards = [0] * K_arms
    Path_Algorithm = [0]
    
    Features_Arms_Frames = {}
    Ev_all = {}
    Cont_all = {}
    for x in range(K_arms):
        Features_Arms_Frames["Features_Arm{0}".format(x)] = pd.DataFrame(columns = Data[Features].columns)
        Ev_all["Ev{0}".format(x)] = 0
        Cont_all["Cont{0}".format(x)] = 0
    
    #We sample the data in order to have different user order selection
    Data = Data.sample(frac=1).reset_index(drop=True)
    
    split_dataframes = split_dataframe_by_position(Data, batches)
    aux_bests = []
    aux_which = []
    for time in tqdm(range(batches)):
        aux_bests = []
        aux_which = []
        print("A")
        for row in range(split_dataframes[time].shape[0]):
            #We track the A/B test experience variant for the user
            Which_Arm = split_dataframes[time].ab_test_experience_id.iloc[row]
        
            #We select user features
            FeaturesArm = split_dataframes[time][Features].iloc[row,].values
        
            #We compute which is the best arm by the Lin-UCB algo
            Best_Arm = ComputeUCB_object.Select_Arm(FeaturesArm)
            aux_which.append(Which_Arm)
            aux_bests.append(Best_Arm)
            #If the best arm is the same as the A/B test experience allocated to the user, we observe the reward and update everything
        
        for row in range(len(aux_bests)):
        
            if aux_bests[row] == aux_which[row]:
                Reward = split_dataframes[time][Reward_Feature].iloc[row]
                FeaturesArm = split_dataframes[time][Features].iloc[row,].values
            
                for x in range(K_arms):
                    Ev_all["Ev{0}".format(x)] = np.append(Ev_all["Ev{0}".format(x)], 0)
            
                Cont_all["Cont{}".format(aux_bests[row])] = Cont_all["Cont{}".format(aux_bests[row])] + 1
                Features_Arms_Frames["Features_Arm{}".format(aux_bests[row])].loc[Cont_all["Cont{}".format(aux_bests[row])]] = FeaturesArm
                Ev_all["Ev{}".format(aux_bests[row])] = Ev_all["Ev{}".format(aux_bests[row])][:-1]
                Ev_all["Ev{}".format(aux_bests[row])] = np.append(Ev_all["Ev{}".format(aux_bests[row])], 1)
                Arms_Chosen[aux_bests[row]] += 1
                Arms_Rewards[aux_bests[row]] += Reward

                ComputeUCB_object.LinUCB_Arms[aux_bests[row]].Update_Reward(Reward, FeaturesArm)

                Times_GameExperienceIsBestArm += 1
                Cumulative_Reward += Reward
                Aligned_Reward.append(Cumulative_Reward/Times_GameExperienceIsBestArm)
                Path_Algorithm.append(aux_bests[row])
                    
    return (Path_Algorithm, Times_GameExperienceIsBestArm, Arms_Rewards, Aligned_Reward, ComputeUCB_object, Arms_Chosen, Ev_all, Features_Arms_Frames)
    
    
def Multiple_Runs(Times, Data, Features, Reward, Alpha, batches):
    Feat_Mult = {}
    Mark_Mult = {}
    Paths_Mult = {}
    Align_Mult = {}
    for x in range(Times):
        Feat_Mult["Feat{0}_Mult".format(x)] = {}
        Mark_Mult["Mark{0}_Mult".format(x)] = {}
        Paths_Mult["Path{0}_Mult".format(x)] = {}
        Align_Mult["Alig{0}_Mult".format(x)] = {}
        
    Times_ArmsChosen_Mult = []
    
    Mean_Reward = []
    Mean_RewardArms = []

    for k in range(Times):
        Paths, Times_Aligned, Arms_Rewards, Aligned_Reward, UCB_Object, Times_ArmsChosen, Mark_dic, Features_dic = LinUCB_Algorithm(Data, Alpha, Features, Reward, batches)
        print(Mark_dic)
                   
        Mark_Mult["Mark{0}_Mult".format(k)] = Mark_dic
        Feat_Mult["Feat{0}_Mult".format(k)] = Features_dic
        Paths_Mult["Path{0}_Mult".format(k)] = Paths
        Align_Mult["Alig{0}_Mult".format(k)] = Aligned_Reward
            
        Mean_Reward.append(np.mean([i / j for i, j in zip(Arms_Rewards, Times_ArmsChosen)]))
        Mean_RewardArms.append([i / j for i, j in zip(Arms_Rewards, Times_ArmsChosen)])
        Times_ArmsChosen_Mult.append(Times_ArmsChosen)
        
    return Paths_Mult, Align_Mult, Mean_Reward, Mean_RewardArms, Mark_Mult, Feat_Mult, Times_ArmsChosen_Mult


