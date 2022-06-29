import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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

def LinUCB_Algorithm(Data, Alpha, Features, Reward_Feature):
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
    
    for row in tqdm(range(Data.shape[0])):
        #We track the A/B test experience variant for the user
        Which_Arm = Data.ab_test_experience_id.iloc[row]
        
        #We select user features
        FeaturesArm = Data[Features].iloc[row,].values
        
        #We compute which is the best arm by the Lin-UCB algo
        Best_Arm = ComputeUCB_object.Select_Arm(FeaturesArm)
        
        #If the best arm is the same as the A/B test experience allocated to the user, we observe the reward and update everything
        if Best_Arm == Which_Arm:
            Reward = Data[Reward_Feature].iloc[row]
            
            for x in range(K_arms):
                Ev_all["Ev{0}".format(x)] = np.append(Ev_all["Ev{0}".format(x)], 0)
            
            Cont_all["Cont{}".format(Best_Arm)] = Cont_all["Cont{}".format(Best_Arm)] + 1
            Features_Arms_Frames["Features_Arm{}".format(Best_Arm)].loc[Cont_all["Cont{}".format(Best_Arm)]] = FeaturesArm
            Ev_all["Ev{}".format(Best_Arm)] = Ev_all["Ev{}".format(Best_Arm)][:-1]
            Ev_all["Ev{}".format(Best_Arm)] = np.append(Ev_all["Ev{}".format(Best_Arm)], 1)
            Arms_Chosen[Best_Arm] += 1
            Arms_Rewards[Best_Arm] += Reward

            ComputeUCB_object.LinUCB_Arms[Best_Arm].Update_Reward(Reward, FeaturesArm)

            Times_GameExperienceIsBestArm += 1
            Cumulative_Reward += Reward
            Aligned_Reward.append(Cumulative_Reward/Times_GameExperienceIsBestArm)
            Path_Algorithm.append(Best_Arm)
                    
    return (Path_Algorithm, Times_GameExperienceIsBestArm, Arms_Rewards, Aligned_Reward, ComputeUCB_object, Arms_Chosen, Ev_all, Features_Arms_Frames)

