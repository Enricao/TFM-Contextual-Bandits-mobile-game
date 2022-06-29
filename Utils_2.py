import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

#Check A/B tests for a given dataset
def Check_ABTestId(Data):
    for ID in Data.ab_test_id.value_counts().index:
        print("ID:", ID)
        print("\tTotal: ", Data[Data.ab_test_id == ID].shape[0])
        if len(Data[Data.ab_test_id == ID].NGU.value_counts().values) == 1:
            print("\tNGU: 0")
        else:
            print("\tNGU:", Data[Data.ab_test_id == ID].NGU.value_counts().values[0])
            print("\tNon-NGU:", Data[Data.ab_test_id == ID].NGU.value_counts().values[1])
        
        print("\tNum Arms:", len(np.unique(Data[Data.ab_test_id ==ID].ab_test_experience_id)))
    
    print("\nUnique A/B:", np.unique(Data.ab_test_id))
    
    
#Perform the algorithm for different alpha values and extract the best one
def ExperimentAlphas(Actual_Dataset, Reward, Features, Alphas, Times):
    print("Score to beat:", np.mean(Actual_Dataset[Reward]))

    print("Parameters for following model:")
    print("\tReward:", Reward)
    print("\tNumber of Arms:", np.unique(Actual_Dataset.ab_test_experience_id).shape[0])
    print("\tAlphas: ", Alphas)
    print("\tNum of Features:", len(Features), "\n")
    Results2 = []
    Best = 0
    Paths_Best, Aligns_Best, Mean_Reward_Best, Mean_RewardArms_Best, Mark_Best, Feat_Best, Times_ArmChosen = [], [], [], [], [], [], []
    for A in Alphas:
        print("EVALUATION WITH DIFFERENT ALPHAS FOR {} RUNNING".format(Reward))
        print("Alpha equals:", A)
        Paths_d7p_A5_Big, Aligns_d7p_A5_Big, Mean_Reward_d7p_A5_Big, Mean_RewardArms_d7p_A5_Big, Mark_Mult_d7p_A5_Big, Feat_Mult_d7p_A5_Big, Times_ArmsChosen_d7p_A5_Big = Multiple_Runs(Times = Times, Data = Actual_Dataset, Features = Features, Reward = Reward, Alpha = A)

        print("\tAlgorithm result is:", np.mean(Mean_Reward_d7p_A5_Big))
        print("\tScore to beat is:", np.mean(Actual_Dataset[Reward]))
        Results2.append(np.mean(Mean_Reward_d7p_A5_Big))
        if np.mean(Mean_Reward_d7p_A5_Big) >= Best:
            Paths_Best = Paths_d7p_A5_Big
            Aligns_Best = Aligns_d7p_A5_Big
            Mean_Reward_Best = Mean_Reward_d7p_A5_Big
            Mean_RewardArms_Best = Mean_RewardArms_d7p_A5_Big
            Mark_Best = Mark_Mult_d7p_A5_Big
            Feat_Best = Feat_Mult_d7p_A5_Big
            Times_ArmChosen = Times_ArmsChosen_d7p_A5_Big
            Best = np.mean(Mean_Reward_d7p_A5_Big)
            BestAlpha = A
            
    return Results2, Paths_Best, Aligns_Best, Mean_Reward_Best, Mean_RewardArms_Best, Mark_Best, Feat_Best, Times_ArmChosen     


#One-hot-encoding for given features 
def CreateDummies(Dataset, Features):
    cat_data = Dataset.copy()
    for column in Features:
        if len(np.unique(cat_data[column])) > 2:
            dummie = pd.get_dummies(cat_data[[column]], drop_first = False)
        else:
            dummie = pd.get_dummies(cat_data[[column]], drop_first = True)

        cat_data = cat_data.join(dummie)
        cat_data.drop([column], axis = 1, inplace = True)
    return cat_data


#Plot algorithm arm selection evolution
def Plot_ArmsSelection(Paths, Run, Alpha):
    import seaborn as sns
    sns.set(style='white')
    PathsAux = pd.DataFrame(Paths["Path{}_Mult".format(Run)], columns = ["X"])
    for i in range(len(np.unique(PathsAux))):
        PathsAux['Arm{}'.format(i)] = (PathsAux["X"] == i).cumsum()
    
    df = PathsAux.loc[:, PathsAux.columns != "X"]
    df.plot()
    plt.ylabel("Times picked as the best arm", fontweight='bold')
    plt.title("Arms selection evolution (Alpha = {})".format(Alpha), fontweight='bold')
    plt.xlabel('Nº cases evaluated by the algorithm', fontweight='bold')
    
    
#Function defined to compute multiple runs of the algorithm and return all the indicator to analyze it    
def Multiple_Runs(Times, Data, Features, Reward, Alpha):
    Feat_Mult = {}
    Mark_Mult = {}
    Paths_Mult = {}
    Align_Mult = {}
    Features_Arms_Mult = {}
    for x in range(Times):
        Feat_Mult["Feat{0}_Mult".format(x)] = {}
        Mark_Mult["Mark{0}_Mult".format(x)] = {}
        Paths_Mult["Path{0}_Mult".format(x)] = {}
        Align_Mult["Alig{0}_Mult".format(x)] = {}
        Features_Arms_Mult["Feat{0}_Mult".format(x)] = {}
        
    Times_ArmsChosen_Mult = []
    
    Mean_Reward_Align = []
    Mean_RewardArms = []

    for k in range(Times):
        Paths, Times_Aligned, Arms_Rewards, Aligned_Reward, UCB_Object, Times_ArmsChosen, Mark_dic, Features_dic = LinUCB_Algorithm(Data, Alpha, Features, Reward)
                   
        Mark_Mult["Mark{0}_Mult".format(k)] = Mark_dic
        Feat_Mult["Feat{0}_Mult".format(k)] = Features_dic
        Paths_Mult["Path{0}_Mult".format(k)] = Paths
        Align_Mult["Alig{0}_Mult".format(k)] = Aligned_Reward
            
        Mean_Reward_Align.append(np.mean(Aligned_Reward))
        Mean_RewardArms.append([i / j for i, j in zip(Arms_Rewards, Times_ArmsChosen)])
        Times_ArmsChosen_Mult.append(Times_ArmsChosen)
        
    return Paths_Mult, Align_Mult, Mean_Reward_Align, Mean_RewardArms, Mark_Mult, Feat_Mult, Times_ArmsChosen_Mult


#Function to extract the average of a given feature for all the given arms
def Extract_Mean_Feature(Mark_Mult, Feat_Mult, Feature, Print = True):
    Feature_aux = {}
    Store = []
    for x in range(len(Mark_Mult["Mark0_Mult"].keys())):
        Feature_aux["Arm{0}".format(x)] = []
        
    for l in range(len(Feat_Mult.keys())):
        for x in range(len(Feat_Mult["Feat{}_Mult".format(l)].keys())):
            if (Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)].shape[0] != 0):
                Feature_aux["Arm{}".format(x)].append(Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Feature].value_counts(normalize = True, sort = True).sort_index().values[0])
            else:
                Feature_aux["Arm{}".format(x)].append(np.mean(Feature_aux["Arm{}".format(x)]))
            
    for l in range(len(Feature_aux.keys())):
        if Print == True:
            print("Proportion", Feature, "equals 0 for arm", l, "is:", 100*np.mean(Feature_aux["Arm{}".format(l)]))
            
        Store.append(100*np.mean(Feature_aux["Arm{}".format(l)]))
        
    return Store


#Function to extract the distribution of country. Used on several plot functions
def Extract_Country(Mark_Mult, Feat_Mult, Print = True):
    Feature_aux = {}
    Store = []
    Country_aux = ['country_alias_country_group1', 'country_alias_country_group2', 'country_alias_country_group3']
    for x in range(len(Mark_Mult["Mark0_Mult"].keys())):
        Feature_aux["Arm{0}".format(x)] = []

    for l in range(len(Feat_Mult.keys())):
        for x in range(len(Feat_Mult["Feat{}_Mult".format(l)].keys())):
            if len(Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = True).values) == 2:
                for p in range(len(Country_aux)):
                            if ("country_alias_country_group3" in Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = False).sort_index().index) == False:
                                a = Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = False).sort_index().values.tolist()
                                a.insert(2, 0)
                                Feature_aux["Arm{}".format(x)].append(a)
                        
                            if ("country_alias_country_group2" in Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = False).sort_index().index) == False:
                                a = Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = False).sort_index().values.tolist()
                                a.insert(1, 0)
                                Feature_aux["Arm{}".format(x)].append(a)
                        
                            if ("country_alias_country_group1" in Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = False).sort_index().index) == False:
                                a = Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = False).sort_index().values.tolist()
                                a.insert(0, 0)
                                Feature_aux["Arm{}".format(x)].append(a)  
            else:
                Feature_aux["Arm{0}".format(x)].append(Feat_Mult["Feat{}_Mult".format(l)]["Features_Arm{}".format(x)][Country_aux].idxmax(axis=1).value_counts(normalize = True, sort = False).sort_index().values)
        
    for k in range(len(Feature_aux.keys())):
        if Print == True:
            print("Proportion country for arm", k, "is:" , np.mean(Feature_aux["Arm{0}".format(k)], axis = 0))
        Store.append(np.mean(Feature_aux["Arm{0}".format(k)], axis = 0))
    
    return Store


#Function to plot a given feature mean from multiple runs
def Plot1(Feature, Mark_Mult, Feat_Mult):
    import seaborn as sns
    aux = Extract_Mean_Feature(Mark_Mult, Feat_Mult, Feature)
    aux2 = [100 - x for x in aux]
    x = []
    for l in range(len(aux)):
        x.append("Arm {}".format(l))
  
    plt.bar(x, aux, color='r', label = "Type 0")
    plt.bar(x, aux2, bottom=aux, color='b', label = "Type 1")
    plt.ylabel("Percentage", fontweight='bold')
    plt.legend()
    plt.title("{}: users for each arm".format(Feature), fontweight='bold')
    plt.show()

    
#Function to plot contry feature from 2 runs    
def PlotFeaturesArm_2_Country(Feat_Mult):
    import seaborn as sns
    datafram = pd.DataFrame({"Run": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                         "Arm": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                         "Type Country": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 
                         })

    Percentage = []
    for Run in range(len(Feat_Mult.keys())):
        for Arm in range(len(Feat_Mult["Feat{}_Mult".format(Run)])):
            for Feature in range(1, 4):
                Percentage = np.append(Percentage, np.mean(Feat_Mult["Feat{}_Mult".format(Run)]["Features_Arm{}".format(Arm)]["country_alias_country_group{}".format(Feature)]))

    datafram["Percentage"] = Percentage
    sns.catplot(kind='bar', data= datafram, col='Run', x='Arm', y='Percentage', hue='Type Country')      

    
#Function to plot contry feature from 1 runs    
def PlotFeaturesArm_2_Country_aux(Feat_Mult):
    import seaborn as sns
    datafram = pd.DataFrame({
                         "Arm": [0, 0, 0, 1, 1, 1],
                         "Type Country": [1, 2, 3, 1, 2, 3], 
                         })

    Feat_Mult = Feat_Best1
    Percentage = []
    for Arm in range(len(Feat_Mult["Feat0_Mult"])):
        for Feature in range(1, 4):
            Percentage = np.append(Percentage, np.mean(Feat_Mult["Feat0_Mult"]["Features_Arm{}".format(Arm)]["country_alias_country_group{}".format(Feature)]))

    datafram["Percentage"] = Percentage
    sns.catplot(kind='bar', data= datafram, x='Arm', y='Percentage', hue='Type Country')   
    
    
#Function to plot country distribution of 2 arms dataset from multiple runs    
def Plot2_2arms(Mark_Mult, Feat_Mult):
    import seaborn as sns
    aux = Extract_Country(Mark_Mult, Feat_Mult)
    aux = np.matrix(aux).T
    aux = aux.tolist()
    x = []
    for l in range(np.array(aux).shape[1]):
        x.append("Arm {}".format(l))
    plt.bar(x, np.array(aux[0]), color='#7f6d5f', label = "Type 1")
    plt.bar(x, np.array(aux[1]), bottom = np.array(aux[0]), color = "#557f2d", label = "Type 2")
    plt.bar(x, np.array(aux[2]), bottom = np.array(aux[0]) + np.array(aux[1]), color = "#2d7f5e", label = "Type 3")
    plt.ylabel("Percentage", fontweight='bold')
    plt.legend()
    plt.title("Users country for each arm", fontweight='bold')
    plt.show()
    
    
#Function to plot performance of the algorithm with 2 arms    
def PerformanceArm2(Mean_Rewards, Alpha, Reward):
    import seaborn as sns
    barWidth = 0.25    
    bars1 = list(list(zip(*Mean_Rewards))[0])
    bars2 = list(list(zip(*Mean_Rewards))[1])
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
 
    # Make the plot
    plt.bar(r1, bars1, color = '#7f6d5f', width = barWidth, edgecolor = 'white', label = 'Arm 0')
    plt.bar(r2, bars2, color = '#557f2d', width = barWidth, edgecolor = 'white', label = 'Arm 1')
        
    plt.ylabel('{}'.format(Reward), fontweight='bold')
    plt.title("{} for each arm (Alpha = {})".format(Reward, Alpha), fontweight='bold')
    plt.xlabel('Execution number', fontweight='bold')
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], range(1, len(list(list(zip(*Mean_Rewards))[0]))+1))
    plt.legend()
    plt.show()

    
#Function to plot a given feature multiple runs distribution 2 arms    
def PlotFeaturesArm_2(Feat_Mult, Alpha, Reward):
    import seaborn as sns
    barWidth = 0.25
    bars1 = []
    bars2 = []
    for i in range(len(Feat_Mult.keys())):
        for j in range(len(Feat_Mult["Feat0_Mult"].keys())):
            if j == 0:
                bars1 = np.append(bars1, np.mean(Feat_Mult["Feat{}_Mult".format(i)]["Features_Arm{}".format(j)][Reward]))
            if j == 1:
                bars2 = np.append(bars2, np.mean(Feat_Mult["Feat{}_Mult".format(i)]["Features_Arm{}".format(j)][Reward]))

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
 
    # Make the plot
    plt.bar(r1, bars1, color = '#7f6d5f', width = barWidth, edgecolor = 'white', label = 'Arm 0')
    plt.bar(r2, bars2, color = '#557f2d', width = barWidth, edgecolor = 'white', label = 'Arm 1')
        
    plt.ylabel("Percentage", fontweight='bold')
    plt.title("Percentage of {} = 0".format(Reward), fontweight='bold')
    plt.xlabel('Execution number', fontweight='bold')
    plt.xticks([r + barWidth/2 for r in range(len(bars1))], range(len(Feat_Mult.keys())))
    plt.legend()
    plt.show() 
              
        
#Function to plot a given reward multiple runs distribution 4 arms        
def PerformanceArm4(Mean_Rewards, Alpha, Reward):
    import seaborn as sns
    barWidth = 0.125  
    bars1 = list(list(zip(*Mean_Rewards))[0])
    bars2 = list(list(zip(*Mean_Rewards))[1])
    bars3 = list(list(zip(*Mean_Rewards))[2])
    bars4 = list(list(zip(*Mean_Rewards))[3])
    
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
 
    # Make the plot
    plt.bar(r1, bars1, color = '#7f6d5f', width = barWidth, edgecolor = 'white', label = 'Arm 0')
    plt.bar(r2, bars2, color = '#557f2d', width = barWidth, edgecolor = 'white', label = 'Arm 1')
    plt.bar(r3, bars3, color = '#b0ad4c', width = barWidth, edgecolor = 'white', label = 'Arm 2')
    plt.bar(r4, bars4, color = '#a6633f', width = barWidth, edgecolor = 'white', label = 'Arm 3')
        
    plt.ylabel("{}".format(Reward), fontweight='bold')
    plt.title("{} for each run and arm (Alpha = {})".format(Reward, Alpha), fontweight='bold')
    plt.xlabel('Execution number', fontweight='bold')
    plt.xticks([r + barWidth + 0.05 for r in range(len(bars1))], range(1, len(list(list(zip(*Mean_Rewards))[0]))+1))
    plt.legend()
    plt.show()
    
    
#Function to plot a given feature multiple runs distribution 4 arms    
def PlotFeaturesArm_4(Feat_Mult, Alpha, Reward):
    import seaborn as sns
    bars1 = []
    bars2 = []
    bars3 = []
    bars4 = []
    barWidth = 0.15
    for i in range(len(Feat_Mult.keys())):
        for j in range(len(Feat_Mult["Feat0_Mult"].keys())):
            if j == 0:
                bars1 = np.append(bars1, np.mean(Feat_Mult["Feat{}_Mult".format(i)]["Features_Arm{}".format(j)][Reward]))
            if j == 1:
                bars2 = np.append(bars2, np.mean(Feat_Mult["Feat{}_Mult".format(i)]["Features_Arm{}".format(j)][Reward]))
            if j == 2:
                bars3 = np.append(bars3, np.mean(Feat_Mult["Feat{}_Mult".format(i)]["Features_Arm{}".format(j)][Reward]))
            if j == 3:
                bars4 = np.append(bars4, np.mean(Feat_Mult["Feat{}_Mult".format(i)]["Features_Arm{}".format(j)][Reward]))
            

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    # Make the plot
    plt.bar(r1, bars1, color = '#7f6d5f', width = barWidth, edgecolor = 'white', label = 'Arm 0')
    plt.bar(r2, bars2, color = '#557f2d', width = barWidth, edgecolor = 'white', label = 'Arm 1')
    plt.bar(r3, bars3, color = '#b0ad4c', width = barWidth, edgecolor = 'white', label = 'Arm 2')
    plt.bar(r4, bars4, color = '#a6633f', width = barWidth, edgecolor = 'white', label = 'Arm 3')
        
    plt.ylabel("Percentage", fontweight='bold')
    plt.title("Percentage of {} = 0".format(Reward), fontweight='bold')
    plt.xlabel('Execution number', fontweight='bold')
    plt.xticks([r + barWidth + 0.1 for r in range(len(bars1))], range(len(Feat_Mult.keys())))
    plt.legend()
    plt.show()

    
#Function to plot country when 5 runs computed    
def Country_5Runs(Feat_Mult):
    import seaborn as sns
    datafram = pd.DataFrame({"Run": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                                 5, 5, 5, 5, 5, 5],
                         "Arm": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                0, 0, 0, 1, 1, 1],
                         "Type Country": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                                         1, 2, 3, 1, 2, 3], 
                         })

    Percentage = []
    for Run in range(len(Feat_Mult.keys())):
        for Arm in range(len(Feat_Mult["Feat{}_Mult".format(Run)])):
            for Feature in range(1, 4):
                Percentage = np.append(Percentage, np.mean(Feat_Mult["Feat{}_Mult".format(Run)]["Features_Arm{}".format(Arm)]["country_alias_country_group{}".format(Feature)]))

    datafram["Percentage"] = Percentage
    sns.catplot(kind='bar', data= datafram, col='Run', x='Arm', y='Percentage', hue='Type Country')  
    
    
#Function to plot country when 8 runs computed    
def Country_8Runs(Feat_Mult):
    import seaborn as sns
    datafram = pd.DataFrame({"Run": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,
                                 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
                         "Arm": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
                                0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                         "Type Country": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                                         1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 
                         })

    Percentage = []
    for Run in range(len(Feat_Mult.keys())):
        for Arm in range(len(Feat_Mult["Feat{}_Mult".format(Run)])):
            for Feature in range(1, 4):
                Percentage = np.append(Percentage, np.mean(Feat_Mult["Feat{}_Mult".format(Run)]["Features_Arm{}".format(Arm)]["country_alias_country_group{}".format(Feature)]))

    datafram["Percentage"] = Percentage
    sns.catplot(kind='bar', data= datafram, col='Run', x='Arm', y='Percentage', hue='Type Country')  
    
    
#Function to plot country when 5 runs computed (4-arm data)    
def PlotFeaturesArm_4_Country5Runs(Feat_Mult):
    import seaborn as sns
    datafram = pd.DataFrame({"Run": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                             "Arm": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,],
                             "Type Country": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                                             1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                                             1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3] 
                            })
    Percentage = []
    for Run in range(len(Feat_Mult.keys())):
        for Arm in range(len(Feat_Mult["Feat{}_Mult".format(Run)])):
            for Feature in range(1, 4):
                Percentage = np.append(Percentage, np.mean(Feat_Mult["Feat{}_Mult".format(Run)]["Features_Arm{}".format(Arm)]["country_alias_country_group{}".format(Feature)]))
                
    datafram["Percentage"] = Percentage
    sns.catplot(kind='bar', data= datafram, col='Run', x='Arm', y='Percentage', hue='Type Country')    
    
    
#Function to plot country when 4 runs computed (4-arm data)    
def PlotFeaturesArm_4_Country4Runs(Feat_Mult):
    import seaborn as sns
    datafram = pd.DataFrame({"Run": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
                             "Arm": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                    0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                             "Type Country": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                                             1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3] 
                            })
    Percentage = []
    for Run in range(len(Feat_Mult.keys())):
        for Arm in range(len(Feat_Mult["Feat{}_Mult".format(Run)])):
            for Feature in range(1, 4):
                Percentage = np.append(Percentage, np.mean(Feat_Mult["Feat{}_Mult".format(Run)]["Features_Arm{}".format(Arm)]["country_alias_country_group{}".format(Feature)]))
                
    datafram["Percentage"] = Percentage
    sns.catplot(kind='bar', data= datafram, col='Run', x='Arm', y='Percentage', hue='Type Country') 
    
    
#Function to plot country when 2 runs computed (4-arm data)      
def PlotFeaturesArm_4_Country(Feat_Mult):
    import seaborn as sns
    datafram = pd.DataFrame({"Run": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                             "Arm": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,],
                             "Type Country": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 
                            })
    Percentage = []
    for Run in range(len(Feat_Mult.keys())):
        for Arm in range(len(Feat_Mult["Feat{}_Mult".format(Run)])):
            for Feature in range(1, 4):
                Percentage = np.append(Percentage, np.mean(Feat_Mult["Feat{}_Mult".format(Run)]["Features_Arm{}".format(Arm)]["country_alias_country_group{}".format(Feature)]))
                
    datafram["Percentage"] = Percentage
    sns.catplot(kind='bar', data= datafram, col='Run', x='Arm', y='Percentage', hue='Type Country')
    
    
#Function to plot cumulative reward    
def Plot_AlignRewards(Align, Run, Reward, Alpha, N):
    import seaborn as sns
    sns.set(style='white')
    #print(np.mean(Align["Alig{}_Mult".format(Run)]))
    plt.plot(range(len(Align["Alig{}_Mult".format(Run)][0:N])),
         Align["Alig{}_Mult".format(Run)][0:N])
    plt.ylabel('{}'.format(Reward), fontweight='bold')
    plt.title("Algorithm evolution (Alpha = {})".format(Alpha), fontweight='bold')
    plt.xlabel('Nº cases evaluated by the algorithm', fontweight='bold')

    
#Function to plot mean feature of day 3    
def printFeatures(a, b):
    for i in list(["return_d1", "ad_revenue_d1", "time_played_d1","num_impressions_d1", "iap_revenue_d1", "num_transactions_d1"]):
        Extract_Mean_Feature(a, b, i)
        print("")

        
#Function to plot mean feature of given fetaures        
def printFeatures2(a, b, Features):
    for i in Features:
        Extract_Mean_Feature(a, b, i)
        print("")

class LinUCB_Arm():
    
    def __init__(self, Arm_index, Dimension, Alpha):
        self.Arm_index = Arm_index
        self.Alpha = Alpha
        self.A = np.identity(Dimension)
        self.b = np.zeros([Dimension, 1])

        
#Lin UCB class, can be found as well with commentaris on its individual .py file.        
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

