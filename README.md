# ContextualBandits: Mobile game application
Lin-UCB contextual bandit algorithm is applied to offline mobile game data to optimize revenue and user experience when testing different game variants.

This repository is a collection of [Jupyter](https://jupyter.org/) notebooks and python files. Notebooks show results obtained and python files contain source code of the algorithm and auxiliar functions to run the notebooks.

## Python files

> - "linUCB.py" file contains the functions needed to run the Lin-UCB algorithm
> - "LIN_UCB_Batches.py" file contains the functions needed to implement Lin-UCB with live data.
> - "Utils_2.py" file contains all the auxiliar functions, such as running the algorithm multiple times, optimizing parameters, or visualizing different types of results. 

Python files are needed to run the notebooks included in the repository.

## Alpha evaluation folder
This folder includes three notebooks that show how the alpha parameter has been evaluated.
> - "D1_Ad_revenue (2-4).ipynb" notebook shows some results when varying alpha parameter and considering as reward the ad revenue after one day.
> - "D1_IAP (2-4).ipynb" notebook shows some results when varying alpha parameter and considering as reward the in-game purchases after one day.
> - "D1_Return (2-4).ipynb" notebook shows some results when varying alpha parameter and considering as reward the return after one day.

## Results folder
This folder contains 5 notebooks where each of the metrics considered are analyzed for different alphas and number of game variants.

> - 

## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* Mail: Enricazuaraolivera@gmail.com

### BibTex reference format for citation for the Code
```
@misc{CB_MGA,
title={Contextual Bandits: Mobile game application},
url={https://github.com/Enricao/TFM-Contextual-Bandits-mobile-game}
author={Enric Azuara Olivera},
  year={2022}
}
```
### BibTex reference format for citation for the report of the Master's Thesis

```
@misc{CB_MGA,
title={Contextual Bandits: Mobile game application},
url={https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/ABrando-MDN-MasterThesis.pdf}
author={Enric Azuara Olivera},
  year={2022}
}
```
