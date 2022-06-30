# ContextualBandits: Mobile game application
Lin-UCB contextual bandit algorithm is applied to offline mobile game data to optimize revenue and user experience when testing different game variants.

This repository is a collection of [Jupyter](https://jupyter.org/) notebooks and python files. Notebooks show results obtained, and python files contain the algorithm's source code and auxiliary functions to run the notebooks.

## Python files

> - "linUCB.py" file contains the functions needed to run the Lin-UCB algorithm
> - "LIN_UCB_Batches.py" file has the functions required to implement Lin-UCB with live data.
> - "Utils_2.py" file has all the auxiliary functions, such as running the algorithm multiple times, optimizing parameters, or visualizing different types of results. 

Python files are needed to run the notebooks included in the repository.

## Alpha evaluation folder
This folder includes three notebooks that show how the alpha parameter has been evaluated and the consequences it has on the algorithm.
> - "D1_Ad_revenue (2-4).ipynb" notebook shows some results when varying alpha parameter and considering as reward the ad revenue after one day.
> - "D1_IAP (2-4).ipynb" notebook shows some results when varying alpha parameter and considering as reward the in-game purchases after one day.
> - "D1_Return (2-4).ipynb" notebook shows some results when varying alpha parameter and considering as reward the return after one day.

## Results folder
This folder contains 5 notebooks where the metrics considered are analyzed for days 1, 3, and 7. We compute the best alpha via grid-search and show how the algorithm behaves with that alpha. This analysis is done for the two-game variants and four-game variants dataset.

> - "AdRevenue_results.ipynb": Metric related with ad revenue obtained
> - "IAP_results.ipynb": Metric related with in-game purchases revenue obtained
> - "NumSessions_results.ipynb": Metric related with number of sessions
> - "Return_results.ipynb": Metric related with if user has returned to the game or not
> - "TimePlayed_results.ipynb": Metric related with time user has spend playing

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
url={https://github.com/Enricao/TFM-Contextual-Bandits-mobile-game/blob/main/TFM_ContextualBandits_MobileGameApplication.pdf}
author={Enric Azuara Olivera},
  year={2022}
}
```
