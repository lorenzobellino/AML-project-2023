# Federated Semantic Segmentation for sefl-driving cars

![](./netdiagram.png)

## Summary
---

This paper proposes a federated semantic segmentation framework for self-driving cars that leverages the power of federated learning to train a deep neural network using segmented datasets obtained from multiple vehicles while preserving data privacy. Traditional methods of semantic segmentation rely on centralized computing, which is impractical in real-world scenarios. The proposed framework includes a central server that coordinates the training process and multiple participating vehicles that provide their segmented data. In addition to the proposed framework, the paper applied domain generalization techniques such as Fourier Domain Adaptation (FDA) to improve the model's generalization and robustness, as well as implemented a pseudo labelling technique to overcome the challenge of unlabelled data from the participating vehicles in a real-world applications. The combination of these techniques with federated learning resulted in a robust and efficient semantic segmentation framework for self-driving cars.

You can read the full report [here](./Report.pdf)

## Setup
---

1) Clone the repository

2) Install the dependencies with poetry

```bash
    poetry install
```
3) Make a new [wandb](https://wandb.ai/) account if you do not have one yet, and create a new wandb project.

4) change the configuration file [here](./src/config/config_options.py) and [here](./src/config/config_transforms.py).

5) run the script
 ```bash
    poetry run python src/run.py -s <STEP> -dts <DATASET> -net <NETWORK>
```
to see other run options, run
```bash
    poetry run python src/run.py -h
```

## Notebooks on colab
---

The same code is available in a notebook format [here](https://github.com/lorenzobellino/Federated-SS-for-self-driving-cars/blob/main/fssfsdc.ipynb) and can be run in colab for free. Since this code is only partially tested due to GPU limitations this could be the better option.
In order to run this notebook:
1) Create a new folder in your google drive
2) Upload the "data" folder from this repository to the folder you created in step 1
3) Make a new [wandb](https://wandb.ai/) account if you do not have one yet, and create a new wandb project.
4) change the configuration for the step you intend to run in the notebook
5) run the SETUP phase in the dotebook and the wanted step
