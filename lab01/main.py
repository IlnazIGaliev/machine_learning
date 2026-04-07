import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    dataset = pd.read_csv("titanic.csv")

    # x = np.linspace(0, 14, 200)
    # y = x ** 2 + np.random.normal(0, 5, 200)
    #
    # plt.scatter(x, y)
    # plt.show()
    dataset = pd.read_csv("titanic.csv")
    Rose = dataset[(dataset["Age"] < 20) &
                   (dataset["Age"] > 16) &
                   (dataset["Sex"] == "female") &
                   (dataset.Pclass == 1)
    ]["Survived"].mean()
    print(Rose)


if __name__ == '__main__':
    main()