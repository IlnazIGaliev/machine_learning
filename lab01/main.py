import pandas as pd

def main():
    dataset = pd.read_csv("titanic.csv")
    Jack = dataset[(dataset["Age"] < 20) &
                   (dataset["Age"] > 16) &
                   (dataset["Sex"] == "female") &
                   (dataset.Pclass == 3)
    ]["Survived"].mean()
    print(Jack)


if __name__ == '__main__':
    main()