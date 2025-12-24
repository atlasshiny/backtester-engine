import pandas as pd

# make sure to generate data set
data_set = pd.read_csv("../data/synthetic.csv")

for event in data_set.itertuples():
    pass # run the data through a method in the Strategy class and if it meets the criteria, execute the strategy