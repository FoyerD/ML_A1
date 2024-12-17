import pandas as pd

file = pd.read_csv("alcohol.csv")
file = file.drop("Units (4 Decimal Places)", axis=1)
file = file.drop("Units per 100ml", axis=1)
file.to_csv("alcohol_clean.csv", index=False)
