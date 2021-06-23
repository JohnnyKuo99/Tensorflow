import pandas as pd

df = pd.DataFrame({'Location': {0: 'Taipei', 1: 'Hsinchu', 2: "Taichung"},
                   'Java': {0: 5, 1: 10, 2: 15},
                   'Python': {0: 2, 1: 4, 2: 6},
                   'C#': {0: 4, 1: 3, 2: 2}})
print(df)
print(pd.melt(df, id_vars=['Location'], value_vars=['Java']))
print(pd.melt(df, id_vars=['Location'], value_vars=['Java', 'Python']))
print(pd.melt(df, id_vars=['Location']))