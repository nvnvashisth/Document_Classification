import pandas as pd

df = pd.read_csv('data.csv')
dl_size = df['download_size']

dl_size = [x[:-3] for x in dl_size]
dl_size = [float(x) for x in dl_size]

print(sum(dl_size))