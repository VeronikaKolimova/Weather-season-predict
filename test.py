import pandas as pd

df = pd.read_csv('weather.csv')

duplicates = df.duplicated()

num_duplicates = duplicates.sum()
print(f"Количество полных дубликатов: {num_duplicates}")

if num_duplicates > 0:
    print("\nДубликаты:")
    print(df[duplicates])