import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('metrics_sparse_gps_buoys.csv', encoding="utf-8")
df.rename(columns={'model+AF8-name': 'Model', 'epe+AF8-mean': 'EPE', 'flall+AF8-mean': 'Fl-all'}, inplace=True)
df.columns = [col.replace("+AF8-", "_") for col in df.columns]
# df = df.replace("+AF8-", "_", regex=True)
df["Model"] = df["Model"].str.replace(r"\+AF8-", "_", regex=True)
df = df[(df["EPE"] <= 8.5) & (df["Fl-all"] <= 47)]

colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H']

plt.figure(figsize=(10, 6))

for idx, (_, row) in enumerate(df.iterrows()):
   plt.scatter(row["EPE"], row["Fl-all"],
               color=colors[idx],
               marker=markers[idx % len(markers)],
               s=100)
   plt.text(row["EPE"] + 0.05, row["Fl-all"] + 0.2, None, fontsize=9)

plt.xlabel("Endpoint Error (pixels)")
plt.ylabel("Fl-all (%)")
plt.title("Performance of Optical Flow Models on RADARSAT-2 ScanSAR")
plt.grid(True)

handles = [
   plt.Line2D([0], [0], marker=markers[idx % len(markers)], color='w',
              markerfacecolor=colors[idx], markersize=10,
              label=row["Model"])
   for idx, (_, row) in enumerate(df.iterrows())
]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("scatterplot.pdf", format="pdf", bbox_inches="tight")
plt.show()