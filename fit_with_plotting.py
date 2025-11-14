# import plotly.express as px

# import parsing as ps
import matplotlib.pyplot as plt
from utils import DataLoader, Modeler
import numpy as np


df = DataLoader.load_aq_data(station_code="SlCzestoArmK")
model = Modeler(n_changepoints=24)

df_res = model.fit(df)
# model.plot_components()
# plt.show()

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(df["ds"], df["y"], s=12, color="C0", label="Observed")
ax.plot(df_res["ds"], np.exp(df_res["yhat"]), color="C1", label="Forecast")
ax.fill_between(
    df_res["ds"],
    np.exp(df_res["yhat_lower"]),
    np.exp(df_res["yhat_upper"]),
    color="C1",
    alpha=0.25,
    label="Forecast interval",
)

ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()
ax.set_title("Observed vs Prophet Forecast with Uncertainty")
plt.tight_layout()
plt.show()
