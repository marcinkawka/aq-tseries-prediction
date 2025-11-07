# import plotly.express as px

# import parsing as ps
from turtle import pd
import matplotlib.pyplot as plt
from prophet import Prophet
from utils import DataLoader, Modeler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

def prophet_error(model : Modeler, df: pd.DataFrame) -> float:
    mean_error = 0
    splits = 5
    tscv = TimeSeriesSplit(n_splits=splits)
    df.dropna(inplace=True)
    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        print(f"Processing fold {i+1}/{splits}")
        train, test = df.iloc[train_index], df.iloc[test_index]

        predictions = model.fit(train, test)
        
        mean_error += mean_absolute_error(test["y"], predictions["yhat"])
        
    mean_error /= splits
    return mean_error


df = DataLoader.load_aq_data(station_code="SlCzestoArmK")
model = Modeler(n_changepoints=4)
mean_error = prophet_error(model, df)
print(f"Mean Absolute Error: {mean_error:.2f}")

# model.fit(df)
# model.plot_fitting()
# plt.show()

# use custom holidays
# evaluate performance

# mean_error = prophet_error(df)

# forecast = model.predict()
# forecast.set_index("ds", inplace=True)
# print(forecast.head())
# df2 = df.set_index("ds", inplace=True)
# # plt.figure(figsize=(10, 6))
# # plt.plot(forecast.index, forecast["yhat"], label="Forecast", color="blue")
# # plt.plot(df.index, df["y"], label="Actual", color="orange")
# # plt.show()
# forecast.to_csv("csv/output.csv", float_format="%.2f", index=True)

# fig2 = model.model.plot_components(fitting_result)
# plt.show()