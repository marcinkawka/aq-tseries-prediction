# import plotly.express as px

# import parsing as ps
import matplotlib.pyplot as plt
from prophet import Prophet
from utils import DataLoader, Modeler


# def prophet_error(df):
#     mean_error = 0
#     splits = 5
#     tscv = TimeSeriesSplit(n_splits=splits)

#     for train_index, test_index in tscv.split(df):
#         train, test = df.iloc[train_index], df.iloc[test_index]
#         predictions, _ = modelling(train, test)
#         mean_error += mean_absolute_error(test["y"], predictions["yhat"])

#     mean_error /= splits
#     print(f"The mean error: {mean_error}")
#     return mean_error


df = DataLoader.load_aq_data(station_code="SlCzestoArmK")
model = Modeler( )
model.fit(df)
model.plot_fitting()
plt.show()

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