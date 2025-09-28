from prophet import Prophet
from pydantic import BaseModel, Field
import pandas as pd


class Modeler(BaseModel):
    model: Prophet = Field(default_factory=Prophet)

    class Config:
        arbitrary_types_allowed = True

    def fit(self, train, test=None) -> pd.DataFrame:
        if test is None:
            test = train
        self.model.fit(train)
        predictions = self.model.predict(test)
        return predictions

    def predict(self, periods=240, freq="h") -> pd.DataFrame:
        time_frame = self.model.make_future_dataframe(
            freq=freq, periods=periods, include_history=False
        )

        forecast = self.model.predict(time_frame)
        return forecast
