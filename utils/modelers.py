from prophet import Prophet
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd


class Modeler(BaseModel):
    model: Prophet =None
    n_changepoints: int = 12*6
    changepoint_prior_scale: float=0.05 # Try: 0.001, 0.01, 0.05, 0.1
    seasonality_prior_scale: float=10
    interval_width: float=0.95
    fourier_order: int=10
    fitting_result: pd.DataFrame = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def model_post_init(self, __context):
        """Initialize Prophet after model validation"""
        if self.model is None:
            self.model = Prophet(n_changepoints=self.n_changepoints,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                interval_width=self.interval_width)
            self.model.add_country_holidays(country_name='PL')

    def fit(self, train, test=None) -> None:
        if hasattr(self.model, 'params'):
            self.model = None
            self.model_post_init(None)

        if test is None:
            test = train
        self.model.fit(train)
        self.fitting_result = self.model.predict(test)
        return self.fitting_result

    def plot_fitting(self) -> None:
        fig = self.model.plot_components(self.fitting_result)
        return fig
    
    def predict(self, periods=240, freq="h") -> pd.DataFrame:
        time_frame = self.model.make_future_dataframe(
            freq=freq, periods=periods, include_history=False
        )

        forecast = self.model.predict(time_frame)
        return forecast
