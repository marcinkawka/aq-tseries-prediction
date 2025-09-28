import pandas as pd
from pydantic import BaseModel


class DataLoader(BaseModel):
    @staticmethod
    def load_aq_data(station_code="MzWarAlNiepo") -> pd.DataFrame:
        df = pd.read_csv(f"csv/{station_code}.csv")
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
        df = df.dropna(subset=["dt"]).set_index("dt").sort_index()

        df["ds"] = df.index
        df = df.loc[:, ["ds", "observed_value"]]
        df = df.rename(columns={"observed_value": "y"})
        df["ds"] = df["ds"].dt.tz_convert(None)
        return df

    @staticmethod
    def load_meteo_data() -> None:
        # Placeholder for meteorological data loading logic
        pass
