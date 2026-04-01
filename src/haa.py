import datetime
import warnings

from ffn import to_price_index
from pandas import concat, set_option
from yfinance import download

warnings.filterwarnings("ignore")


def _normalize_index(df):
    df = df.copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def predict_haa(end_date: datetime.date = datetime.date.today()):
    """
    Returns the investment signal for the Hybrid Asset Allocation strategy.

    Args:
        end_date (datetime.date, optional): The end date for the prediction.
            Defaults to today's date.

    Returns:
        dict: A dictionary containing the prediction signal.
    """
    name = "Default Haa"

    # Inputs
    offensives = ["SPY", "IWM", "VEA", "VWO", "DBC", "VNQ", "IEF", "TLT"]
    defensives = ["IEF", "BIL"]
    protectives = ["TIP"]
    number_of_assets = 4
    all_stocks = offensives + defensives + protectives

    delta = datetime.timedelta(days=500)
    start_date = end_date - delta

    # Download the daily stock data
    data = download(all_stocks, start=start_date, end=end_date)["Close"]
    data = _normalize_index(data)

    # Download today's intraday data, 1 minute interval. Keep only the last data point.
    todays_data = download(all_stocks, period="1d", interval="1m")["Close"].iloc[-1].to_frame().T
    todays_data = _normalize_index(todays_data)

    # If intraday data is newer than daily data, add today's intraday data to the dataset.
    # This allows taking today's data into account when calculating the momentum score.
    if end_date == datetime.date.today() and data.index[-1].date() != todays_data.index[-1].date():
        print(data.index[-1].date(), todays_data.index[-1].date())
        print("Adding today's intra day data to the dataset")

        # Set the index to the next day
        todays_data.index = [data.index[-1] + datetime.timedelta(days=1)]
        data = concat([data, todays_data], axis=0)

    # Normalize after concat in case any index metadata changed
    data = _normalize_index(data)

    # Remove duplicate index entries before downstream operations
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    yahoo_data = data.pct_change().dropna()

    # Overwrite yahoo data with the stocks' price index without per-column assignment
    # to avoid duplicate-label reindexing issues in newer pandas versions.
    indexed_columns = {}
    for column in yahoo_data.columns:
        series = yahoo_data[column].dropna()
        indexed_columns[column] = to_price_index(series)

    yahoo_data = concat(indexed_columns, axis=1)
    yahoo_data = yahoo_data.sort_index()

    data_monthly = yahoo_data.resample("BME").last()

    # Replace last index date (which will be last business day of the month)
    # with today's date for clarity.
    data_monthly.rename(
        index={
            data_monthly.index[-1]: datetime.datetime(
                end_date.year, end_date.month, end_date.day
            )
        },
        inplace=True,
    )
    data_monthly.index = data_monthly.index.strftime("%B %d, %Y")

    # Calculate momentum score
    score = sum(
        [(data_monthly / data_monthly.shift(m) - 1) / 4 for m in [1, 3, 6, 12]]
    ).dropna()

    set_option("display.max_columns", None)

    print("\nMomentum scores:")
    print(score[-5:])

    # Calculate if there's absolute momentum
    # (If protectives, TIP, have positive momentum score)
    absolute_momentum = score.apply(
        lambda x: min(x[protectives]) > 0, axis=1
    ).to_frame("absolute_momentum")

    def pick_4_best_assets(x):
        if x["absolute_momentum"]:
            of_momentums = x[offensives].astype(float).sort_values(ascending=False).iloc[
                :number_of_assets
            ]
            def_momentum = x[defensives].astype(float).sort_values(ascending=False).iloc[
                :1
            ]
            pos_of_momentums = of_momentums.loc[lambda values: values > 0]

            a = len(pos_of_momentums)
            while a < number_of_assets:
                a += 1
                pos_of_momentums = concat([pos_of_momentums, def_momentum], axis=0)

            return pos_of_momentums.index.to_list()
        else:
            return x[defensives].astype(float).idxmax()

    prediction = score.join(absolute_momentum).apply(pick_4_best_assets, axis=1).dropna()

    signal = {
        "name": name,
        "period": end_date.strftime("%B %Y"),
        "signal": prediction[-2:].to_list()[0],
    }

    print("\nPredictions for month after:")
    print(prediction[:])

    return signal


prediction = predict_haa()

print(
    "\nThe allocation for {} is {}. The Hybrid Asset Allocation".format(
        prediction["period"], prediction["signal"]
    )
)
print("strategy dictates to (re)allocate on the first trading day of the month.\n")

print("The last prediction in the table is based on incomplete data for that month (unless it's the")
print("last day of the month). Therefore it should only be used as an indicator for what the next")
print("month's allocation might become.")

exit(0)
