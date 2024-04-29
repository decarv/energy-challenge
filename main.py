import os
import numpy as np
import pandas as pd
import re
import time
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


# CONST
FORECAST_FILES_PATH = "forecast_files"


def get_dates(file_path: str) -> list[str]:
    """Interpreta o nome do arquivo para obter as datas em que foi feita a previsão e a data da previsão.
    Retorna uma lista com dois elementos, o primeiro é a data da previsão e o segundo é a data prevista.
    """
    date_comp = re.compile(r".*ETA40_p(\d{6})a(\d{6})\.dat")
    match = date_comp.match(file_path)
    if match is None:
        raise ValueError(f"Invalid file name: {file_path}")

    match_1: re.match = match.group(1)
    match_2: re.match = match.group(2)

    forecast_date = f"{match_1[:2]}-{match_1[2:4]}-{match_1[4:]}"
    forecasted_date = f"{match_2[:2]}-{match_2[2:4]}-{match_2[4:]}"
    # forecast_date = datetime.strptime(match.group(1), "%d%m%y").date()
    # forecasted_date = datetime.strptime(match.group(2), "%d%m%y").date()
    return [forecast_date, forecasted_date]


def read_data_file(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as f:
        raw_file = f.readlines()

    list_dados = [line.split() for line in raw_file]
    float_raw_lines = [list(map(float, raw_line)) for raw_line in list_dados]
    return pd.DataFrame(float_raw_lines, columns=['lat', 'long', 'data_value'])


def read_contour_file(file_path: str) -> pd.DataFrame:
    line_split_comp = re.compile(r'\s*,')

    with open(file_path, 'r') as f:
        raw_file = f.readlines()

    l_raw_lines = [line_split_comp.split(raw_file_line.strip()) for raw_file_line in raw_file]
    l_raw_lines = list(filter(lambda item: bool(item[0]), l_raw_lines))
    float_raw_lines = [list(map(float, raw_line))[:2] for raw_line in l_raw_lines]
    header_line = float_raw_lines.pop(0)
    assert len(float_raw_lines) == int(header_line[0])
    return pd.DataFrame(float_raw_lines, columns=['lat', 'long'])


def contour_limits(contour_df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        contour_df['lat'].max(),
        contour_df['lat'].min(),
        contour_df['long'].max(),
        contour_df['long'].min(),
    )


def filter_data(contour_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    max_lat, min_lat, max_long, min_long = contour_limits(contour_df)
    return data_df[(min_lat <= data_df['lat']) & (data_df['lat'] <= max_lat) &
                   (min_long <= data_df['long']) & (data_df['long'] <= max_long)]


def hit(p1, p2, lat, long) -> int:
    safe_div = float("inf")
    if diff := p2['long'] - p1['long']:
        safe_div = diff
    if (
            ((p1['long'] > long) != (p2['long'] > long)) and
            (lat < ((p2['lat'] - p1['lat']) * (long - p1['long']) / safe_div + p1['lat']))
    ):
        return 1
    return 0


def inside_contour(contour_df, lat: float, long: float) -> bool:
    zip_range = zip(range(len(contour_df)-1), range(1, len(contour_df)))
    hits: int = sum((hit(contour_df.iloc[i], contour_df.iloc[j], lat, long) for i, j in zip_range))
    return (hits & 1) == 1


def apply_contour(contour_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:

    contour_df = pd.concat([contour_df, contour_df.iloc[[0]]], ignore_index=True)
    points_inside_contour_df: pd.DataFrame = data_df[
        data_df.apply(
            lambda row: inside_contour(contour_df, row['lat'], row['long']), axis=1
        )
    ]
    return points_inside_contour_df


def hitv(contour_df, data_df):
    lats = data_df['lat'].to_numpy(np.float64).reshape(1, -1)
    longs = data_df['long'].to_numpy(np.float64).reshape(1, -1)

    df_1_lats = contour_df.iloc[:-1, 0].to_numpy(np.float64).reshape(-1, 1)
    df_1_longs = contour_df.iloc[:-1, 1].to_numpy(np.float64).reshape(-1, 1)
    df_2_lats = contour_df.iloc[1:, 0].to_numpy(np.float64).reshape(-1, 1)
    df_2_longs = contour_df.iloc[1:, 1].to_numpy(np.float64).reshape(-1, 1)

    safe_div = np.where(df_2_longs - df_1_longs != 0, df_2_longs - df_1_longs, np.inf)

    cond_1_1 = df_1_longs > longs
    cond_1_2 = df_2_longs > longs
    cond_1 = cond_1_1 != cond_1_2
    op = (longs - df_1_longs) * (df_2_lats - df_1_lats) / safe_div + df_1_lats
    cond_2 = lats < op
    cond = cond_1 & cond_2
    hits = np.sum(cond, axis=0)

    return ((hits & 1) == 1).astype(bool)


def apply_contourv(contour_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = filter_data(contour_df, data_df)
    contour_df = pd.concat([contour_df, contour_df.iloc[[0]]], ignore_index=True)
    mask = hitv(contour_df, data_df)
    masked_data = data_df.iloc[mask, :]
    return masked_data


def generate_solution(acc_prec_df: pd.DataFrame) -> None:
    acc_prec_df.sort_values(by='forecasted_date', inplace=True, ignore_index=True)
    acc_prec_df['data_value_acc'] = acc_prec_df['data_value'].cumsum()
    print(acc_prec_df)

    pallete = sns.color_palette('muted', 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax0 = sns.lineplot(x="forecasted_date", y="data_value_acc", data=acc_prec_df, color=pallete[1])
    sns.barplot(x="forecasted_date", y="data_value", data=acc_prec_df, color=pallete[0])
    l1 = ax0.lines[0]
    x = l1.get_xydata()[:, 0]
    y = l1.get_xydata()[:, 1]
    ax.fill_between(x, y, color=pallete[1], alpha=0.3)

    plt.title("Precipitação Média Prevista por Dia (mês 12/2021)")
    plt.ylabel("Precipitação Média Prevista")
    plt.xlabel("")
    plt.xticks(ticks=acc_prec_df["forecasted_date"],
               labels=[fd.split("-")[0] for fd in acc_prec_df["forecasted_date"]])
    orange_patch = mpatches.Patch(color=pallete[1], label="acumulada")
    blue_patch = mpatches.Patch(color=pallete[0], label="diária")
    ax.legend(handles=[orange_patch, blue_patch])
    plt.grid(axis="y", lw=0.2)

    plt.tight_layout()
    plt.savefig(f"./misc/solution.png")


def main() -> None:

    contour_df: pd.DataFrame = read_contour_file('PSATCMG_CAMARGOS.bln')

    forecast_filepaths: list[str] = [
        os.path.join(os.path.abspath("."), FORECAST_FILES_PATH, filename)
        for filename in os.listdir(FORECAST_FILES_PATH)
    ]

    acc_prec_df: pd.DataFrame = pd.DataFrame(columns=['forecast_date', 'forecasted_date', 'data_value'])

    for fp in forecast_filepaths:
        dates: list[str] = get_dates(fp)

        data_df: pd.DataFrame = read_data_file(fp)
        filtered_data_df: pd.DataFrame = filter_data(contour_df, data_df)

        data_inside_contour_df: pd.DataFrame = apply_contourv(contour_df=contour_df, data_df=filtered_data_df)

        acc_prec_df.loc[len(acc_prec_df)] = [
            dates[0],
            dates[1],
            data_inside_contour_df.iloc[:, 2].mean(),
        ]

    generate_solution(acc_prec_df)


if __name__ == '__main__':
    main()
