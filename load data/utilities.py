# SPDX-FileCopyrightText: 2022 Aleksander Grochowicz
#
# SPDX-License-Identifier: GPL-3.0-or-later

# This script contains functions that are used for `load_regression` and `create_artifical_demand` to generate load data for 1980--2020.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.deterministic import DeterministicProcess, Seasonality
import pandas as pd
from scipy.stats import pearsonr
from math import sqrt
import calendar
from datetime import date
import holidays
import tabulate


def compute_correlation(data, data_comparison, place, column=None, frequency=["1D"]):
    """Compute correlations between the demand and the temperature in the given dataset.

    Keyword arguments:
    data -- dataset as dict (for demand, e.g. artificial)
    data_comparison -- dataset given as real demand, often the same dataset as data
    place -- list of keys of data
    column -- if demand is given as one of the columns (default None)
    frequency -- frequencies for resampled data to check correlation on (default '1D')
    """

    correlation = {}
    for i in place:
        correlation[i] = pd.DataFrame(index=["corr"], columns=frequency)
        for f in frequency:
            if column == None:
                correlation[i][f] = round(
                    pearsonr(
                        data[i].resample(f).mean(),
                        data_comparison[i]["demand"]
                        .fillna(method="bfill")
                        .resample(f)
                        .mean(),
                    )[0],
                    2,
                )
            else:
                correlation[i][f] = round(
                    pearsonr(
                        data[i][column].resample(f).mean(),
                        data_comparison[i]["demand"]
                        .fillna(method="bfill")
                        .resample(f)
                        .mean(),
                    )[0],
                    2,
                )
    corr = pd.concat(correlation)
    corr.droplevel(level=1)
    return corr


def compute_rmse(data, data_comparison, place, column=None, frequency=["1D"]):
    """Compute RMSE between the demand and the temperature in the given dataset.

    Keyword arguments:
    data -- dataset as dict (for demand, e.g. artificial)
    data_comparison -- dataset given as real demand, often the same dataset as data
    place -- list of keys of data
    column -- if demand is given as one of the columns (default None)
    frequency -- frequencies for resampled data to check correlation on (default '1D')
    """
    rse = {}
    rmse = {}
    for i in place:
        rse[i] = pd.DataFrame(index=["RSE"], columns=frequency)
        rmse[i] = pd.DataFrame(index=["RMSE"], columns=frequency)
        for f in frequency:
            if column == None:
                rse[i][f] = (
                    (
                        data[i].resample(f).mean()
                        - data_comparison[i]["demand"]
                        .fillna(method="bfill")
                        .resample(f)
                        .mean()
                    )
                    ** 2
                ).mean()
                rmse[i][f] = round(sqrt((rse[i][f].mean())), 1)
            else:
                rse[i][f] = (
                    (
                        data[i][column].resample(f).mean()
                        - data_comparison[i]["demand"]
                        .fillna(method="bfill")
                        .resample(f)
                        .mean()
                    )
                    ** 2
                ).mean()
                rmse[i][f] = round(sqrt((rse[i][f].mean())), 1)
    RMSE = pd.concat(rmse)
    RMSE.droplevel(level=1)
    return RMSE


def load_regression(
    place, normalised_profile, daily=False, trend=1, fourier_component=0
):
    """Regress on the weekly (daily) profile for each country with Fourier component and trend. Currently only works for weekly.

    Keyword arguments:
    place -- list of keys of data
    normalised_profile -- normalised values (hourly demand/daily average)
    daily -- do a regression for each day (default False)
    trend -- include (order of) trend (default 1, i.e. linear trend)
    fourier_component -- include Fourier terms (default 0)
    """

    load = {}
    aic = {}
    parameters_weekly = {}
    for i in normalised_profile.keys():
        if daily == False:
            p = 168
            index = normalised_profile[i].index
        seas = Seasonality(period=p)
        det_process = DeterministicProcess(
            index, order=trend, fourier=fourier_component, additional_terms=[seas]
        )
        load_test = AutoReg(
            normalised_profile[i]["demand"].fillna(method="bfill"),
            lags=0,
            trend="n",
            seasonal=False,
            deterministic=det_process,
        )
        load[i] = load_test.fit()
        aic[i] = round(load[i].aic, 3)
        # Read weekly parameters
        shift_params = trend + fourier_component
        parameters_weekly[i] = load[i].params[shift_params : 168 + shift_params]
        parameters_weekly[i].index = range(0, 168)
    data_par_weekly = pd.DataFrame.from_dict(parameters_weekly)
    aic_copy = pd.DataFrame(index=aic.keys())
    aic_copy["aic"] = aic.values()
    aic = aic_copy
    # Reorder parameters (by shifting by the numeric value of the first weekday of the first observation
    first_day = []
    first_day = normalised_profile[place[0]].iloc[0]["weekday"]
    shift = 24 * int(first_day)
    new_index = list(range((168 - shift), 168))
    new_index.extend(list(range(0, (168 - shift))))
    new_par = data_par_weekly.reindex(new_index)
    data_par_weekly_reordered = new_par
    data_par_weekly_reordered.index = range(0, 168)
    return load, aic, data_par_weekly_reordered


def compute_cdd_hdd(data, place, threshold_hdd=15.5, threshold_cdd=15.5):
    """[To be used for data containing load.]Compute CDD and HDD on the given data.

    Keyword arguments:
    data -- dataframe with daily values (as a dict) to manipulate
    place -- list of keys of data
    threshold_hdd (default 15.5) #Fix temperature threshold to be at 15.5 degrees Celcius, as for the EU in https://en.wikipedia.org/wiki/Heating_degree_day.
    threshold_cdd (default 15.5)
    """

    daily_hc = {}
    for i in place:
        daily_hc[i] = data[i].copy(deep=True)
        daily_hc[i].columns = ["demand", "weekday", "heating", "holiday"]
        daily_hc[i]["heating"] = np.maximum(threshold_hdd - data[i]["temp"], 0)
        daily_hc[i]["cooling"] = np.maximum(data[i]["temp"] - threshold_cdd, 0)
    return daily_hc


def compute_cdd_hdd_artificial(data, place, threshold_hdd=15.5, threshold_cdd=15.5):
    """[To be used for creating artificial data without load]. Compute CDD and HDD on the given data.

    Keyword arguments:
    data -- dataframe with daily values (as a dict) to manipulate
    threshold_hdd (default 15.5) #Fix temperature threshold to be at 15.5 degrees Celcius, as for the EU in https://en.wikipedia.org/wiki/Heating_degree_day.
    threshold_cdd (default 15.5)"""

    daily_hc = {}
    for i in place:
        daily_hc[i] = data[i].copy(deep=True)
        daily_hc[i].columns = ["heating", "weekday", "holiday"]
        daily_hc[i]["heating"] = np.maximum(threshold_hdd - data[i]["temp"], 0)
        daily_hc[i]["cooling"] = np.maximum(data[i]["temp"] - threshold_cdd, 0)
    return daily_hc


def plot_cdd_hdd(data, place):
    fig2, axes2 = plt.subplots(ncols=1, nrows=nb_c, figsize=(6, 6 * nb_c))
    for (i, x) in enumerate(place):
        sns.scatterplot(
            x=data[x]["heating"],
            y=data[x][data[x]["weekday"] == 6]["demand"],
            ax=axes2[i],
            color="green",
        )
        sns.scatterplot(
            x=data[x]["heating"],
            y=data[x][data[x]["weekday"] == 5]["demand"],
            ax=axes2[i],
            color="b",
        )
        sns.scatterplot(
            x=data[x]["heating"],
            y=data[x][data[x]["weekday"] < 5]["demand"],
            ax=axes2[i],
            color="r",
        )
        sns.scatterplot(
            x=data[x]["heating"],
            y=data[x][data[x]["holiday"] == True]["demand"],
            ax=axes2[i],
            color="black",
        )
        axes2[i].set_title(f"HDD vs. Demand for {place[i]}")
        axes2[i].set_ylim(0, data[x]["demand"].max())
        axes2[i].set_xlim(0.5, data[x]["heating"].max())
    plt.show()
    fig2, axes2 = plt.subplots(ncols=1, nrows=nb_c, figsize=(6, 6 * nb_c))
    for (i, x) in enumerate(place):
        sns.scatterplot(
            x=data[x]["cooling"],
            y=data[x][data[x]["weekday"] == 6]["demand"],
            ax=axes2[i],
            color="green",
        )
        sns.scatterplot(
            x=data[x]["cooling"],
            y=data[x][data[x]["weekday"] == 5]["demand"],
            ax=axes2[i],
            color="b",
        )
        sns.scatterplot(
            x=data[x]["cooling"],
            y=data[x][data[x]["weekday"] < 5]["demand"],
            ax=axes2[i],
            color="r",
        )
        sns.scatterplot(
            x=data[x]["cooling"],
            y=data[x][data[x]["holiday"] == True]["demand"],
            ax=axes2[i],
            color="black",
        )
        axes2[i].set_title(f"CDD vs. Demand for {x}")
        axes2[i].set_ylim(0, data[x]["demand"].max())
        axes2[i].set_xlim(0.5, 10)
    plt.show()


def daily_regression(
    place,
    dict,
    weekly_regression,
    daily=False,
    trend=1,
    fourier_component=0,
):
    """Regress on the daily values for each country with Fourier component and trend.

    Keyword arguments:
    place -- list of keys of data
    normalised_profile -- normalised values (hourly demand/daily average) (default weekly_load)
    daily -- do a regression for each day (default False)
    trend -- include (order of) trend (default 1, i.e. linear trend)
    fourier_component -- include Fourier terms (default 0)
    """

    def read_parameters_nolags(
        dict, model, lags=168, cooling=False, shift=fourier_component + trend
    ):
        """Based on dict and its keys, read out the parameter of a model of the form
        y ~ dummy_hour_of_the_week + temperature (lags is the seasonal component)"""
        parameters_daily = {}
        temp_par = {}
        trend_par = {}
        fourier_par = {}
        for i in dict.keys():
            if trend == 0:
                trend_par[i] = 0
                if fourier_component == 0:
                    fourier_par[i] = 0
                else:
                    fourier_par[i] = model[i].params[0:shift]
            else:
                trend_par[i] = model[i].params[0]
                if fourier_component == 0:
                    fourier_par[i] = 0
                else:
                    fourier_par[i] = model[i].params[1:shift]
            parameters_daily[i] = model[i].params[shift : shift + lags]
            parameters_daily[i].index = range(0, lags)
            if cooling == True:
                temp_par[i] = model[i].params[shift + lags : shift + lags + 2]
            else:
                temp_par[i] = model[i].params[lags + shift]
        data_par = pd.DataFrame.from_dict(parameters_daily)
        if cooling == True:
            data_temp = pd.DataFrame(temp_par).T
            data_temp.columns = ["par_heating", "par_cooling"]
        # Reorder the parameters:
        first_day = []
        first_day = int(weekly_regression[place[0]].iloc[0]["weekday"])
        new_index = list(range((7 - first_day), 7))
        new_index.extend(list(range(0, (7 - first_day))))
        new_par_daily = data_par.reindex(new_index)
        data_par_daily_reordered = new_par_daily
        data_par_daily_reordered.index = range(0, 7)
        return data_par_daily_reordered, data_temp, trend_par, fourier_par

    res_hc = {}
    aic_hc = {}
    for i in weekly_regression.keys():
        if daily == False:
            p = 7
            index = weekly_regression[i].index
        seas = Seasonality(period=7)
        det_process = DeterministicProcess(
            index,
            order=trend,
            period=p,
            additional_terms=[seas],
            fourier=fourier_component,
        )
        test = AutoReg(
            weekly_regression[i]["demand"].fillna(method="bfill"),
            lags=0,
            trend="n",
            seasonal=False,
            exog=weekly_regression[i][["heating", "cooling"]],
            deterministic=det_process,
        )
        res_hc[i] = test.fit()
        aic_hc[i] = round(res_hc[i].aic, 3)
    aic_copy = pd.DataFrame(index=aic_hc.keys())
    aic_copy["aic"] = aic_hc.values()
    aic_hc = aic_copy
    for i in res_hc.keys():
        # remove statistically insignificant parameters
        for (j, x) in enumerate(res_hc[i].params):
            if res_hc[i].pvalues[j] > 0.025:
                print(
                    i,
                    j,
                    "has p-value",
                    res_hc[i].pvalues[j],
                    "therefore we consider it statistically insignificant and set the parameter to 0",
                )
                res_hc[i].params[j] = 0
    par_hc, temp_hc, trend_par, fourier_par = read_parameters_nolags(
        res_hc, res_hc, lags=7, cooling=True, shift=fourier_component + trend
    )
    trend_copy = pd.DataFrame(index=trend_par.keys())
    trend_copy["par_trend"] = trend_par.values()
    trend_par = trend_copy
    fourier_copy = pd.DataFrame(index=fourier_par.keys())
    fourier_par = fourier_copy
    return par_hc, temp_hc, aic_hc, trend_par, fourier_par


def create_daily_data(
    daily_profile,
    trend,
    input_data,
    temp_par,
    place,
    start,
    end,
    validation_days=0,
):
    """Create artificial data based on two regressions from before, one on the weekly load profile and one on the regression on daily values. Also outputs number of days for\
    validation purposes in subsequent period.
    
    Keyword arguments:
    daily_profile -- regression on daily temperature/demand values
    trend -- whether to include trend parameters
    input_data -- temperature data to be used as input
    temp_par -- temperature parameters from regression
    place -- list of keys of data 
    [start, end) -- time period to be studied 
    validation_days -- number of days to move trend for validation runs
    """

    artificial_daily = {}
    first_day = []
    first_day = int(input_data[place[0]].iloc[0]["weekday"])
    days = 0
    for year in range(start, end):
        if calendar.isleap(year) == True:
            days += 366
        else:
            days += 365
    for i in place:
        artificial_daily[i] = np.zeros(days)
        for j in range(days):
            k = (j + first_day) % 7
            if input_data[i]["holiday"].iloc[j] == 1:
                artificial_daily[i][j] = (
                    trend.loc[i] * (validation_days + j)
                    + daily_profile[i].iloc[6]
                    + input_data[i].iloc[j]["heating"] * temp_par.loc[i]["par_heating"]
                    + input_data[i].iloc[j]["cooling"] * temp_par.loc[i]["par_cooling"]
                )
            else:
                artificial_daily[i][j] = (
                    trend.loc[i] * (validation_days + j)
                    + daily_profile[i].iloc[k]
                    + input_data[i].iloc[j]["heating"] * temp_par.loc[i]["par_heating"]
                    + input_data[i].iloc[j]["cooling"] * temp_par.loc[i]["par_cooling"]
                )
        artificial_daily[i] = pd.Series(artificial_daily[i], index=input_data[i].index)
    return artificial_daily, days, first_day


def plot_artificial(artificial, real, frequency, place):
    """Plot time series of artificial daily data (resampled if desired) compared to base data."""
    fig, ax = plt.subplots(nrows=len(place), ncols=1, figsize=(24, 6 * nb_c))
    for (i, x) in enumerate(place):
        artificial[x].resample(frequency).mean().plot(ax=ax[i], color="r", alpha=0.5)
        real[x]["demand"].resample(frequency).mean().plot(
            ax=ax[i], color="b", alpha=0.5
        )
        ax[i].set_title(
            f"Actual measured demand in {x} and projected demand based on regression with frequency {frequency}."
        )
        red_patch = mpl.patches.Patch(color="red", alpha=0.5, label="Projected demand")
        blue_patch = mpl.patches.Patch(
            color="blue", alpha=0.5, label="Real demand data"
        )
        ax[i].legend(handles=[red_patch, blue_patch])


def create_hourly_data(artificial, weekly_profile, real, firstday, place, start, end):
    """Create artificial hourly data from given daily data."""

    artificial_hourly = {}
    for i in place:
        shift = firstday * 24
        timesteps = 0
        for year in range(start, end):
            if calendar.isleap(year) == True:
                timesteps += 8784
            else:
                timesteps += 8760
        artificial_hourly[i] = np.zeros(timesteps)
        for j in range(timesteps):
            k = (j + shift) % 168
            artificial_hourly[i][j] = (
                0 + artificial[i][j // 24] * weekly_profile[i].iloc[k]
            )
        artificial_hourly[i] = pd.Series(artificial_hourly[i], index=real[i].index)
    return artificial_hourly


def plot_scatter_artificial(artificial, real, place, frequency=["1H", "1D"]):
    fig, ax = plt.subplots(
        nrows=nb_c, ncols=len(frequency), figsize=(6 * len(frequency), 6 * nb_c)
    )
    for (i, f) in enumerate(frequency):
        for (j, x) in enumerate(place):
            sns.scatterplot(
                x=real[x]["demand"].resample(f).mean(),
                y=artificial[x].resample(f).mean(),
                ax=ax[j, i],
            )
            sns.scatterplot(
                x=artificial[x].resample(f).mean(),
                y=artificial[x].resample(f).mean(),
                ax=ax[j, i],
                color="red",
            )
            ax[j, i].set_title(
                f"Projected demand vs. real demand {x} with frequency {f}"
            )
            ax[j, i].set_xlabel(f"Real demand in {x}")
            ax[j, i].set_ylabel(f"Projected demand in {x}")
            ax[j, i].set_xlim(
                artificial[x].resample(f).mean().min(),
                artificial[x].resample(f).mean().max(),
            )
            ax[j, i].set_ylim(
                artificial[x].resample(f).mean().min(),
                artificial[x].resample(f).mean().max(),
            )
    plt.show()
