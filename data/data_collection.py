import datetime
from statistics import stdev
import time

import requests
import xlsxwriter as xlsxwriter
from pandas_datareader import data
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

from deap import base, creator, tools
import random

import ast

def test_finance():
    date = datetime.datetime(2013, 9, 20)
    nextday = date + datetime.timedelta(days=900)
    df = data.DataReader('AAPL', 'google', date, nextday)
    print(df.values.tolist())
    print(df)
    print('----------------')
    print(df.iloc[0])
    print('----------------')
    print(df.iloc[0][0])
    return df


def get_stock_graph_data(ticker):
    url = 'https://www.google.ca/search?q={}&source=lnms&tbm=fin&sa=X&ved=0'

    # tells server who is doing the search
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    }


    # get html code from the google search
    try:
        response = requests.get(url.format(ticker), headers=headers)
    except:
        return 'not found'

    html_code = response.text
    start_index = html_code.find('["chart_data')
    end_index = html_code.find(']\n]\n]', start_index)
    chart_data = html_code[start_index:end_index]

    chart_data = chart_data.replace('\\n', '')
    chart_data = chart_data.replace('\\', '')
    chart_data = chart_data.replace('%', '')
    chart_data = chart_data.replace('"', '')

    chart_data_start_index = chart_data.find('[[[')
    chart_data_end_index = chart_data.find(']]]')
    chart_data = chart_data[chart_data_start_index+3:chart_data_end_index+1]


    chart_data_list = ast.literal_eval(chart_data)
    price_list = []

    for five_minute_interval in chart_data_list:
        price_list.append(five_minute_interval[2])
    print(price_list)
    return price_list

def get_ticker_list():
    # Excel file
    excelfile = 'NYSE.xlsx'
    # all sheets get loaded into dataframe
    df_all_sheets = pd.ExcelFile(excelfile)
    # Loads a single sheet into a DataFrame by name
    excel_df = df_all_sheets.parse('entire NYSE')
    # gets number of rows ([1] would be number of columns)
    rows = excel_df.shape[0]
    # convert into array
    excel_df = excel_df.values

    # create lists for dynamic storage of names and tickers
    ticker_list = []
    for x in range(0, rows):
        ticker_list.append(excel_df[x][0])

    return ticker_list


def get_prices():
    ticker_list = get_ticker_list()
    for ticker in ticker_list:
        time.sleep(2)
        try:
            price_list = get_stock_graph_data(ticker)
            textfile = open('price_data16.txt', 'a', encoding='utf-8')
            textfile.write(str(price_list)+'\n')
            textfile.close()
        except:
            pass

get_prices()