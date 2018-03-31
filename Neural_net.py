import ast


def get_data(open_file):
    for line in open_file:
        price_every_five_minutes_list = ast.literal_eval(line)
        print(len(price_every_five_minutes_list))


def scale_data(stock_price_data):
    percent_change_data = []
    # go through every stock
    for i in range(len(stock_price_data)):
        percent_change_data.append([])
        # go through every price of the current stock except the last price
        for j in range(len(stock_price_data[i])-1):
            current_percent_change = (stock_price_data[i][j+1]-stock_price_data[i][j]) \
                                     /stock_price_data[i][j]
            percent_change_data[i].append[current_percent_change]

    return percent_change_data

def assign_value(stock_price_data):
    # go through every stock
    for i in range(len(stock_price_data)):
        # go through every price of the current stock
        for j in range(len(stock_price_data[i])):
            