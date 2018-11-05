import pandas as pd
import numpy as np
import cvxpy as cvx
import time
import os

from statsmodels.tsa.arima_model import ARMA, ARIMA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller


# Loads data in OHLCVV format
def load_data(file):
    headers = ['Open', 'High', 'Low', 'Close', 'Volume', 'Base Volume']
    df = pd.read_csv(file, names=headers)
    return df


# MODELS

#Autoregressive Integrated Moving Average
def fit_arima(data):
    p = 1 #AR Lag
    q = 1 #MA Lag
    d = 0 #Order of integration
    order = [p, d, q]
    
    arima_model = ARIMA(data, order)
    arima_result = arima_model.fit()

    fittedvalues = arima_result.fittedvalues.values
    arparams = arima_result.arparams
    maparams = arima_result.maparams
   
    return fittedvalues, arparams, maparams

def ARIMA_model(data):
    fittedvalues,arparams,maparams = fit_arima(data)
    arima_pred = pd.Series(fittedvalues)
    plt.plot(data, color=sns.xkcd_rgb["pale purple"])
    plt.plot(arima_pred, color=sns.xkcd_rgb["jade green"])
    plt.title('ARIMA(p=1,d=0,q=1) model predictions');
    print(f"AR parameter {arparams[0]:.2f}, MA parameter {maparams[0]:.2f}")
    plt.show()

# Comparing hedge position to original position. Used in pair trading
def hedge_ratio(df1, df2):
    lr = LinearRegression()
    lr.fit(df1.values.reshape(-1,1),df2.values.reshape(-1,1))

    hedge_ratio = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    print(f"hedge ratio from regression is {hedge_ratio:.4f}, intercept is {intercept:.4f}")
    return hedge_ratio

#Hedge ratio spread
def spread(df1, df2):
    hedge_r = hedge_ratio(df1, df2)
    spread = (df2-df1)*hedge_r

#Check hedge ratio spread 
def is_spread_stationary(spread, p_level=0.05):
    adf_result = adfuller(spread)
    pvalue = adf_result[1]
    
    print(f"pvalue {pvalue:.4f}")
    if pvalue <= p_level:
        print(f"pvalue is <= {p_level}, assume spread is stationary")
        return True
    else:
        print(f"pvalue is > {p_level}, assume spread is not stationary")
        return False

#Optimize weights on positions taken
def optimize_portfolio_of_2(varA, varB, rAB): #var of stock 1, var of stock 2, correlation
    cov = np.sqrt(varA)*np.sqrt(varB)*rAB
    x = cvx.Variable(2)
    P = np.array([[varA, cov],[cov, varB]])
    objective = cvx.Minimize(cvx.quad_form(x,P))
    constraints = [sum(x)==1]
    problem = cvx.Problem(objective, constraints)
    min_value = problem.solve()
    xA,xB = x.value
    
    return xA, xB

def optimize_portfolio(returns, index_weights, scale=.00001):
    m = returns.shape[0]
    cov = np.cov(returns)
    x = cvx.Variable(m)
    portfolio_variance = cvx.quad_form(x, cov)
    distance_to_index = cvx.norm(x - index_weights)
    objective = cvx.Minimize(portfolio_variance + scale * distance_to_index)
    constraints = [x >= 0, sum(x) == 1]
    cvx.Problem(objective, constraints).solve()
    x_values = x.value
    
    return x_values


# TECHNICAL ANALYSIS


# cluster list into groups --used for calculating support
def cluster(data, maxgap):
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][0]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups

# Returns support levels. COUNT is amount of times a level must be tested to be considered support. larger data = larger numbers
def support_price(df, count, decimal_places):
    support_levels = []
    lows = df['Low'].tolist()
    maxgap = (sum(lows)/len(lows)) * 0.002
    grouped = cluster(lows, maxgap=maxgap)
    for i in grouped:
        if len(i) > count:
            support_level = round(sum(i)/len(i), decimal_places)
            support_levels.append(support_level)
    support_price = min(support_levels)
    return support_price

# same thing as support price but in pandas DF (Used for charting)
# COUNT is the number of times a level of support must be hit to be considered support
def support_level(df, timeframe, count, decimal_places):
    for index, row in df.iterrows():
        if index > timeframe:
            support_levels = []
            lows = df.iloc[index-timeframe:index]['Low'].tolist()
            maxgap = (sum(lows)/len(lows)) * 0.002
            grouped = cluster(lows, maxgap=maxgap)
            for i in grouped:
                if len(i) > count:
                    support_level = round(sum(i)/len(i), decimal_places)
                    support_levels.append(support_level)
            if support_levels:
                support_price = min(support_levels)
            else:
                for i in grouped:
                    if len(i) > 5:
                        support_level = round(sum(i)/len(i), decimal_places)
                        support_levels.append(support_level)
                if support_levels:
                    support_price = min(support_levels)
                else:
                    support_price = min(lows)

        else:
            support_price = df.iloc[index]['Low']
        df.at[index, 'Support'] = support_price 
    return df

def resistence_level(df, timeframe, count, decimal_places):
    for index, row in df.iterrows():
        if index > timeframe:
            res_levels = []
            highs = df.iloc[index-timeframe:index]['High'].tolist()
            maxgap = (sum(highs)/len(highs)) * 0.002
            grouped = cluster(highs, maxgap=maxgap)
            for i in grouped:
                if len(i) > count:
                    res_level = round(sum(i)/len(i), decimal_places)
                    res_levels.append(res_level)
            if res_level:
                res_price = max(res_levels)
            else:
                for i in grouped:
                    if len(i) > 5:
                        res_level = round(sum(i)/len(i), decimal_places)
                        res_levels.append(res_level)
                if res_levels:
                    res_price = min(res_levels)
                else:
                    res_price = min(highs)

        else:
            res_price = df.iloc[index]['High']
        df.at[index, 'Resistence'] = res_price 
    return df

# Calculate ratios. Used for normalizing data
def ratio(df, var, ma):
    label = str(var) + '/' + str(ma) + ' Ratio'
    df[label] = round((df[var] / df[ma]), 2)
    return df

# Volume Weight Average Price
def VWAP(df, timeframe):
    df['VWAP'] = df['Base Volume'].rolling(timeframe).sum()/df['Volume'].rolling(timeframe).sum()
    return df

# Slope of a MA
def slope(df, line, n):
    slope = pd.Series(((df[line] - df[line].shift(n)) / n), name = str(n) + ' Period Slope Of ' + str(line))
    slope = round(slope, 2)
    df = df.join(slope)
    return df

# Price distance from MA
def distance_from_MA(df, ma):
    distance_MA = pd.Series(round((((df['Close'] - df[ma]) / df['Close']) * 100), 2), name = 'Distance From ' + str(ma))
    df = df.join(distance_MA)
    return df

# Price range
def get_range(df, timeframe):
    low = pd.Series(df['Low'].shift(1).rolling(timeframe).min(), name = 'Range_Low')
    high = pd.Series(df['High'].shift(1).rolling(timeframe).max(), name = 'Range_High')
    range_difference = pd.Series(round((((high-low) / low) * 100), 2), name = 'Range_Spread')
    df = df.join(high)
    df = df.join(low)
    df = df.join(range_difference)
    df['Range_Mid'] = (df['Range_High']+df['Range_Low'])/2
    return df

# Calculate bid and ask for market making
def mm_bid_ask(df, margin):
    df = Median(df, 'Close', 120)
    bid = pd.Series(df['Close 120 Median'] * (1-margin), name = 'Bid')
    ask = pd.Series(df['Close 120 Median'] * (1+margin), name = 'Ask')
    df = df.join(ask)
    df = df.join(bid)
    return df

#Standard Deviation  
def standard_deviation(df, data_column, n):  
    df = df.join(pd.Series(df[data_column].rolling(n).std(), name = str(n) + ' Period Std Dev'))
    return df  

#Log return
def log_return(df, n):
    periodReturn = pd.Series((np.log(df['Close']/df['Close'].shift(n))), name = str(n) + ' Period Log Return')
    periodReturn = round(periodReturn * 100, 2)
    df = df.join(periodReturn)
    return df

#Standard return
def std_return(df, n):
    periodReturn = pd.Series(round((((df['Close']-df['Close'].shift(n))/df['Close'].shift(n)) * 100), 2), name = str(n) + ' Period Return')
    df = df.join(periodReturn)
    return df

#Forward log return
def forward_return(df, n):
    periodReturn = pd.Series((np.log(df['Close'].shift(-n)/df['Close'])), name = str(n) + ' Forward Return')
    periodReturn = round(periodReturn * 100, 2)
    df = df.join(periodReturn)
    return df

#Moving Average  
def MA(df, variable, n):  
    MA = pd.Series(df[variable].rolling(n).mean(), name = str(variable) + ' ' + str(n) + ' MA')
    df = df.join(MA)  
    return df

#Median
def Median(df, variable, n):  
    med = pd.Series(df[variable].rolling(n).median(), name = str(variable) + ' ' + str(n) + ' Median')
    df = df.join(med)  
    return df

#Median as a percentage between high & low
def median_balance_point(df, high, low):  
    pos = pd.Series(round(((df['Close']-df[low])/(df[high]-df[low])), 2), name = 'Med Balance Point')
    df = df.join(pos)  
    return df

#Mean of a series of mediam balance points
def med_mean(df, high, low, timeframe):
    df = median_balance_point(df, high, low)
    mean = pd.Series(round(df['Med Balance Point'].rolling(timeframe).mean(), 2), name = 'Med Mean')
    df = df.join(mean)
    return df

#Exponential Moving Average  
def EMA(df, variable, n):  
    EMA = pd.Series(pd.ewma(df[variable], span = n, min_periods = n - 1), name = str(variable) + ' ' + str(n) + ' EMA')  
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['Close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['Close'].diff(n - 1)  
    N = df['Close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC')  
    df = df.join(ROC)  
    return df

#Bollinger Bands  
def BBANDS(df, underlying, n):  
    MA = pd.Series(pd.rolling_mean(df[underlying], n))  
    MSD = pd.Series(pd.rolling_std(df[underlying], n))  
    b1 = MA + (MSD * 2)
    B1 = pd.Series(b1, name = 'Upper Band')  
    df = df.join(B1)  
    b2 = MA - (MSD * 2) 
    B2 = pd.Series(b2, name = 'Lower Band')  
    df = df.join(B2)  
    return df

#Stochastic oscillator
def stoch(df):  
    SO = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'Stoch')  
    df = df.join(SO)  
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)  
def stoch_MA(df, nK, nD, nS=1):  
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))  
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))  
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()  
    df = df.join(SOk)  
    df = df.join(SOd)  
    return df  

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')  
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)  
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))  
    df = df.join(ADX)  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(pd.ewma(df['Close'], span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(pd.ewma(df['Close'], span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD')  
    MACDsign = pd.Series(pd.ewma(MACD, span = 9, min_periods = 8), name = 'MACD Signal')  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACD Difference')  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

#Mass Index  
def MassI(df):  
    Range = df['High'] - df['Low']  
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)  
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)  
    Mass = EX1 / EX2  
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['Close'].diff(r1 - 1)  
    N = df['Close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['Close'].diff(r2 - 1)  
    N = df['Close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['Close'].diff(r3 - 1)  
    N = df['Close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['Close'].diff(r4 - 1)  
    N = df['Close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df

#Relative Strength Index  
def RSI(df, n):  
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')  
        DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))  
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    df = df.join(RSI)  
    return df

#Accumulation/Distribution  
def accumulation_distribution(df, n):  
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist')  
    df = df.join(AD)  
    return df

#Ultimate Oscillator  
def ultimate(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['High'] + df['Low'] + df['Close']) / 3  
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI')  
    df = df.join(CCI)  
    return df


# OTHER

#Drop NAN values
def dropNAN(df, column):
    df = df[np.isfinite(df[column])]
    return df

#Delete DF column
def deleteColumn(df, column):
    df.drop(columns = [column])
    return df

#Resize data to n size
def dataSize(df, n):
    df = df[-n:]
    return df
