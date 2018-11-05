from grapher import save_graph, save_graph_mm
from models_indicators_TA import *
import random
import csv
import time

#load data file from here
fileloc = # file location

#save chart to this filename
backname = '_75_day_mm_backtest_25bp.jpg'

#save test results to this filename
csvfile = 'ETH_MM_25bp.csv'

#pairs to be tested
eth_pairs = []


# Used for rounding decimal places. Ignore.
two = ('BTCUSDT', 'ETHUSDT', 'BCCUSDT', 'LTCUSDT', 'NEOUSDT', 'ONTUSDT', 'QTUMUSDT')
four = ('EOSUSDT', 'BNBUSDT', 'IOTAUSDT', 'ICXUSDT', 'PAXUSDT', 'ETCUSDT', 'NULSUSDT')
five = ('LTCETH', 'DASHETH', 'XMRETH', 'BCCETH', 'ZECETH', 'REPETH', 'XRPUSDT', 'ADAUSDT', 'XLMUSDT', 'TRXUSDT', 'VETUSDT')
six = ('BCCBTC', 'BTGBTC', 'DASHBTC', 'DGDBTC', 'ETCBTC', 'ETHBTC', 'LTCBTC', 'NEOBTC', 'OMGBTC', 'REPBTC', 'XMRBTC', 'BNBETH', 'ETCETH', 
    'QTUMETH', 'MCOETH', 'MTLETH', 'NEOETH', 'ARKETH', 'LSKETH', 'OMGETH', 'TRIGETH', 'EOSETH', 'STRATETH', 'BTGETH', 'KMDETH', 'ZECBTC',
    'ONTETH', 'PPTETH', 'WANETH', 'NASETH', 'BNTETH', 'NANOETH', 'STEEMETH', 'WAVESETH', 'WTCETH', 'AEETH', 'ICXETH', 'AIONETH', 'ZENBTC',
    'DCRBTC', 'INSETH', 'RLCETH')
seven = ('AEBTC', 'BNBBTC', 'CLOAKBTC', 'EOSBTC', 'ICXBTC', 'GXSBTC', 'GVTBTC', 'KMDBTC', 'LSKBTC', 'LUNBTC', 'NANOBTC', 'NASBTC', 'MODBTC', 'HCBTC'
    'NXSBTC', 'ONTBTC', 'PPTBTC', 'STRATBTC', 'WANBTC', 'WAVESBTC', 'WTCBTC', 'OAXETH', 'GRSETH', 'ARNETH', 'VIBEETH', 'SALTBTC', 'VIABTC', 'EDOBTC',
    'PIVXBTC')
eight = ('ADABTC', 'ADXBTC', 'AMBBTC', 'APPCBTC', 'ARNBTC', 'BATBTC', 'BCPTBTC', 'BQXBTC', 'BRDBTC', 'BTSBTC',
    'CMTBTC', 'CVCBTC', 'ELFBTC', 'ENGBTC', 'EVXBTC', 'GNTBTC', 'GRSBTC', 'IOTABTC', 'KNCBTC', 'LINKBTC', 'LOOMBTC',
    'MANABTC', 'MDABTC', 'NULSBTC', 'POWRBTC', 'RDNBTC', 'SUBBTC', 'SYSBTC', 'THETABTC', 'XLMBTC', 'XRPBTC', 'ZRXBTC', 
    'ADAETH', 'XLMETH', 'XRPETH', 'ELFETH', 'FUNETH', 'BCPTETH', 'SNGLSETH', 'DENTETH', 'XVGETH', 'ZRXETH', 'QKCETH', 
    'MANAETH', 'XEMETH', 'GTOETH', 'IOTAETH', 'LOOMETH', 'SNMETH', 'NULSETH', 'MFTETH', 'TRXETH', 'VETETH', 'ZILETH', 
    'QLCETH', 'DNTETH', 'KEYETH', 'CMTETH', 'PHXETH', 'STORMETH', 'IOTXETH', 'ENJETH', 'LENDETH', 'POEETH', 'DOCKETH',
    'CHATETH', 'CNDETH', 'TNTETH', 'WPRETH', 'MTHETH', 'YOYOBTC', 'PHXBTC', 'RVNBTC', 'QKCBTC', 'RLCBTC', 'DNTBTC', 'INSBTC',
    'ASTBTC', 'GOBTC', 'AIONBTC', 'GTOBTC', 'VIBBTC', 'LRCBTC', 'XEMBTC', 'DLTBTC', 'WINGSBTC', 'WABIBTC', 'OAXBTC',
    'POAETH', 'AGIETH', 'YOYOETH') 

#random.shuffle(eth_pairs) 


#Create a spreadsheet of trades and win & profit ratios
def log_backtesting(filename, pair, tradecount, profit, gain_perc, win, loss, win_perc):
    try:
        with open(filename) as f:
            numline = 10
    except:
    	numline = 0
    entry = [pair, tradecount, profit, gain_perc, win, loss, win_perc]
    with open(filename, 'a', newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        if numline == 0:
            wr.writerow(['PAIR', 'TRADES', 'PROFIT', 'GAIN %', 'WINS', 'LOSSES', 'WIN %'])
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(entry)
        fp.close()

count = 0
for pair in eth_pairs:
    now = time.time()
    if pair in five:
        decimal_places = 5
    elif pair in six:
        decimal_places = 6
    elif pair in seven:
        decimal_places = 7
    elif pair in eight:
        decimal_places = 8
    elif pair in two:
        decimal_places = 2
    elif pair in four:
        decimal_places = 4
    coin_value = pair.replace(str('ETH'), '')
    file = pair + fileloc
    df = pd.read_csv(file)

    '''
    CREATE INDICATORS HERE USING models_indicators_TA FILE

    df = standard_deviation(df, 'Close', 10) 
    df = MA(df, '10 Period Std Dev', 10) 
    df = ratio(df, '10 Period Std Dev', '10 Period Std Dev 10 MA') 
    df = get_range_mm(df, 10)
    df = get_range(df, 10)
    df = mm_bid_ask(df, 0.005)
    df['VolumeRatio'] = round(df['Base Volume']/df['Base Volume'].rolling(1152).mean(), 2)
    df['BuyRatio'] = round(df['Taker Base Volume']/df['Base Volume'], 2)
    df['SellRatio'] = 1 - df['BuyRatio']

    '''

    print('Loading Market Making Algo..')

    '''
    CREATE ALGO HERE

    conditions_mm = [
    (df['10 Period Std Dev/10 Period Std Dev 10 MA Ratio'] < 100)
    ]
    '''

    choices = [df['Close']]
    choices_ask_bid = [np.nan]
    df['MM'] = np.select(conditions_mm, choices, default=0)
    #df['Bid'] = np.select(conditions_mm2, choices_ask_bid, default=df['Bid'])
    #df['Ask'] = np.select(conditions_mm3, choices_ask_bid, default=df['Ask'])

    for index, row in df.iterrows():
        if row['MM'] == 0:
            df.at[index, 'Bid'] = np.nan
            df.at[index, 'Ask'] = np.nan
        #limits highest bid to max 1/4% below the high
        if (row['Range_High'] - (row['Range_High'] * 0.0025)) < row['Bid']:
            df.at[index, 'Bid'] = (row['Range_High'] - (row['Range_High'] * 0.0025))
        #limits highest bid to at least 1/4% above the low
        if (row['Range_Low'] + (row['Range_Low'] * 0.0025)) > row['Ask']:
            df.at[index, 'Ask'] = (row['Range_Low'] + (row['Range_Low'] * 0.0025))
        #limits highest ask to range high
        if (row['Range_High']) < row['Ask']:
            df.at[index, 'Ask'] = row['Range_High']
        #limits lowest bid to range low
        if (row['Range_Low']) > row['Bid']:
            df.at[index, 'Bid'] = row['Range_Low']


    buys = []
    sells = []
    trigger = 0
    for index, row in df.iterrows():
        if row['Bid'] >= row['Low']:
        	if trigger == 0:
	            buys.append([index, float(row['Bid']), row['VolumeRatio'], row['BuyRatio'], row['SellRatio']])
	            trigger = 5
	            #print([index, row['Bottom'], row['VolumeRatio']])
        
        if row['Ask'] <= row['High']:
            if trigger == 5:
                sells.append([index, float(row['Ask']), row['VolumeRatio'], row['BuyRatio'], row['SellRatio']])
                trigger = 0

    profit = []
    gain_perc = []
    win =[]
    loss = []
    n = 0
    while n < len(sells):
        prof = sells[n][1] - buys[n][1]
        profit.append(prof)
        gain_perc.append(round(((prof/buys[n][1])*100), 2))
        win.append(1) if prof > 0 else loss.append(1)
        n = n+1

    if n > 0:
        overal_gain_perc = round(sum(gain_perc)/len(sells), 2)
        win = sum(win)
        loss = sum(loss)
        profit = round(float(overal_gain_perc/100) * float(win+loss), 2)
        win_perc = round((win/(win+loss))*100, 2)

        file_csv = csvfile
        log_backtesting(file_csv, pair, win+loss, profit, overal_gain_perc, win, loss, win_perc)

    filename = str(coin_value) + backname
    save_graph_mm(df, filename, buys, sells, 'Close')
    print('Chart saved as ' + str(filename))

    end = time.time()
    count = count+1
    eta_count = len(eth_pairs) - count
    eta = round(((end-now) * eta_count)/60)
    print('Completed ' + str(count) + '/' + str(len(eth_pairs)) + ' ETH Pairs.. ' + str(eta) + ' minutes remaining..')

