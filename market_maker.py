#from trade_file import *
from models_indicators_TA import *
import pandas as pd
import numpy as np
import time
import datetime
from random import randint


class Bot:
    def __init__(self):
        try:
            print('Market maker started')

        except:
            print('Check network connection.')

    def start(self):
        start = time.time()

        # -----SETTINGS------------------------------
        main_coin = 'ETH'
        coins = [] # Coins to be traded. ----ex: ['LOOM', 'WTC', 'REP']. Ideally you would have balances in all coins being traded, albeit that is not necessary.
        position_size = # Size (in main_coin amount) to use to per order. ----ex: 3
        profit_basis_points = # Profit margin in basis points. 1bp is equal to 0.01%. ----ex: 40
        stdev_ratio = # Normalized short term standard deviation, used to disable the bot when the market starts moving fast. To disable this function, enter a high number >10.
        max_position = # The maximum position size per coin (in main_coin). ----ex: 15
        time_interval = # The length of each round in seconds. ----ex: 180 (3 min)

        # optional - have the bot message you on slack/discord whenever there is action
        error_channel = 'errors'
        slack_channel = 'market-making-ETH'
        botname = ''

        minutes = 99999999999999  #How long to run the bot for. 
        # -------------------------------------------

        pairs = []
        for i in coins:
            pair = str(i) + str(main_coin)
            pairs.append(pair)

        profit_spread = (profit_basis_points/2) / 10000

        positions_file = str(main_coin) + '_Market_Making.csv'

        #Pair counter
        #This may need to be modified to work to the syntax of your exchange
        #The purpose of this is to round the order price to the appropriate decimal places to avoid throwing an error
        five = ('LTCETH', 'DASHETH', 'XMRETH', 'BCCETH', 'ZECETH', 'REPETH')
        six = ('BNBETH', 'ETCETH', 'QTUMETH', 'MCOETH', 'MTLETH', 'NEOETH', 'ARKETH', 'LSKETH', 'OMGETH', 'TRIGETH', 'EOSETH', 'STRATETH', 'BTGETH', 'KMDETH',
            'ONTETH', 'PPTETH', 'WANETH', 'NASETH', 'BNTETH', 'NANOETH', 'STEEMETH', 'WAVESETH', 'WTCETH', 'AEETH', 'ICXETH', 'AIONETH')
        seven = ('OAXETH', 'GRSETH', 'ARNETH', 'VIBEETH')
        eight = (
            'ADAETH', 'XLMETH', 'XRPETH', 'ELFETH', 'FUNETH', 'BCPTETH', 'SNGLSETH', 'DENTETH', 'XVGETH', 'ZRXETH', 'QKCETH', 
            'MANAETH', 'XEMETH', 'GTOETH', 'IOTAETH', 'LOOMETH', 'SNMETH', 'NULSETH', 'MFTETH', 'TRXETH', 'VETETH', 'ZILETH', 
            'QLCETH', 'DNTETH', 'KEYETH', 'CMTETH', 'PHXETH', 'STORMETH', 'IOTXETH', 'ENJETH', 'LENDETH', 'POEETH', 'DOCKETH',
            'CHATETH', 'CNDETH', 'TNTETH', 'WPRETH', 'MTHETH', 'PAXETH') 


        #Coin decimal place counter
        #The amount of decimal places each coin can carry. This is used to round positions & balances to the appropriate decimal place.
        zero = ('ADA', 'ADX', 'AMB', 'APPC', 'ARN', 'BAT', 'BCPT', 'BQX', 'BRD', 'BTS', 'CMT', 'CVC', 'ENG', 'EVX', 'GNT', 'GRS', 'XLM', 'XRP', 'ELF', 'XVG', 
            'FUN', 'ZRX', 'DENT', 'QKC', 'SNGLS', 'XEM', 'KNC', 'BCPT', 'GTO', 'IOTA', 'LOOM', 'NULS', 'MFT', 'TRX', 'VET', 'ZIL', 'QLC', 'SNM', 'KNC', 'LINK', 
            'MANA', 'MDA', 'PAX', 'POWR', 'RDN', 'SUB', 'SYS', 'THETA', 'DNT', 'KEY', 'OAX', 'STORM', 'PAX', 'PHX', 'IOTX', 'ENJ', 'LEND', 'POE', 'POA', 'DATA', 'DOCK',
            'CHAT', 'CND', 'VIBE', 'TNT', 'WPR', 'MTH')
        two = ('BNB', 'ETC', 'QTUM', 'MCO', 'MTL', 'NEO', 'LSK', 'OMG', 'ARK', 'EOS', 'STRAT', 'TRIG', 'BTG', 'KMD', 'ONT', 'PPT', 'WAN', 'NAS', 'LTC', 'DASH', 
            'XMR', 'BCC', 'ZEC', 'REP', 'NANO', 'BNT', 'AE', 'CLOAK', 'ICX', 'GXS', 'GVT', 'LUN', 'NXS', 'WAVES', 'WTC', 'DGD', 'STEEM', 'AION')

        # Create dataframe of positions
        snapshot_df = pd.DataFrame(columns=['Coin', 'Start_Bal', 'Position', 'Cost_Basis', 'Total_Traded', 'Profit', 'Bid_at', 'Offer_at', 'Bid_order', 'Offer_order', 'Commissions'])
        coin_list = np.asarray(coins)

        if main_coin not in coin_list:
            coin_list = np.insert(coin_list, 0, str(main_coin))

        snapshot_df['Coin'] = coin_list

        # Try to load dataframe from saved snapshot if available. 
        # The bot will constantly save snapshots of the portfolio so that if the bot is disrupted for any reason, it can load and pick off where it left.
        try:
            snapshot_df = pd.read_csv(positions_file)

        # If this is your first time running the bot it will create a new portfolio snapshot.
        # Note that the current balances in your account will be considered position neutral,
        # meaning that if you have a balance of a certain coin it will maintain that balance and
        # not sell to 0.
        except:
            balances = get_all_balances() 
            for i in coin_list:
                coin_bal = get_balances_for_coin(str(i), balances)
                if i in zero:
                    snapshot_df['Start_Bal'][snapshot_df.Coin == str(i)] = int(coin_bal)
                elif i == main_coin:
                    snapshot_df['Start_Bal'][snapshot_df.Coin == str(i)] = round(coin_bal, 4)
                else:
                    snapshot_df['Start_Bal'][snapshot_df.Coin == str(i)] = round(coin_bal, 2)
                snapshot_df['Position'][snapshot_df.Coin == str(i)] = 0
                snapshot_df['Cost_Basis'][snapshot_df.Coin == str(i)] = 0
                snapshot_df['Total_Traded'][snapshot_df.Coin == str(i)] = 0
                snapshot_df['Profit'][snapshot_df.Coin == str(i)] = 0
                snapshot_df['Bid_at'][snapshot_df.Coin == str(i)] = 'OFF'
                snapshot_df['Offer_at'][snapshot_df.Coin == str(i)] = 'OFF'
                snapshot_df['Bid_order'][snapshot_df.Coin == str(i)] = 0
                snapshot_df['Offer_order'][snapshot_df.Coin == str(i)] = 0

                snapshot_df['Commissions'][snapshot_df.Coin == str(i)] = 0
        snapshot_df.to_csv(positions_file, index=False)

        while time.time() < start+ 60 * int(minutes):
            try:
                # if you are using an exchange that uses exchange tokens to pay for fees,
                # add a function here that will replenish the token if the balance is low
                # checkExchangeTokenBalance()

                print()
                print(datetime.datetime.now())
                print()
                print('Base Currency: ' + str(main_coin))
                print()

                #check for orders, update, and check for execution on existing orders on ALL COINS
                snapshot_df = check_open_order_list(snapshot_df, main_coin)
                #Update positions
                snapshot_df = update_positions(snapshot_df)

                print(snapshot_df)
                print()

                # Save portfolio state
                try:
                    snapshot_df.to_csv(positions_file, index=False)
                except:
                    print('Couldnt save snapshot file')

                # Loop through pairs, gather data, make calculations, and put in trades
                for i in pairs:
                    # DATA MODIFIERS
                    coin_value = i.replace(str(main_coin), '')
                    df = load_dataframe(str(i)) # pull data from your source to create a DF. OHLCV format.
                    df = standard_deviation(df, 'Close', 30) # Calculates the standard deviation for the chosen timeframe and adds to the DF (in this case 30 min)
                    df = MA(df, '30 Period Std Dev', 1440) # Calculates the MA of the standard deviation above for a given timeframe (in this case 1440 min = 1 day)
                    df = ratio(df, '60 Period Std Dev', '60 Period Std Dev 1440 MA') # Normalizes the stdev above so it can be scaled across all coins
                    df = mm_bid_ask(df, profit_spread) # Calculates points at which the bot should bid and offer at, based on the profit spread in settings.
                    df = get_range_mm(df, 3000) # Calculates the low and high (range) the pair is trading in for a given timeframe. Used to make other calculations.

                    #Figure out how to round the current pair
                    if i in five:
                        decimal_places = 5
                    elif i in six:
                        decimal_places = 6
                    elif i in seven:
                        decimal_places = 7
                    elif i in eight:
                        decimal_places = 8

                    # This limits the bid to a maximum price of a 1/4% lower than the range high. 
                    if (df['Range_High'].iloc[-1] - (df['Range_High'].iloc[-1] * 0.0025)) < df['Bid'].iloc[-1]:
                        bid_at_price = round(float(df['Range_High'].iloc[-1] - (df['Range_High'].iloc[-1] * 0.0025)), decimal_places)

                    # This limits the bid to be at lowest the range low. If not, it will be bidding too low and will not likely fill.
                    # Note that as the market drops lower the bid will either drop along with it or disable altogether if volatility is high.
                    elif (df['Range_Low'].iloc[-1]) > df['Bid'].iloc[-1]:
                        bid_at_price = df['Range_Low'].iloc[-1]

                    else:
                        bid_at_price = round(float(df['Bid'].iloc[-1]), decimal_places)

                    # Limits the offer to a minimum price of 1/4% above the range low
                    if (df['Range_Low'].iloc[-1] + (df['Range_Low'].iloc[-1] * 0.0025)) > df['Ask'].iloc[-1]:
                        offer_at_price = round(float(df['Range_Low'].iloc[-1] + (df['Range_Low'].iloc[-1] * 0.0025)), decimal_places)

                    # Limits max offer to range high
                    elif (df['Range_High'].iloc[-1]) < df['Ask'].iloc[-1]:
                        offer_at_price = df['Range_High'].iloc[-1]

                    else:
                        offer_at_price = round(float(df['Ask'].iloc[-1]), decimal_places)

                    # Updates the DF with calculations and activity
                    snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = bid_at_price
                    snapshot_df['Offer_at'][snapshot_df.Coin == str(coin_value)] = offer_at_price
                    midpoint = round((float(df['Range_High'].iloc[-1]) + float(df['Range_Low'].iloc[-1])) / 2, decimal_places)

                    # Prints calculations. Useful to make sure everything is functioning correctly.
                    print(str(coin_value) + ' -- Stdev: ' + str(df['60 Period Std Dev/60 Period Std Dev 1440 MA Ratio'].iloc[-1]) + ', Bid: ' + str(bid_at_price) + ', Ask: ' + str(offer_at_price) + ', Current: ' + str(df['Close'].iloc[-1]))
                    print(str(coin_value) + ' -- 30 min low: ' + str(df['Low'].iloc[-30:-10].min()) + ', 30 min high: ' + str(df['High'].iloc[-30:-10].max()))
                    print(str(coin_value) + ' -- 2 Day Low: ' + str(df['Range_Low'].iloc[-1]) + ', 2 Day High: ' + str(df['Range_High'].iloc[-1]) + '. Midpoint: ' + str(midpoint))

                    # The midpoint of the range is used to disable/enable activity.
                    # This is so the bot does not get too aggressive buying near the top, or selling near the lows.
                    # Note that this only works for highly correlated pairs and can be disabled.
                    if df['Close'].iloc[-1] >= midpoint:
                        buy_position_trading = 'OFF'
                        sell_position_trading = 'ON'
                    else:
                        sell_position_trading = 'OFF'
                        buy_position_trading = 'ON'

                    # This will disable the bid if the market has moved so quickly the bid calculation hasnt caught up, and is bidding above current price.
                    if df['Low'].iloc[-30:-10].min() < df['Bid'].iloc[-1]:
                        snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    # This will disable the bid if there is little chance of getting filled.
                    # In other words, it will activate only if the price is within 3/4% of the bid price (in this case)
                    # The purpose of this is not to tie up capital when it can be used on a different pair.
                    if df['Close'].iloc[-1] > (df['Bid'].iloc[-1]*1.0075):
                        snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    # Same as above, on the sell side.
                    if df['Close'].iloc[-1] < (df['Ask'].iloc[-1]*0.9925):
                        snapshot_df['Offer_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    # Same as above, on the sell side
                    if df['High'].iloc[-30:-10].max() > df['Ask'].iloc[-1]:
                        snapshot_df['Offer_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    # Disables the bot based on stdev setting above (volatility trigger)
                    if df['30 Period Std Dev/30 Period Std Dev 1440 MA Ratio'].iloc[-1] > stdev_ratio: # Modify the DF column to whatever the setting was above
                        snapshot_df['Offer_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'
                        snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    # Trading will NOT disable if there is an open position, meaning it will try to close the position regardless of conditions.
                    if (buy_position_trading == 'OFF') and (float(snapshot_df['Position'][snapshot_df.Coin == str(coin_value)]) > -1):
                        snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    if (sell_position_trading == 'OFF') and (float(snapshot_df['Position'][snapshot_df.Coin == str(coin_value)]) < 1):
                        snapshot_df['Offer_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    # Limits to max position based on setting
                    if ((float(snapshot_df['Position'][snapshot_df.Coin == str(coin_value)])*bid_at_price) > max_position):
                        snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = 'OFF'

                    # Trading will NOT disable if there is an open position, meaning it will try to close positions and remain position neutral.
                    if float(snapshot_df['Position'][snapshot_df.Coin == str(coin_value)]) > 1:
                        snapshot_df['Offer_at'][snapshot_df.Coin == str(coin_value)] = round(float(df['Ask'].iloc[-1]), decimal_places)
                        if float(snapshot_df['Cost_Basis'][snapshot_df.Coin == str(coin_value)]) > 0:
                            snapshot_df['Offer_at'][snapshot_df.Coin == str(coin_value)] = min(round(float(df['Ask'].iloc[-1]), decimal_places), float(snapshot_df['Cost_Basis'][snapshot_df.Coin == str(coin_value)])*1.005)

                    elif float(snapshot_df['Position'][snapshot_df.Coin == str(coin_value)]) < -1:
                        snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = round(float(df['Bid'].iloc[-1]), decimal_places)
                        if float(snapshot_df['Cost_Basis'][snapshot_df.Coin == str(coin_value)]) > 0:
                            snapshot_df['Bid_at'][snapshot_df.Coin == str(coin_value)] = max(round(float(df['Bid'].iloc[-1]), decimal_places), float(snapshot_df['Cost_Basis'][snapshot_df.Coin == str(coin_value)])*(1-0.005))

                print()
                for index, row in snapshot_df.iterrows():
                    if index > 1:
                        print(str(row['Coin']) + ' -- Position: ' + str(row['Position']) +
                            ', Cost Basis: ' + str(row['Cost_Basis']) +
                            ', Bid price: ' + str(row['Bid_at']) +
                            ', Ask price: ' + str(row['Offer_at']))

                        pair = str(row['Coin']) + str(main_coin)

                        if str(row['Coin']) in zero:
                            coin_dec_place = 0
                        elif str(row['Coin']) in two:
                            coin_dec_place = 2

                        if str(row['Coin']) in five:
                            decimal_places = 5
                        elif str(row['Coin']) in six:
                            decimal_places = 6
                        elif str(row['Coin']) in seven:
                            decimal_places = 7
                        elif str(row['Coin']) in eight:
                            decimal_places = 8

                        #Place bid:
                        if row['Bid_at'] != 'OFF':
                            bid, ask = coin_quote(pair) # Get current bid/ask quote

                            # If bid point is below current ask, re-adjust the bid so it is in between the current bid and ask.
                            if ask < float(snapshot_df.at[index, 'Bid_at']):
                                bid_price = round((bid + ((ask-bid)*0.25)), decimal_places)
                            else:
                                bid_price = float(snapshot_df.at[index, 'Bid_at'])
                            
                            # Get balance of coin
                            main_coin_bal = get_coin_balance(str(main_coin))
                            print(str(main_coin) + ' Balance: ' + str(main_coin_bal))

                            # Randomizes the position size between 80-120%. Bidding exactly 3 ETH at a time makes it too obvious a bot is placing orders
                            size_bid = float(randint(position_size * 80, position_size * 120)/100)

                            qty_bid = round(size_bid/bid_price)

                            # Tries to close out position first
                            # It will bid/ask up to 150% of the order size to try close out the position quicker.
                            if float(row['Position']) < -0.5:
                                position = round(abs(float(row['Position'])), coin_dec_place)
                                real_bid = min(position, int(qty_bid*1.5))
                            else:
                                real_bid = qty_bid

                            # Last check to make sure you have the balance to put in the order
                            if main_coin_bal >= (real_bid * bid_price):
                                try:
                                    # Enter order
                                    new_bid_id = place_bid(pair, real_bid, bid_price, decimal_places)
                                    message = 'Bid placed for ' + str(real_bid) + ' ' + str(row['Coin']) + ' @ ' + str(bid_price)
                                    print(message)
                                    snapshot_df['Bid_order'][snapshot_df.Coin == str(row['Coin'])] = int(new_bid_id)
                                    # Messages you when an order is placed. It gets annoying after awhile but is good for testing.
                                    slack_message(message, slack_channel, botname)
                                except:
                                    message = 'Problem putting in bid for ' + str(real_bid) + ' ' + str(row['Coin']) + ' @ ' + str(snapshot_df.at[index, 'Bid_at'])
                                    print(message)
                                    #Messages when there is an error. Keep this on
                                    slack_message(message, error_channel, botname)
                            else:
                                print('Insufficient balance to place bid')

                        #Place Offer
                        if row['Offer_at'] != 'OFF':
                            bid, ask = coin_quote(pair)
                            if bid > float(snapshot_df.at[index, 'Offer_at']):
                                ask_price = round((ask - ((ask-bid)*0.25)), decimal_places)
                            else:
                                ask_price = snapshot_df.at[index, 'Offer_at']

                            target_coin_bal = get_coin_balance(str(row['Coin']))
                            print(str(row['Coin']) + ' Balance: ' + str(target_coin_bal))
                            size_ask = float(randint(position_size * 80, position_size * 120)/100)
                            qty_ask = round(size_ask/ask_price)

                            if float(row['Position']) > 0.5:
                                position = round(float(row['Position']), coin_dec_place)
                                real_ask = min(position, int(qty_ask*1.5))
                            else:
                                real_ask = qty_ask

                            if target_coin_bal >= real_ask:
                                try:
                                    new_sell_id = place_offer(pair, real_ask, ask_price, decimal_places)
                                    message = 'Offer placed for ' + str(real_ask) + ' ' + str(row['Coin']) + ' @ ' + str(ask_price)
                                    print(message)
                                    snapshot_df['Offer_order'][snapshot_df.Coin == str(row['Coin'])] = int(new_sell_id)
                                except:
                                    message = 'Problem putting in sell order for ' + str(real_ask) + ' ' + str(row['Coin']) + ' @ ' + str(snapshot_df.at[index, 'Offer_at'])
                                    print(message)
                                    slack_message(message, error_channel, botname)
                            else:
                                print('Insufficient balance to place sell order')

                        # Save account state again
                        try:
                            snapshot_df.to_csv(positions_file, index=False)
                        except:
                            print('Couldnt save state')

                # Rest X amount of time until it rotates again
                time.sleep(time_interval)

            except:
                print('Ran into an error')

                #Save state in the event of an error
                try:
                    snapshot_df.to_csv(positions_file, index=False)
                except:
                    print('Couldnt save state')
                time.sleep(time_interval)






