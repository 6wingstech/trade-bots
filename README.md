
Contents: 
- Market making bot structure
- Library for adding TA and financial modelling
- Library for adding basic machine learning
- File for backtesting algorithms and strategies
- File for graphing

HOW TO USE:

1. BOT

This bot is designed to use a REST API to gather live data at given intervals, make calculations, and execute trades off those calculations. It is set up for market making on cryptocurrency exchanges. It will not work in this current state, but can be made functional easily by doing the following:
  - Input settings for the algorithm at the top of the file, or make your own algo altogether.
  - Add a wrapper to connect to your exchange, and input appropriate credentials
  - Make sure that the wrapper functions are tied correctly to the bot file

There is a step by step tutorial on the file explaining how it works. It is far from perfect so make it better!

Backtest your algorithms first before running it. More below.

2. TA

The TA library uses a pandas dataframe so some knowledge about that is required. 

How it works:

  -load file into dataframe (via the function on there). You can do it directly from your datasource or from a .csv file in OHLCVV format.
  
    df = load_data(file)
    
  -add TA/indicators from the file via df = indicator(params).
  
    df = MA('Close', 20) #would add a 20 period moving average to the closing price, and add it to the dataframe.
    
  Use the graphing file to visualize or make calculations separately on another file. To do so, import the graphing file and 
  
    oscillator_graph(df, 'Close', oscillator1, oscillator2, ...)
    
  for oscillators, and
  
    overlay_graph(df, 'Close', ma1, ma2, ...)
    
  for overlay indicators like MAs. Modify the graphing file to make your own custom charts.
  
  
3. MACHINE LEARNING

This file is still quite messy and I will continue to add to it.

The algorithm will be saved into a .pkl file and will be loaded from there to make predictions in live trading. The input from the live data must be in the exact same format/order as was used during the training.

My experience with machine learning in trading is mixed and my overall opinion is that it is rather unreliable, but I still do believe there is potential. Having said that, I have 2 comments regarding ML:
  1. Use machine learning for deciding variables, not for actual trading. In other words, use ML to optimize position sizes, profit margins, etc. - all the variables a human would have to decide on their own. This is where ML really shines, in optimizing efficiency and in support, rather than proprietary trading on its own.
  2. If you really want to implement ML into a trading strategy, pair it with triggers! In other words, don't feed the entire dataset into a ML algorithm and expect it to figure it out. Filter your data through triggers and then run it through ML. For example, use the MACD to generate buy triggers and feed a dataset of just those triggers into ML to decide whether those were actual buys or not. 

That is regarding technical trading but ML has more applications which I will add on later. For instance, NLP has a good application in analyzing news/blog articles quickly to add a bias to a trading strategy. In the stock world, some people use ML to analyze satelite imagery of how full parking lots are for predicting future sales of that store. This may be irrelevant to cryptocurrencies as of now, but its about being creative.

4. BACKTESTING

This should be where most of your time will be spent. The backtesting file above is quite specific and may need to be modified to suit your needs, but should work fine as is once the commented areas are filled in properly. 

How it works:

  1. Add all indicators/modifiers in appropriate spot
  2. Create algo based off those modifiers
  3. Add pairs for testing at the top & input proper location of data file
  4. Run the file
  
Note you will be required to have the OHLCVV data for all the pairs being tested. The script will loop through your pairs and automatically save a chart.jpg file that looks like this, indicating points at which the system would have bought and subsequently sold.

![alt text](http://6wingstech.com/wp-content/uploads/2018/11/ETH-YOYO_150_day_backtesting_chart_new.jpg)

At the same time, it will log those trades in data format for analysis. A spreadsheet will be created that looks like this:

![alt text](http://6wingstech.com/wp-content/uploads/2018/11/spreadsheet.jpg)

On my computer, it takes approximately 20 min to backtest 160 pairs, each with about 100,000 lines of data on a csv.

Keep going until you get something that is consistent! Remember, any system will take losses at some point, so focus on long term feasibility, not individual trades.

5. CREATING ALGORITHMS

Try different things. What you think should work usually doesnt, so keep backtesting. A few comments on that:

  1. The buy algorithm should work independently from the sell algorithm. In other words, the sell point that the sell side algorithm     calculates should not be based on where the buy was triggered, but rather on its own calculation of a good sell point - often times a   completely different calculation than used on the buy side.
  2. Think risk management. A 80% win rate sounds good in theory but if the 20% of losses are not handled properly it could wipe you       out. Have good implementation of stop losses but give yourself reasonable space so you dont get stopped out too frequently.
  3. Think long term performance. Dont try to eliminate individual trades that were big losses. Often times, those trades were not         unreasonable trades when analyzed individually, they were simply bad outcomes. If longer term the strategy is sound, eliminating         individual losses would eliminate even more in gains.
  4. Think like a bot. Patience is not an issue and the fact that cryptocurrencies trade 24/7 is a huge advantage. The most profitable     strategies for me were the most boring ones a human could never do. 
  
  
  Experience: 11+ years trading stocks, derivatives, and now cryptocurrencies. I manage a small portfolio of several investors run 100% by algorithms. jp@6wingstech.com
