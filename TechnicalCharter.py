import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
from datetime import date
import requests
import math
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

MyPortfolio = ['MSFT','AAPL','GOOGL','AMZN','SQ','CBA.AX','TLS.AX','IOZ.AX','WBC.AX']
USStonks = ['MSFT','AAPL','GOOG','AMZN','SQ']
AUSStonks = ['CBA.AX','TLS.AX','IOZ.AX', 'WOW.AX', 'FMG.AX', 'MQG.AX', 'NCM.AX', 'RHC.AX', 'SEK.AX','MSFT','AAPL','GOOG','AMZN','SQ','SPY']

def FibonacciGrapher(CompanyCode, dates, homie, selldate, scost,selldate1, scost1): 
    CompanyCode = CompanyCode
    x = 0
    y = 0
    buyloop = 0
    w = 0
    differenceidentify = 0
    count = 125
    High = []
    Low = []
    Fib68 = []
    FibValue68 = []
    Fib50 = []
    FibValue50 = []
    Fib38 = []
    FibValue38 = []
    HighValue = []
    LowValue = []
    Extension = []
    ExtensionValue = []
    ExtensionValue2 = []
    FibValue382 = []
    HighValue2 = []
    LowValue2 = []
    LowValue3 = []
    LowValue4 = []
    LowValue5 = []
    LowValue6 = []
    HighValue5 = []
    HighValue6 = []
    HighValue3 = []
    HighValue4 = []
    FibValue502 = []
    FibValue682 = []
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    close_prices = stock['Close']
    df1 = pd.DataFrame(stock, columns=['Close'])
    periods = math.floor(days / 100)
    breh = 0
    maxposition = 0
    while x < periods:
        breh = close_prices[(days -1 - (x+1)*100):((days-1) - (x * 100))]
        maxposition = np.where(breh == breh.max())
        maxposition = maxposition[0][0]
        if maxposition > 94:
            differenceidentify = 100 - maxposition + 2
        else:
            differenceidentify = 0
        High.append(max(close_prices[(days - 1 - (x+1)*100):((days-1) - (x * 100) + differenceidentify)]))
        Low.append(min(close_prices[(days -1 - (x+1)*100):((days-1) - (x * 100))]))
        Fib68.append((High[x]-Low[x])*0.618 + Low[x])
        Fib50.append((High[x]-Low[x])*0.50 + Low[x])
        Fib38.append((High[x]-Low[x])*0.382 + Low[x])
        Extension.append(round((((float(High[x]) - float(Low[x]))*1.618)+float(Low[x])),2))
        x = x + 1
        differenceidentify = 0
    df1 = df1.dropna()
    print('###################################')
    print("The company is: ", CompanyCode)
    print("The last traded price is: ", round(df1['Close'][(days-1)],2))
    print("The extension level is: ",round(Extension[0],2))
    print("The last high was: ",round(High[0],2))
    print('The 68 Fibonacci lvl is: ',round(Fib68[0],2))
    print('The 50 Fibonacci lvl is: ',round(Fib50[0],2))
    print('The 32 Fibonacci lvl is: ',round(Fib38[0],2))
    print("The last low was: ",round(Low[0],2))
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.set_ylim((min(close_prices)*(0.95)), (max(close_prices)*(1.05)))
    ax.plot(df1['Close'], color='r')
    while y < (days-count):
        HighValue.append(None)
        LowValue.append(None)
        ExtensionValue.append(None)
        FibValue68.append(None)
        FibValue50.append(None)
        FibValue38.append(None)
        y = y + 1
    while w < count:
        HighValue.append(round(float(High[0]), 2))
        LowValue.append(round(float(Low[0]), 2))
        FibValue68.append(round(float(Fib68[0]), 2))
        FibValue50.append(round(float(Fib50[0]), 2))
        FibValue38.append(round(float(Fib38[0]), 2))
        ExtensionValue.append(round(float(Extension[0]), 2))
        w = w + 1
    y = 0
    w = 0
    while y < (days - (2*count)):
        HighValue2.append(None)
        LowValue2.append(None)
        ExtensionValue2.append(None)
        FibValue682.append(None)
        FibValue502.append(None)
        FibValue382.append(None)
        y = y + 1
    while w < count*2:
        HighValue2.append(round(float(High[1]), 2))
        LowValue2.append(round(float(Low[1]), 2))
        FibValue682.append(round(float(Fib68[1]), 2))
        FibValue502.append(round(float(Fib50[1]), 2))
        FibValue382.append(round(float(Fib38[1]), 2))
        ExtensionValue2.append(round(float(Extension[1]), 2))
        w = w + 1
    y = 0
    while y < (days):
        LowValue3.append(round(float(Low[2]), 2))
        LowValue4.append(round(float(Low[3]), 2))
        LowValue5.append(round(float(Low[4]), 2))
        LowValue6.append(round(float(Low[5]), 2))
        HighValue3.append(round(float(High[2]), 2))
        HighValue4.append(round(float(High[3]), 2))
        HighValue5.append(round(float(High[4]), 2))
        HighValue6.append(round(float(High[5]), 2))
        y = y + 1
    df1['Low3'] = LowValue3
    df1['Low4'] = LowValue4
    df1['High3'] = HighValue3
    df1['High4'] = HighValue4
    df1['Low5'] = LowValue5
    df1['Low6'] = LowValue6
    df1['High5'] = HighValue5
    df1['High6'] = HighValue6
    df1['HFib'] = HighValue
    df1['LFib'] = LowValue
    df1['Extension'] = ExtensionValue
    df1['Fib68'] = FibValue68
    df1['Fib50'] = FibValue50
    df1['Fib38'] = FibValue38
    df1['HFib2'] = HighValue2
    df1['LFib2'] = LowValue2
    df1['Extension2'] = ExtensionValue2
    df1['Fib682'] = FibValue682
    df1['Fib502'] = FibValue502
    df1['Fib382'] = FibValue382
    df1['MMA'] = df1.rolling(window=50).mean()['Close']
    df1['SMA'] = df1.rolling(window=20).mean()['Close']
    df1['LMA'] = df1.rolling(window=100).mean()['Close']
    ax.plot(df1['LMA'], color='black')
    ax.plot(df1['MMA'], color='purple')
    ax.plot(df1['SMA'], color='brown')
    ax.plot(df1['HFib'], color='g')
    ax.plot(df1['LFib'], color='b')
    ax.plot(df1['Extension'], color='black')
    ax.plot(df1['Fib68'], color='purple')
    ax.plot(df1['Fib50'], color='orange')
    ax.plot(df1['Fib38'], color='brown')
    ax.plot(df1['HFib2'], color='g')
    ax.plot(df1['High3'], color='g')
    ax.plot(df1['High4'], color='g')
    ax.plot(df1['High5'], color='g')
    ax.plot(df1['High6'], color='g')
    ax.plot(df1['LFib2'], color='b')
    ax.plot(df1['Low3'], color='b')
    ax.plot(df1['Low4'], color='b')
    ax.plot(df1['Low5'], color='b')
    ax.plot(df1['Low6'], color='b')
    ax.plot(df1['Extension2'], color='black')
    ax.plot(df1['Fib682'], color='purple')
    ax.plot(df1['Fib502'], color='orange')
    ax.plot(df1['Fib382'], color='brown')
    while buyloop < len(homie):
        plt.plot(dates[buyloop],homie[buyloop],marker = 'o', markersize = 10,color='green')
        buyloop = buyloop + 1
    sellloop = 0
    while sellloop <  len(scost):
        plt.plot(selldate[sellloop],scost[sellloop],marker = 'o', markersize = 10,color='yellow')
        sellloop = sellloop + 1
    sellloop = 0
    while sellloop <  len(scost1):
        plt.plot(selldate1[sellloop],scost1[sellloop],marker = 'o', markersize = 10,color='blue')
        sellloop = sellloop + 1
    plt.suptitle('Fibonacci Graph with Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Prices')
    plt.show()
    return

def MACD_BuySignal_graphed():
    CompanyCode = input("Enter the company code: ")
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2019,9,1), end=date.today())
    days = stock['Close'].count()
    timer = 0
    timetrack = []
    while timer < (days - 33):
        timetrack.append(0)
        timer = timer + 1
    df2 = pd.DataFrame(stock, columns = ['Close'])
    df2['26 EMA'] = df2.ewm(span = 26, min_periods = 26).mean()['Close']
    df2['12 EMA'] = df2.ewm(span = 12, min_periods = 12).mean()['Close']
    df2['MACD'] = df2['12 EMA'] - df2['26 EMA']
    df2['Signal Line'] = df2.ewm(span = 9, min_periods = 9).mean()['MACD']
    df2 = df2.dropna()
    df2['Zero Line'] = timetrack
    fig = plt.figure(figsize = (12, 7))
    ax = fig.add_subplot(111)
    ax.plot(df2['MACD'], color = 'red')
    ax.plot(df2['Signal Line'], color = 'green')
    ax.plot(df2['Zero Line'], color = 'blue')
    plt.legend(["MACD", "Signal Line"], loc ="upper left") 
    plt.suptitle('MACD Plotting')
    plt.xlabel('Date')
    plt.ylabel('MACD and Signal Values')
    plt.show()
    return

def MoneyFlowIndex():
    AbsTP = []
    x = 0
    y = 0
    z = 0
    w = 0
    PosRatio = 0
    NegRatio = 0
    Positive = []
    Negative = []
    MFR = []
    Equat = 0
    MFI = [50,50,50,50,50,50,50,50,50,50,50,50,50,50]
    SellRange = []
    seller = 0
    BuyRange = []
    buyer = 0
    CompanyCode = input("Enter the company code: ")
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2019,9,1), end=date.today())
    days = stock['Close'].count()
    while buyer < days:
        BuyRange.append(30)
        buyer = buyer + 1
    while seller < days:
        SellRange.append(75)
        seller = seller + 1
    df2 = pd.DataFrame(stock)
    df2['Typical Price'] = (df2['Close'] + df2['High'] + df2['Low'])/3
    AbsTP.append(df2['Typical Price'].iloc[x])
    while x < (days - 1):
        if df2['Typical Price'].iloc[(x+1)] > df2['Typical Price'].iloc[x]:
            AbsTP.append(df2['Typical Price'].iloc[(x+1)])
        else:
            AbsTP.append((df2['Typical Price'].iloc[(x+1)])*(-1))
        x = x + 1
    df2['Abs TP'] = AbsTP
    df2['Raw Money'] = df2['Abs TP'] * df2['Volume']
    while y < days:
        if df2['Raw Money'].iloc[y] > 0:
            Positive.append(df2['Raw Money'].iloc[y])
            Negative.append(0)
        else:
            Negative.append(df2['Raw Money'].iloc[y])
            Positive.append(0)
        y = y + 1
    while z < 14:
        PosRatio = PosRatio + Positive[z]
        NegRatio = NegRatio + Negative[z]
        z = z + 1
    while z < days:
        MFR.append((PosRatio/(-1*NegRatio)))
        PosRatio = PosRatio - Positive[(z - 14)] + Positive[z]
        NegRatio = NegRatio - Negative[(z - 14)] + Negative[z]
        z = z + 1
    while w < len(MFR):
        Equat = 100 - (100/(1+MFR[w]))
        MFI.append(Equat)
        w = w + 1
    df2['MFI'] = MFI
    df2['SELL'] = SellRange
    df2['BUYER'] = BuyRange
    fig = plt.figure(figsize = (12, 7))
    ax = fig.add_subplot(111)
    ax.set_title('Money Flow Index')
    ax.plot(df2['MFI'], color = 'blue')
    ax.plot(df2['BUYER'], color = 'green')
    ax.plot(df2['SELL'], color = 'red')
    plt.xlabel('Date')
    plt.ylabel('MFI')
    plt.show()
    return


#MACD_BuySignal_graphed()
#MoneyFlowIndex()

def Screener(codes):
    costbases = 0
    sharecount = 0
    counter = 0
    pscost = 0
    currentprices = 0
    currentcostbases = 0
    TYPValue = 0
    AbsTP = []
    x = 0
    y = 0
    z = 0
    w = 0
    PosRatio = 0
    NegRatio = 0
    Positive = []
    Negative = []
    MFR = []
    Equat = 0
    timer = 0
    timetrack = []
    MFI = [50,50,50,50,50,50,50,50,50,50,50,50,50,50]
    SellRange = []
    seller = 0
    BuyRange = []
    buyer = 0
    CompanyCode = codes
    if "." in CompanyCode:
        market = pdr.get_data_yahoo('^AXJO',start=datetime.datetime(2020,1,1), end=date.today())
    else:
        market = pdr.get_data_yahoo('^GSPC',start=datetime.datetime(2020,1,1), end=date.today())
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2020,1,1), end=date.today())
    days = stock['Close'].count()
    df2 = pd.DataFrame(stock)
    df1 = pd.DataFrame(market)
    df2.index = pd.to_datetime(df2.index)
    df2['LMA'] = df2.rolling(window=100).mean()['Close']
    df2['SMA'] = df2.rolling(window=30).mean()['Close']
    df2['Typical Price'] = (df2['Close'] + df2['High'] + df2['Low'])/3
    AbsTP.append(df2['Typical Price'].iloc[x])
    while timer < days:
        timetrack.append((timer+1))
        timer = timer + 1
    while x < (days - 1):
        if df2['Typical Price'].iloc[(x+1)] > df2['Typical Price'].iloc[x]:
            AbsTP.append(df2['Typical Price'].iloc[(x+1)])
        else:
            AbsTP.append((df2['Typical Price'].iloc[(x+1)])*(-1))
        x = x + 1
    df2['Abs TP'] = AbsTP
    df2['Raw Money'] = df2['Abs TP'] * df2['Volume']
    while y < days:
        if df2['Raw Money'].iloc[y] > 0:
            Positive.append(df2['Raw Money'].iloc[y])
            Negative.append(0)
        else:
            Negative.append(df2['Raw Money'].iloc[y])
            Positive.append(0)
        y = y + 1
    while z < 14:
        PosRatio = PosRatio + Positive[z]
        NegRatio = NegRatio + Negative[z]
        z = z + 1
    while z < days:
        MFR.append((PosRatio/(-1*NegRatio)))
        PosRatio = PosRatio - Positive[(z - 14)] + Positive[z]
        NegRatio = NegRatio - Negative[(z - 14)] + Negative[z]
        z = z + 1
    while w < len(MFR):
        Equat = 100 - (100/(1+MFR[w]))
        MFI.append(Equat)
        w = w + 1
    df2['MFI'] = MFI
    df2['Timer'] = timetrack
    df2['26 EMA'] = df2.ewm(span = 26, min_periods = 26).mean()['Close']
    df2['12 EMA'] = df2.ewm(span = 12, min_periods = 12).mean()['Close']
    df2['MACD'] = df2['12 EMA'] - df2['26 EMA']
    df2['Signal Line'] = df2.ewm(span = 9, min_periods = 9).mean()['MACD']
    xtra = 0
    trade_return = []
    maxValnear = []
    bigposition = []
    market_return = []
    largebuy = 0
    largebuycounter = 0
    small = 0
    smallcounter = 0
    dates = []
    homie = []
    bbb = 0
    bbbcounter = 0
    while counter < (days):
        TYPValue = round(float(df2['Close'][counter]),2) + TYPValue
        if round(float(df2['MFI'][counter]),2) < 30:
            if df2['MACD'][counter] < 0:
                if (df2['MACD'][counter-1]) < (df2['Signal Line'][counter-1]):
                    if abs((df2['MACD'][counter-1] - df2['Signal Line'][counter-1])) < abs((df2['MACD'][counter-2] - df2['Signal Line'][counter-2])):
                        if abs(df2['MACD'][counter-1]) > abs(df2['MACD'][counter-2]):
                            dates.append(df2.index.date[counter])
                            homie.append(round(float(df2['Close'][counter]),2))
                            if (counter + 1) == days:
                                costbases = costbases + round(float(df2['Close'][(counter)]),2)
                            else:
                                costbases = costbases + round(float(df2['Close'][(counter+1)]),2)
                            bbb = bbb + round(float(df2['SMA'][counter]),2)
                            bbbcounter = bbbcounter + 1
                            Valnear = df2['Close'][(counter):(counter+20)]
                            maxValnear.append(max(Valnear))
                            maxposition = np.where(Valnear == Valnear.max())
                            maxposition = maxposition[0][0]
                            bigposition.append(int(maxposition))
                            sharecount = sharecount + 1
                            trade_return.append(((maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2))*100)
                            market_return.append(((df1['Close'][counter+int(maxposition)] - round(float(df1['Close'][(counter)]),2))/round(float(df1['Close'][(counter)]),2))*100)
                            if df2['Close'][counter] < df2['LMA'][counter]*0.95:
                                largebuy = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2)*100 + largebuy
                                largebuycounter = largebuycounter + 1
                            else:
                                small = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2)*100 + small
                                smallcounter = smallcounter + 1
        counter = counter + 1
    wackamole = 0
    if round(float(df2['MFI'][(days - 1)]), 2) < 40:
        if df2['MACD'][days-1] < 0:
            print("################################")
            if "." in CompanyCode:
                print('Australian Market - '+ CompanyCode)
            else:
                print('US Market - '+ CompanyCode)
            currentprices = round(float(df2['Close'][(days-1)]),2)
            if sharecount > 0:
                pscost = costbases / sharecount
                wackamole = bbb/bbbcounter
            else:
                pscost = 0
                wackamole = 0
            print("Total number of shares bought:", sharecount)
            print("Total cost base of shares:", round(costbases,2))
            print("Per share cost base is:", round(pscost,2))
            print("The typical price for shares bought in a similar time are: ", round(wackamole, 2))
            print("################################")
            print("5th last MFI:", round(float(df2['MFI'][(days - 5)]), 2))
            print("4th last MFI:", round(float(df2['MFI'][(days - 4)]), 2))
            print("3rd last MFI:", round(float(df2['MFI'][(days - 3)]), 2))
            print("2nd last MFI:", round(float(df2['MFI'][(days - 2)]), 2))
            print("Most recent MFI:", round(float(df2['MFI'][(days - 1)]), 2))
            print("The current price is:",currentprices)
            print("################################")
            print("The MACD 3 days ago was:",df2['MACD'][days-3])
            print("The MACD 2 days ago was:",df2['MACD'][days-2])
            print("The MACD 1 day ago was:",df2['MACD'][days-1])
            print("The Signal Line 3 days ago was:",df2['Signal Line'][days-3])
            print("The Signal Line 2 days ago was:",df2['Signal Line'][days-2])
            print("The Signal Line 1 day ago was:",df2['Signal Line'][days-1])
            print("The difference 3 days ago was:", (df2['MACD'][days-3] - df2['Signal Line'][days-3]))
            print("The difference 2 days ago was:", (df2['MACD'][days-2] - df2['Signal Line'][days-2]))
            print("The difference 1 day ago was:", (df2['MACD'][days-1] - df2['Signal Line'][days-1]))
            print("The current 100 MA value is:", round(df2['LMA'][days-1],3))
            print("The current discount to the 100 MA value is:", round((((df2['Close'][days-1]-df2['LMA'][days-1])/df2['LMA'][days-1])*100),3),"%")
            return
    else:
        return

#for element in AUSStonks:
#    Screener(element)

def Test():
    costbases = 0
    sharecount = 0
    counter = 0
    pscost = 0
    currentprices = 0
    currentcostbases = 0
    TYPValue = 0
    AbsTP = []
    x = 0
    y = 0
    z = 0
    w = 0
    PosRatio = 0
    NegRatio = 0
    Positive = []
    Negative = []
    MFR = []
    Equat = 0
    timer = 0
    timetrack = []
    MFI = [50,50,50,50,50,50,50,50,50,50,50,50,50,50]
    SellRange = []
    seller = 0
    BuyRange = []
    buyer = 0
    CompanyCode = input("Enter the company code: ")
    print("################################")
    if "." in CompanyCode:
        market = pdr.get_data_yahoo('^AXJO',start=datetime.datetime(2018,1,1), end=date.today())
        print('Australian Market - '+ CompanyCode)
    else:
        market = pdr.get_data_yahoo('^GSPC',start=datetime.datetime(2018,1,1), end=date.today())
        print('US Market - '+ CompanyCode)
    stock = pdr.get_data_yahoo(CompanyCode,start=datetime.datetime(2018,1,1), end=date.today())
    days = stock['Close'].count()
    cuck = market['Close'].count()  
    while timer < days:
        timetrack.append((timer+1))
        timer = timer + 1
    while buyer < days:
        BuyRange.append(25)
        buyer = buyer + 1
    while seller < days:
        SellRange.append(80)
        seller = seller + 1
    df2 = pd.DataFrame(stock)
    df1 = pd.DataFrame(market)
    df2.index = pd.to_datetime(df2.index)
    df2['LMA'] = df2.rolling(window=100).mean()['Close']
    df2['SMA'] = df2.rolling(window=30).mean()['Close']
    df2['Typical Price'] = (df2['Close'] + df2['High'] + df2['Low'])/3
    AbsTP.append(df2['Typical Price'].iloc[x])
    while x < (days - 1):
        if df2['Typical Price'].iloc[(x+1)] > df2['Typical Price'].iloc[x]:
            AbsTP.append(df2['Typical Price'].iloc[(x+1)])
        else:
            AbsTP.append((df2['Typical Price'].iloc[(x+1)])*(-1))
        x = x + 1
    df2['Abs TP'] = AbsTP
    df2['Raw Money'] = df2['Abs TP'] * df2['Volume']
    while y < days:
        if df2['Raw Money'].iloc[y] > 0:
            Positive.append(df2['Raw Money'].iloc[y])
            Negative.append(0)
        else:
            Negative.append(df2['Raw Money'].iloc[y])
            Positive.append(0)
        y = y + 1
    while z < 14:
        PosRatio = PosRatio + Positive[z]
        NegRatio = NegRatio + Negative[z]
        z = z + 1
    while z < days:
        MFR.append((PosRatio/(-1*NegRatio)))
        PosRatio = PosRatio - Positive[(z - 14)] + Positive[z]
        NegRatio = NegRatio - Negative[(z - 14)] + Negative[z]
        z = z + 1
    while w < len(MFR):
        Equat = 100 - (100/(1+MFR[w]))
        MFI.append(Equat)
        w = w + 1
    df2['MFI'] = MFI
    df2['SELL'] = SellRange
    df2['BUYER'] = BuyRange
    df2['Timer'] = timetrack
    df2['26 EMA'] = df2.ewm(span = 26, min_periods = 26).mean()['Close']
    df2['12 EMA'] = df2.ewm(span = 12, min_periods = 12).mean()['Close']
    df2['MACD'] = df2['12 EMA'] - df2['26 EMA']
    df2['Signal Line'] = df2.ewm(span = 9, min_periods = 9).mean()['MACD']
    xtra = 0
    trade_return = []
    maxValnear = []
    bigposition = []
    market_return = []
    largebuy = 0
    largebuycounter = 0
    small = 0
    smallcounter = 0
    dates = []
    homie = []
    bbb = 0
    bbbcounter = 0
    while counter < (days):
        TYPValue = round(float(df2['Close'][counter]),2) + TYPValue
        if round(float(df2['MFI'][counter]),2) < 30:
            if df2['MACD'][counter] < 0:
                if (df2['MACD'][counter-1]) < (df2['Signal Line'][counter-1]):
                    if abs((df2['MACD'][counter-1] - df2['Signal Line'][counter-1])) < abs((df2['MACD'][counter-2] - df2['Signal Line'][counter-2])):
                        if abs(df2['MACD'][counter-1]) > abs(df2['MACD'][counter-2]):
                            dates.append(df2.index.date[counter])
                            homie.append(round(float(df2['Close'][counter]),2))
                            if (counter + 1) == days:
                                costbases = costbases + round(float(df2['Close'][(counter)]),2)
                            else:
                                costbases = costbases + round(float(df2['Close'][(counter+1)]),2)
                            bbb = bbb + round(float(df2['SMA'][counter]),2)
                            bbbcounter = bbbcounter + 1
                            Valnear = df2['Close'][(counter):(counter+20)]
                            maxValnear.append(max(Valnear))
                            maxposition = np.where(Valnear == Valnear.max())
                            maxposition = maxposition[0][0]
                            bigposition.append(int(maxposition))
                            sharecount = sharecount + 1
                            trade_return.append(((maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2))*100)
                            market_return.append(((df1['Close'][counter+int(maxposition)] - round(float(df1['Close'][(counter)]),2))/round(float(df1['Close'][(counter)]),2))*100)
                            if df2['Close'][counter] < df2['LMA'][counter]*0.95:
                                largebuy = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2)*100 + largebuy
                                largebuycounter = largebuycounter + 1
                            else:
                                small = (maxValnear[sharecount-1] - round(float(df2['Close'][(counter)]),2))/round(float(df2['Close'][(counter)]),2)*100 + small
                                smallcounter = smallcounter + 1
        counter = counter + 1
    TYPValue = TYPValue / (days)
    print("The last buy date is: ", dates[len(dates)-1])
    sellcounter = 0
    selldate = []
    sprice = 0
    scost = []
    discount = 0
    scount = 0
    while sellcounter < (days - 1):
        if round(float(df2['MFI'][sellcounter]),2) > 73:
            if df2['Signal Line'][sellcounter] > 0:
                if (df2['MACD'][sellcounter-1]) > (df2['Signal Line'][sellcounter-1]):
                    if abs((df2['MACD'][sellcounter-1] - df2['Signal Line'][sellcounter-1])) < abs((df2['MACD'][sellcounter-2] - df2['Signal Line'][sellcounter-2])):
                        selldate.append(df2.index.date[sellcounter])
                        scost.append(round(float(df2['Close'][sellcounter]),2))
                        sprice = sprice + round(float(df2['Close'][(sellcounter + 1)]),2)
                        scount = scount + 1
        sellcounter = sellcounter + 1
    sellcounter = 0
    selldate1 = []
    sprice1 = 0
    scost1 = []
    discount1 = 0
    scount1 = 0
    while sellcounter < (days - 1):
        if round(float(df2['MFI'][sellcounter]),2) > 73 or round(float(df2['MFI'][sellcounter-1]),2) > 73:
            if df2['Signal Line'][sellcounter] > 0:
                if (df2['MACD'][sellcounter-1]) > (df2['Signal Line'][sellcounter-1]):
                    if df2['MACD'][sellcounter] < df2['Signal Line'][sellcounter]:
                        selldate1.append(df2.index.date[sellcounter])
                        scost1.append(round(float(df2['Close'][sellcounter]),2))
                        sprice1 = sprice1 + round(float(df2['Close'][(sellcounter + 1)]),2)
                        scount1 = scount1 + 1
        sellcounter = sellcounter + 1
    if sharecount == 0:
        currentprices = round(float(df2['Close'][(days-1)]),2)
        print("No Shares.")
        print("The typical share price during the period was:", round(TYPValue,2))
        print("################################")
        print("5th last MFI:", round(float(df2['MFI'][(days - 5)]), 2))
        print("4th last MFI:", round(float(df2['MFI'][(days - 4)]), 2))
        print("3rd last MFI:", round(float(df2['MFI'][(days - 3)]), 2))
        print("2nd last MFI:", round(float(df2['MFI'][(days - 2)]), 2))
        print("Most recent MFI:", round(float(df2['MFI'][(days - 1)]), 2))
        print("The current price is:",currentprices)
        print("################################")
        print("The MACD 3 days ago was:",df2['MACD'][days-3])
        print("The MACD 2 days ago was:",df2['MACD'][days-2])
        print("The MACD 1 day ago was:",df2['MACD'][days-1])
        print("The Signal Line 3 days ago was:",df2['Signal Line'][days-3])
        print("The Signal Line 2 days ago was:",df2['Signal Line'][days-2])
        print("The Signal Line 1 day ago was:",df2['Signal Line'][days-1])
        print("The difference 3 days ago was:", (df2['MACD'][days-3] - df2['Signal Line'][days-3]))
        print("The difference 2 days ago was:", (df2['MACD'][days-2] - df2['Signal Line'][days-2]))
        print("The difference 1 day ago was:", (df2['MACD'][days-1] - df2['Signal Line'][days-1]))
        print("The current 100 MA value is:", df2['LMA'][days-1])
        print("The current discount to the 100 MA value is:", (df2['Close'][days-1]-df2['LMA'][days-1])/df2['LMA'][days-1])
        FibonacciGrapher(CompanyCode, dates, homie, selldate, scost, selldate1, scost1)
        return
    
    currentprices = round(float(df2['Close'][(days-1)]),2)
    pscost = costbases / sharecount
    if largebuycounter > 0:
        print("The close return when bought with at least a 5% discount was: ", round(largebuy / largebuycounter,2))
    if smallcounter > 0:
        print("The close return for every other purchase: ", round(small / smallcounter,2))
    print("Total number of shares bought:", sharecount)
    print("Total cost base of shares:", round(costbases,2))
    print("Per share cost base is:", round(pscost,2))
    print("The typical price for shares bought in a similar time are: ", round(bbb/bbbcounter, 2))
    print("################################")
    print("5th last MFI:", round(float(df2['MFI'][(days - 5)]), 2))
    print("4th last MFI:", round(float(df2['MFI'][(days - 4)]), 2))
    print("3rd last MFI:", round(float(df2['MFI'][(days - 3)]), 2))
    print("2nd last MFI:", round(float(df2['MFI'][(days - 2)]), 2))
    print("Most recent MFI:", round(float(df2['MFI'][(days - 1)]), 2))
    print("The current price is:",currentprices)
    print("################################")
    print("The MACD 3 days ago was:",round(df2['MACD'][days-3],3))
    print("The MACD 2 days ago was:",round(df2['MACD'][days-2],3))
    print("The MACD 1 day ago was:",round(df2['MACD'][days-1],3))
    print("The Signal Line 3 days ago was:",round(df2['Signal Line'][days-3],3))
    print("The Signal Line 2 days ago was:",round(df2['Signal Line'][days-2],3))
    print("The Signal Line 1 day ago was:",round(df2['Signal Line'][days-1],3))
    print("The difference 3 days ago was:", round((df2['MACD'][days-3] - df2['Signal Line'][days-3]),3))
    print("The difference 2 days ago was:", round((df2['MACD'][days-2] - df2['Signal Line'][days-2]),3))
    print("The difference 1 day ago was:", round((df2['MACD'][days-1] - df2['Signal Line'][days-1]),3))
    print("The current 100 MA value is:", round(df2['LMA'][days-1],3))
    print("The current discount to the 100 MA value is:", round(((df2['Close'][days-1]-df2['LMA'][days-1])/df2['LMA'][days-1])*100,3),'%')
    FibonacciGrapher(CompanyCode, dates, homie, selldate, scost, selldate1, scost1)
    return

Test()
