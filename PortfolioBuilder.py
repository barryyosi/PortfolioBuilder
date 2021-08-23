import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import date
import itertools
import math
import yfinance as yf
from typing import List


class PortfolioBuilder:
    def __init__(self):
        self.x_vectors = []
        self.x_vectorsFlag = False
        self.m = 1
        self.a = 1
        self.ticker_list = []
        self.df = 0
        self.b_tilda_vectors = []
        self.b_expo_tilda_vectors = []
        self.Omega = []
        self.OmegaFlag = False


    def get_daily_data(self, tickers_list: List[str], start_date: date, end_date: date = date.today()) -> pd.DataFrame:
        """
        get stock tickers adj_close price for specified dates.

        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame

        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """

        self.ticker_list = tickers_list
        self.df = web.get_data_yahoo(tickers_list, start_date, end_date)['Adj Close']
        if self.df.isnull().values.any():
            raise ValueError
        return self.df

    def x_vectors_calc(self):
        """This method calculates the x_vectors set, parm: none, return: none"""
        if self.x_vectorsFlag == False:
            self.x_vectors = []
            for date in range(1, len(self.df.index)):
                p0 = self.df.values[date - 1]
                p1 = self.df.values[date]
                self.x_vectors.append(list(p1 / p0))
                self.x_vectorsFlag = True
        else:
            pass

    def s_wealth(self, b_list: list,period):
        self.x_vectors_calc()
        S_T = np.prod([np.dot(b_list[i], self.x_vectors[i]) for i in range(period)])

        return S_T

    def compute_t_next(self,day):
        """

        :param day: used for computing  UPA for b_day+1
        :return: vector: b_t_next ("b_(t+1)") is the computed vector using UPA
        """
        if self.OmegaFlag == False:   ### Allows single computation of Omega set
            self.Omega = [b for b in itertools.product(np.arange(0, 1 + (1 / self.a) * 10 ** (-1), (1 / self.a)), repeat=len(self.ticker_list))
                     if 1 + 1 / self.a * 10 ** (-1) >= sum(b) >= 1 - 1 / self.a * 10 ** (-1)]
            self.OmegaFlag = True

        numinator_elements = (np.array(np.array(b_w) * self.s_wealth([b_w for i in range(day)], day)) for b_w in self.Omega)  ### list of computed (b_w * S_T(b_w)) vectors
        numinator_sum_vector = np.zeros(len(self.ticker_list))  ### initialization of the numinator summarized vector
        for vector in numinator_elements:
            numinator_sum_vector += vector

        denominator_sum_int = sum(list(self.s_wealth([b_w for i in range(day)],day) for b_w in self.Omega))

        b_t_next = numinator_sum_vector / denominator_sum_int
        self.b_tilda_vectors.insert(day,b_t_next)


    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day
        """
        self.a = portfolio_quantization
        self.b_tilda_vectors = []
        self.b_tilda_vectors.insert(0,tuple(1/len(self.ticker_list) for stock in self.ticker_list))
        attained_wealth = []
        for day in range(0,len(self.df.index)):
            self.compute_t_next(day)
            attained_wealth.append(self.s_wealth(self.b_tilda_vectors,day))
        return(attained_wealth)

    def compute_expo_t_next(self,period):
        b_t_next = []
        self.x_vectors_calc()
        denominator = sum([(self.b_expo_tilda_vectors[period-1][k]*np.exp((self.m * self.x_vectors[period-1][k])/np.dot(self.b_expo_tilda_vectors[period-1],self.x_vectors[period-1]))) for k in range(len(self.ticker_list))])

        for j in range(len(self.ticker_list)):
            nominator = self.b_expo_tilda_vectors[period-1][j]*np.exp((self.m*self.x_vectors[period-1][j])/np.dot(self.b_expo_tilda_vectors[period-1],self.x_vectors[period-1]))
            b_t_next.insert(j,nominator/denominator)

        self.b_expo_tilda_vectors.insert(period,b_t_next)

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks

        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: returns a list of floats, representing the growth trading  per day
        """
        self.m = learn_rate
        self.b_expo_tilda_vectors = []
        self.b_expo_tilda_vectors.insert(0,tuple(1/len(self.ticker_list) for stock in self.ticker_list))
        attained_expo_wealth = [1.0]
        for day in range(1, len(self.df.index)):
            self.compute_expo_t_next(day)
            attained_expo_wealth.insert(day, self.s_wealth(self.b_expo_tilda_vectors, day))
        return attained_expo_wealth





if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    import time
    t0 = time.time()
    barry = PortfolioBuilder()
    df = barry.get_daily_data(['GOOG','AAPL', 'MSFT'], date(2020, 1, 1), date(2020, 2, 1))
    print ("df length is", len(df))
    universal = barry.find_universal_portfolio(20)
    print(universal,'\n',"universal length is:",len(universal))
    expo = barry.find_exponential_gradient_portfolio()
    print(expo)
    t1 = time.time()
    print(t1-t0,"seconds to run the test")