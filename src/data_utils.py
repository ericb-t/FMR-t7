#!/usr/local/anaconda3/bin/python
#
#  Visualization of dataset and desrciptive statistic 
#
import sys
import os
import operator
import math
import pandas as pd 
import ipdb 
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import string

VERSION = 'v7'
MAX_SANE_RET = 2.0

# convert string YYYY-MM-DD to datetime object
def strdate2dtm(dt):
    y = int(dt[:4])
    m = int(dt[5:7])
    d = int(dt[8:])
    return datetime(y,m,d)

# return number of calendar dates between 2 dates, specified as strings YYYY-MM-DD
def get_ndays(dt1, dt2):
    'number of days between 2 dates '
    dt1x = strdate2dtm(dt1)
    dt2x = strdate2dtm(dt2)
    return (dt2x - dt1x).days

# number of days since last sale 
def ndays_since_last_sale(row):
    return get_ndays(row['Date_p'], row['Date'])

# convert SALE date MM-DD-YYYY ---> YYYY-MM-DD
def convert_date(row):
    return row['SALEDATE'][6:] + "-" + row['SALEDATE'][0:2] + "-"+row['SALEDATE'][3:5]

# convert PREV SALE date MM-DD-YYYY ---> YYYY-MM-DD
def convert_prev_date(row):
    return row['PREVSALEDATE'][6:] + "-" + row['PREVSALEDATE'][0:2] + "-"+row['PREVSALEDATE'][3:5]

# convert string to unique integer 
def convertToNumber (s):
    return int.from_bytes(s.encode(), 'big')

# create unique key for the property 
def property_key(row):
    p  = row['PROPERTYADDRESS']
    assert(len(p) > 0)

    for c in ['PROPERTYADDRESS', 'PROPERTYFRACTION', 'PROPERTYHOUSENUM', 'PROPERTYUNIT', 'PROPERTYZIP']:
        s = str(row[c])
        if len(s)>0: 
            p += "_" + s
            
    return p 
    
#
# organize and preprocess property assessment data and market baseline defined by IYR (liquid real estate ETF by Ishares) 
#
class PropertyAssessmentData:
    def __init__(self, dataset_filename, baseline):
        self.df = pd.read_csv(dataset_filename)

        interesting_features = ['PARID',
                                'PROPERTYADDRESS', 'PROPERTYHOUSENUM', 'PROPERTYUNIT','PROPERTYFRACTION',
                                'PROPERTYZIP', 
                                'SALEDATE',  'SALEPRICE', 'SALECODE',
                                'PREVSALEPRICE', 'PREVSALEDATE',
                                'CLASS', # R - RESIDENTIAL, U - UTILITIES, I - INDUSTRIAL, C - COMMERCIAL, O - OTHER, G - GOVERNMENT, F - AGRICULTURAL.
                                'COUNTYTOTAL',
                                'LOCALTOTAL',
                                'LOTAREA', 
                                'FAIRMARKETTOTAL',
                                'GRADE', # quality of construction
                                'CONDITION', 'CONDITIONDESC',
                                'CDU', 'CDUDESC']

        self.df = self.df[interesting_features]
        self.df = self.df[~(self.df.PREVSALEDATE.isnull())]
        
        # select records representing valid sales based on SALECODE
        valid_sale_codes = (self.df.SALECODE == '0') | (self.df.SALECODE == 'U') | (self.df.SALECODE == 'UR')
        self.df = self.df[valid_sale_codes]

        # focus on residential buildings
        self.df = self.df[self.df.CLASS == 'R']

        # select valid sale dates 
        self.df['Date'] = self.df.apply(convert_date, axis=1)
        self.df['Date_p'] = self.df.apply(convert_prev_date, axis=1)
        self.df['days_since_last_sell'] = self.df.apply(ndays_since_last_sale, axis=1)

        # return on each transaction (divided by number of days) 
        self.df['ret'] = (self.df['SALEPRICE'] - self.df['PREVSALEPRICE']) / (self.df['PREVSALEPRICE']*self.df['days_since_last_sell'])
        
        self.df = self.df[(self.df.Date >= '2006-01-01') & (self.df.Date <= '2020-10-30')]
        self.df = self.df.set_index(['Date']).sort_index()

        # unique property key
        self.df['PKEY'] = self.df.apply(property_key, axis=1)

        # baseline dataset 
        self.bl = pd.read_csv(baseline)
        self.bl = self.bl[self.bl.Date < '2020-10-30']
        self.bl = self.bl.set_index(['Date'])[['AdjClose']].sort_index().rename(columns={'AdjClose' : 'BASELINEPX'})

        bl_p = self.bl[['BASELINEPX']].shift(1).rename(columns={'BASELINEPX' : 'BASELINEPX_P'})
        self.bl = self.bl.join(bl_p)

        # daily baseline returns 
        self.bl['bret'] = (self.bl['BASELINEPX'] - self.bl['BASELINEPX_P']) / self.bl['BASELINEPX_P']

        # break point for debuging
        #ipdb.set_trace()
        return

    def find_number_of_properties_with_more_than_one_transaction(self):
        t = self.df[['PKEY','SALEPRICE']].groupby(['PKEY']).count().rename(columns={'SALEPRICE' : 'NTRANSACTIONS'})
        return t[t.NTRANSACTIONS > 1].size

    def price_range_histogram(self):
        ''' distribution of sale prices '''
        counts = {}
        dfx = self.df[['SALEPRICE']].dropna()
        print ('number of unfiltered records = ', dfx.size)
        
        #
        # as a sanity check we limit our analysis to properties which are worth at least 5K 
        #
        MIN_SALEPRICE = 5e3
        dfx = dfx[dfx.SALEPRICE > MIN_SALEPRICE]

        number_of_records = dfx.size
        print ('number_of_records = ', number_of_records)

        counts[1] = dfx[dfx.SALEPRICE <= 100e3].size / number_of_records
        counts[2] = dfx[(dfx.SALEPRICE > 100e3) & (dfx.SALEPRICE <=200e3)].size / number_of_records
        counts[3] = dfx[(dfx.SALEPRICE > 200e3) & (dfx.SALEPRICE <=300e3)].size / number_of_records
        counts[4] = dfx[(dfx.SALEPRICE > 300e3) & (dfx.SALEPRICE <=400e3)].size / number_of_records
        counts[5] = dfx[(dfx.SALEPRICE > 400e3) & (dfx.SALEPRICE <=500e3)].size / number_of_records
        counts[6] = dfx[(dfx.SALEPRICE > 500e3) & (dfx.SALEPRICE <=600e3)].size / number_of_records
        counts[7] = dfx[(dfx.SALEPRICE > 600e3) & (dfx.SALEPRICE <=700e3)].size / number_of_records
        counts[8] = dfx[(dfx.SALEPRICE > 700e3) & (dfx.SALEPRICE <=800e3)].size / number_of_records
        counts[9] = dfx[(dfx.SALEPRICE > 800e3) & (dfx.SALEPRICE <=900e3)].size / number_of_records
        counts[10] = dfx[(dfx.SALEPRICE > 900e3)].size / number_of_records

        return counts

    def calc_index(self, min_price, max_price, colname=None, colval=None):
        '''calculate average prices '''
        if colname == None:
            t = self.df[['SALEPRICE', 'PREVSALEPRICE', 'ret']]
        else:
            t = self.df[['SALEPRICE', 'PREVSALEPRICE', 'ret', colname]]
        t = t[(t.SALEPRICE >= min_price) & (t.SALEPRICE <= max_price) & (t.PREVSALEPRICE >= min_price) & (t.PREVSALEPRICE <= max_price) ]

        if colname != None:
            t = t.loc[t[colname] == colval].copy()
        
        t = t[np.abs(t.ret) < MAX_SANE_RET][['ret']].groupby(['Date']).mean()

        if colname == None:
            t = t.join(self.bl[['bret']])
        return t.cumsum()

    def number_of_sales_per_day(self):
        t = self.df[['SALEPRICE']].groupby('Date').count()
        return t.rename(columns={'SALEPRICE' : 'number_of_sales_per_day'})

    # calculate stat by group 
    def stat_by_group(self, colname, sdate, edate):
        t = self.df[['ret', 'SALEPRICE', 'LOTAREA', colname]].reset_index()
        t = t[(t.Date >= sdate) & (t.Date <= edate)]

        #t = t.loc[t[colname] == colvalue]
        #t = t[['ret', colname]]

        tr = t[np.abs(t.ret) < MAX_SANE_RET][['ret', colname]]
        t_out = tr.groupby(colname).mean()
        dispersion = tr.groupby(colname).std().rename(columns={'ret' : 'std'})
        counts = tr.groupby(colname).count().rename(columns={'ret' : 'count'})
        t_out = t_out.join(dispersion)
        t_out['IR'] = t_out['ret']/t_out['std']

        median_prices = t[np.abs(t.ret) < MAX_SANE_RET][['SALEPRICE', colname]].groupby(colname).median().rename(columns={'SALEPRICE': 'median_price'})
        average_prices = t[np.abs(t.ret) < MAX_SANE_RET][['SALEPRICE', colname]].groupby(colname).mean().rename(columns={'SALEPRICE': 'average_price'})

        standard_deviations_of_prices = t[np.abs(t.ret) < MAX_SANE_RET][['SALEPRICE', colname]].groupby(colname).std().rename(columns={'SALEPRICE': 'std_price'})
        
        t_out = t_out.join(counts).join(median_prices).join(standard_deviations_of_prices)

        # relative error in prices 
        t_out['price_rel_err'] = t_out['median_price'] / t_out['std_price']

        median_lotareas = t[np.abs(t.ret) < MAX_SANE_RET][['LOTAREA', colname]].groupby(colname).median().rename(columns={'LOTAREA': 'median_lotarea'})
        t_out = t_out.join(median_lotareas)
        
        return t_out
    
    # return different structure conditions
    def different_structure_conditions(self):
        return self.df.CDUDESC.unique()

    # descriptive statistic of lotareas
    def lotareas_stats(self):
        print (self.df[['LOTAREA']].describe())
        return 

    # baseline return
    def calc_baseline_return(self):
        bl = self.bl.reset_index()[['Date','BASELINEPX']]
        PX_START = bl[(bl.Date > '2014-01-01') & (bl.Date < '2015-12-31')]['BASELINEPX'].mean()
        PX_END = bl[(bl.Date > '2020-06-30')]['BASELINEPX'].mean()
        print ('EDBG ', PX_START, PX_END)
        return (PX_END - PX_START) / PX_START

    # price by lotarea and CDU
    def price_by_lotarea(self):
        #                    LOTAREA
        #count  6.645900e+04
        #mean   1.390668e+04
        #std    4.791695e+04
        #min    0.000000e+00
        #25%    3.700000e+03
        #50%    7.500000e+03
        #75%    1.306800e+04
        #max    4.225320e+06
        df = self.df[['SALEPRICE', 'LOTAREA', 'CDUDESC']]
        df.loc[df.LOTAREA < 3.7e3, 'LBIN'] = 'L1'
        df.loc[(df.LOTAREA >=3.7e3 ) & (df.LOTAREA < 7.5e3), 'LBIN'] = 'L2'
        df.loc[(df.LOTAREA >=7.5e3 ) & (df.LOTAREA < 1.3e4), 'LBIN'] = 'L3'
        df.loc[(df.LOTAREA >=1.3e4 ) & (df.LOTAREA < 4.22e6), 'LBIN'] = 'L4'
        df.loc[(df.LOTAREA >=4.22e6 ), 'LBIN'] = 'L5'

        print (df[['SALEPRICE', 'LBIN']].groupby('LBIN').mean())
        print ('=======================================')

        print (df[['SALEPRICE', 'CDUDESC', 'LBIN']].groupby(['CDUDESC', 'LBIN']).mean())
        print ('=======================================')

        
        return
        

    
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ('Usage: %s <filename> <request>  <optional arguments> ' % sys.argv[0])
        sys.exit(1)
    
    filename = 'data/assessments.csv'
    baseline = 'data/IYR.csv'
    req = sys.argv[1]

    show_plot = False 
    
    print ('dataset file: ', filename)
    print ('request:        ', req)

    pad = PropertyAssessmentData(filename, baseline)
    print ('number of properties with more than one transaction: ',
           pad.find_number_of_properties_with_more_than_one_transaction())

    print ('structure conditions: ', pad.different_structure_conditions())

    print ('baseline return: ', pad.calc_baseline_return())

    pad.lotareas_stats()

    pad.price_by_lotarea()
    
    ##################################################################
    if req == 'price_range_hist':
        price_dist = pad.price_range_histogram()
        #print ('price_dist = ', price_dist)
        out = open('price_ranges.csv', 'w')
        out.write('price_range_bin,count\n')
        out.flush()
        for j in price_dist:
            out.write('%d,%f\n' % (j, price_dist[j]))
            out.flush()
    elif req == 'calc_index':
        min_price = float(sys.argv[2])
        max_price = float(sys.argv[3])
        price_index_file = sys.argv[4]

        colname = None
        colval = None
        
        if sys.argv[5] != "None":
            colname = sys.argv[5]
            colval = sys.argv[6]

        price_index = pad.calc_index(min_price, max_price, colname, colval)
        
        if price_index_file != "None":
            price_index.to_csv(price_index_file)

        print (price_index)
        price_index.plot()
        plt.show()
        show_plot = True
    elif req == 'number_of_sales':
        ns = pad.number_of_sales_per_day()
        ns.plot()
        show_plot = True
    elif req == 'stat_by_group':
        colname = sys.argv[2]

        sdate = '2001-01-01'
        edate = '2014-01-01'
        stats_train = pad.stat_by_group(colname, sdate, edate)

        print ('========       Trained  =======  ', sdate, ' | ', edate)
        print (stats_train)
        stats_train.to_csv('train.csv')

        sdate = '2014-01-02'
        edate = '2015-12-31'
        stats_validate = pad.stat_by_group(colname, sdate, edate)
        print ('========   Validate ======== ', sdate, ' | ', edate)
        print (stats_validate)
        stats_validate.to_csv('validate.csv')

        sdate = '2020-07-01'
        edate = '2020-10-30'
        stats_invest = pad.stat_by_group(colname, sdate, edate)
        print ('========   Invest ======== ', sdate, ' | ', edate)
        print (stats_invest)
        stats_invest.to_csv('invest.csv')
        
    else:
        print ( "invalid request")
        
    if show_plot:
        plt.show()

    print ("(done)")
    sys.exit(0)
