# FMR-t7

Usage: 

 1) generate histogram of sale prices per value bins:
                      data_utils.py price_range_hist 

 2)  generate timeseries of number of transactions 
                      data_utils.py number_of_sales

 3)  calculate price index
                      data_utils.py calc_index 50e3 500e3 price_index.csv None None 

 4)  calculate price index for given group determined by column/feature and selected value
                      data_utils.py calc_index 10e3 500e3 rets_cdu_fair.csv CDUDESC FAIR 

 5)  generate statistic by group for different intervals 
                      data_utils.py stat_by_group CDUDESC 


