##**Project 4, Group 3 References**


https://www.alphavantage.co/documentation/

https://stackoverflow.com/questions/36814100/pandas-to-numeric-for-multiple-columns

https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html

https://stockanalysis.com/pro/the-barbell-investor/

https://matplotlib.org/stable/gallery/color/named_colors.html

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

https://stackoverflow.com/questions/51921142/how-to-load-a-model-saved-in-joblib-file-from-google-cloud-storage-bucket

https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it

https://www.datacamp.com/tutorial/random-forests-classifier-python

https://keras.io/api/keras_tuner/hyperparameters/

https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html

https://seekingalpha.com/article/4679188-junk-bond-default-surge-continues-in-2024

https://pypi.org/project/imbalanced-learn/

https://www.youtube.com/watch?v=YUsx5ZNlYWc

https://www.youtube.com/watch?v=VtchVpoSdoQ&t=582s

# Current ratio = totalCurrentAssets / totalCurrentLiabilities
    # Reference: https://www.investopedia.com/terms/c/currentratio.asp
    
# Long-term Debt / Capital = longTermDebt / (longTermDebt + totalShareholderEquity)
    # Reference: https://www.investopedia.com/terms/l/longtermdebt-capitalization.asp *Preferred stock doesn't appear to be available on Alpha Vantage and isn't always issued
    
# Debt/Equity Ratio = totalLiabilities / totalShareholderEquity
    # Reference: https://www.investopedia.com/terms/d/debtequityratio.asp
    
# Gross Margin = 100 × (totalRevenue - costofGoodsAndServicesSold) / totalRevenue
    # Reference: https://www.investopedia.com/terms/g/grossmargin.asp, https://www.omnicalculator.com/finance/margin#gross-margin-formula
    
# Operating Margin = operatingIncome / totalRevenue
    # Reference: https://www.investopedia.com/terms/o/operatingmargin.asp
    
# EBIT Margin: ((totalRevenue - costofGoodsAndServicesSold - operatingExpenses) / totalRevenue) * 100
    # Reference: https://www.investopedia.com/terms/e/ebit.asp
    
# EBITDA Margin: (incomeBeforeTax + depreciationAndAmortization) / totalRevenue
    # Reference: https://www.investopedia.com/terms/e/ebitda-margin.asp
    
# Pre-Tax Profit Margin: (incomeBeforeTax / totalRevenue) * 100
    # Reference: https://www.investopedia.com/terms/p/pretax-margin.asp
    
# Net Profit Margin: (netIncome / totalRevenue) * 100
    # Reference: https://www.investopedia.com/terms/n/net_margin.asp
    
# Asset Turnover Ratio: totalRevenue / ((totalAssets + totalAssetsPrevious) / 2)
    # totalAssetsPrevious: totalAssets(from preYear) [1] = previous year
    
# ROE - Return On Equity: netIncome / ((totalShareholderEquity(curYear) + totalShareholderEquity(preYear)) / 2)
    # Reference: https://www.investopedia.com/terms/r/returnonequity.asp
    
# Return On Tangible Equity: netIncome / (avgShareholderEquity - intangibleAssets)
    # Reference: https://www.wallstreetprep.com/knowledge/return-on-tangible-equity-rote/
    
# ROA - Return On Assets: netIncome / totalAssets
    # Reference: https://www.investopedia.com/terms/r/returnonassets.asp
    
# ROI - Return On Investment: (netIncome / ((totalShareholderEquity(curYear) + totalShareholderEquity(preYear)) / 2)) * 100
    # Reference: https://www.wallstreetprep.com/knowledge/return-on-equity-roe/, https://www.investopedia.com/terms/r/returnoninvestment.asp
    
# Operating Cash Flow Per Share: operatingCashflow / commonStockSharesOutstanding
    # Reference: https://www.investopedia.com/terms/c/cashflowpershare.asp, https://www.wallstreetprep.com/knowledge/cash-flow-per-share/
    
# Free Cash Flow Per Share: (operatingCashflow - capitalExpenditures) / commonStockSharesOutstanding
    # Reference: https://www.investopedia.com/terms/f/freecashflowpershare.asp
    
