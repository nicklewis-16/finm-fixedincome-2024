import pandas as pd
import numpy as np
import datetime
import holidays

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.linear_model import LinearRegression

from scipy.optimize import minimize
from scipy import interpolate

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


def bday(date):
    """
    Check if a given date is a business day in the US.

    Parameters:
    date (datetime.date): The date to check.

    Returns:
    bool: True if the date is a business day, False otherwise.
    """
    us_bus = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return bool(len(pd.bdate_range(date, date, freq=us_bus)))

def prev_bday(date, force_prev=False):
    """
    Returns the previous business day given a date.

    Parameters:
    date (str or datetime.datetime): The input date in the format 'YYYY-MM-DD' or as a datetime object.
    force_prev (bool, optional): If True, forces the function to return the previous business day even if the input date is already a business day. Defaults to False.

    Returns:
    str or datetime.datetime: The previous business day as a string in the format 'YYYY-MM-DD' if the input date is a string, or as a datetime object if the input date is a datetime object.
    """
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date2str = True
    else:
        date2str = False
        
    if force_prev:
        date += -datetime.timedelta(days=1)
    while not bday(date):
        date += -datetime.timedelta(days=1)
    
    if date2str:
        date = date.strftime('%Y-%m-%d')
        
    return date

def get_coupon_dates(quote_date, maturity_date):
    """
    Returns a list of coupon dates between the quote date and maturity date.

    Parameters:
    quote_date (str or datetime.datetime): The quote date in the format 'YYYY-MM-DD' or as a datetime object.
    maturity_date (str or datetime.datetime): The maturity date in the format 'YYYY-MM-DD' or as a datetime object.

    Returns:
    list: A list of coupon dates between the quote date and maturity date.
    """
    if isinstance(quote_date, str):
        quote_date = datetime.datetime.strptime(quote_date, '%Y-%m-%d')
        
    if isinstance(maturity_date, str):
        maturity_date = datetime.datetime.strptime(maturity_date, '%Y-%m-%d')
    
    # divide by 180 just to be safe
    temp = pd.date_range(end=maturity_date, periods=np.ceil((maturity_date - quote_date).days / 180), freq=pd.DateOffset(months=6))
    # filter out if one date too many
    temp = pd.DataFrame(data=temp[temp > quote_date])

    out = temp[0]
    return out



def make_figure_number_issues_paying(CFmatrix):
    """
    Creates a figure showing the number of treasury issues with coupon or principal payment over time.

    Parameters:
    CFmatrix (numpy.ndarray): The cash flow matrix representing the treasury issues.

    Returns:
    None
    """
    mask_issues_paying = (CFmatrix!=0).sum()

    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(mask_issues_paying,marker='*',linestyle='None')

    #set ticks every quarter
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(2,5,8,11)))
    ax.xaxis.set_major_locator(mdates.YearLocator(month=2))

    #format ticks
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=60, horizontalalignment='right')

    ax.margins(x=0)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))

    plt.ylabel('number of treasury issues with coupon or principal payment')
    plt.title('Number of Treasuries Paying')

    plt.show()



def filter_treasuries(data, t_date=None, filter_maturity=None, filter_maturity_min=None, drop_duplicate_maturities=False, filter_tips=True, filter_yld=True):
    """
    Filter treasury data based on specified criteria.

    Parameters:
    - data: DataFrame, the treasury data to be filtered.
    - t_date: datetime, the date to filter the data on. If None, the latest date in the data will be used.
    - filter_maturity: int, the maximum maturity in years to filter the data on. Default is None.
    - filter_maturity_min: int, the minimum maturity in years to filter the data on. Default is None.
    - drop_duplicate_maturities: bool, whether to drop duplicate maturities. Default is False.
    - filter_tips: bool, whether to filter out TIPS (Treasury Inflation-Protected Securities). Default is True.
    - filter_yld: bool, whether to filter out securities with zero yield. Default is True.

    Returns:
    - outdata: DataFrame, the filtered treasury data.
    """
    outdata = data.copy()
    
    if t_date is None:
        t_date = outdata['CALDT'].values[-1]
    
    outdata = outdata[outdata['CALDT'] == t_date]
    
    # Filter out redundant maturity
    if drop_duplicate_maturities:
        outdata = outdata.drop_duplicates(subset=['TMATDT'])
    
    # Filter by max maturity
    if filter_maturity is not None:
        mask_truncate = outdata['TMATDT'] < (t_date + np.timedelta64(365 * filter_maturity + 1, 'D'))
        outdata = outdata[mask_truncate]

    # Filter by min maturity
    if filter_maturity_min is not None:
        mask_truncate = outdata['TMATDT'] > (t_date + np.timedelta64(365 * filter_maturity_min - 1, 'D'))
        outdata = outdata[mask_truncate]

    outdata = outdata[outdata['ITYPE'].isin([11, 12]) == (not filter_tips)]
        
    if filter_yld:
        outdata = outdata[outdata['TDYLD'] > 0]
        
    return outdata



def calc_cashflows(quote_data, filter_maturity_dates=False):
    """
    Calculate cashflows based on quote data.

    Args:
        quote_data (pd.DataFrame): DataFrame containing quote data.
        filter_maturity_dates (bool, optional): Flag to filter cashflows based on maturity dates. 
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing calculated cashflows.
    """
    CF = pd.DataFrame(data=0, index=quote_data.index, columns=quote_data['TMATDT'].unique())

    for i in quote_data.index:
        coupon_dates = get_coupon_dates(quote_data.loc[i,'CALDT'],quote_data.loc[i,'TMATDT'])

        if coupon_dates is not None:
            CF.loc[i,coupon_dates] = quote_data.loc[i,'TCOUPRT']/2

        CF.loc[i,quote_data.loc[i,'TMATDT']] += 100

    CF = CF.fillna(0).sort_index(axis=1)
    CF.drop(columns=CF.columns[(CF==0).all()],inplace=True)

    if filter_maturity_dates:
        CF = filter_treasury_cashflows(CF, filter_maturity_dates=True)
        
    return CF



def filter_treasury_cashflows(CF, filter_maturity_dates=False, filter_benchmark_dates=False, filter_CF_strict=True):
    """
    Filter treasury cashflows based on specified criteria.

    Parameters:
    CF (DataFrame): The cashflow data.
    filter_maturity_dates (bool): Flag indicating whether to filter by maturity dates. Default is False.
    filter_benchmark_dates (bool): Flag indicating whether to filter by benchmark dates. Default is False.
    filter_CF_strict (bool): Flag indicating whether to filter cashflows strictly. Default is True.

    Returns:
    DataFrame: The filtered cashflow data.
    """
    mask_benchmark_dts = []
    
    # Filter by using only benchmark treasury dates
    for col in CF.columns:
        if filter_benchmark_dates:
            if col.month in [2,5,8,11] and col.day == 15:
                mask_benchmark_dts.append(col)
        else:
            mask_benchmark_dts.append(col)
    
    if filter_maturity_dates:
        mask_maturity_dts = CF.columns[(CF>=100).any()]
    else:
        mask_maturity_dts = CF.columns
    
    mask = [i for i in mask_benchmark_dts if i in mask_maturity_dts]

    CF_filtered = CF[mask]
          
    if filter_CF_strict:
        # drop issues that had CF on excluded dates
        mask_bnds = CF_filtered.sum(axis=1) == CF.sum(axis=1)
        CF_filtered = CF_filtered[mask_bnds]

    else:
        # drop issues that have no CF on included dates
        mask_bnds = CF_filtered.sum(axis=1) > 0
        CF_filtered = CF_filtered[mask_bnds]
        
        
    # update to drop dates with no CF
    CF_filtered = CF_filtered.loc[:,(CF_filtered>0).any()]
    
    return CF_filtered



def get_maturity_delta(t_maturity, t_current):
    """
    Calculates the maturity delta in years between the given maturity date and the current date.

    Parameters:
    t_maturity (datetime): The maturity date.
    t_current (datetime): The current date.

    Returns:
    float: The maturity delta in years.
    """
    maturity_delta = (t_maturity - t_current) / pd.Timedelta('365.25 days')
    
    return maturity_delta



def discount_to_intrate(discount, maturity, n_compound=None):
    """
    Calculates the interest rate given the discount factor and maturity.

    Parameters:
    discount (float): The discount factor.
    maturity (float): The time to maturity in years.
    n_compound (int, optional): The number of times interest is compounded per year. 
                                If not provided, interest is continuously compounded.

    Returns:
    float: The interest rate.

    """
    if n_compound is None:
        intrate = - np.log(discount) / maturity
    
    else:
        intrate = n_compound * (1/discount**(1/(n_compound * maturity)) - 1)    
        
    return intrate




def intrate_to_discount(intrate, maturity, n_compound=None):
    """
    Calculates the discount factor given an interest rate and maturity.

    Parameters:
    intrate (float): The interest rate.
    maturity (float): The time to maturity in years.
    n_compound (int, optional): The number of times interest is compounded per year. 
                                If not provided, the discount factor is calculated using continuous compounding.

    Returns:
    float: The discount factor.
    """
    
    if n_compound is None:
        discount = np.exp(-intrate * maturity)
    else:
        discount = 1 / (1+(intrate / n_compound))**(n_compound * maturity)

    return discount



def compound_rate(intrate, compound_input, compound_output):
    """
    Calculates the compound rate based on the given interest rate and compounding periods.

    Parameters:
    intrate (float): The interest rate.
    compound_input (float): The number of compounding periods for the input rate.
    compound_output (float): The number of compounding periods for the output rate.

    Returns:
    float: The compound rate.

    """
    if compound_input is None:
        outrate = compound_output * (np.exp(intrate/compound_output) - 1)
    elif compound_output is None:
        outrate = compound_input * np.log(1 + intrate/compound_input)
    else:
        outrate = ((1 + intrate/compound_input) ** (compound_input/compound_output) - 1) * compound_output

    return outrate







def bootstrap(params, maturity):
    """
    Calculates the interpolated interest rate for a given maturity using the bootstrap method.

    Parameters:
    params (tuple): A tuple containing the estimated maturities and betas.
    maturity (float): The maturity for which the interest rate needs to be calculated.

    Returns:
    float: The interpolated interest rate for the given maturity.
    """
    estimated_maturities = params[0]
    betas = params[1]
    estimated_rates = discount_to_intrate(betas, estimated_maturities)
    
    f = interpolate.interp1d(estimated_maturities, estimated_rates, bounds_error=False, fill_value='extrapolate')
    
    rate = f(maturity)

    return rate



def nelson_siegel(params, maturity):
    """
    Calculates the Nelson-Siegel interest rate based on the given parameters and maturity.

    Parameters:
    params (list): A list of parameters [a, b, c, d] used in the Nelson-Siegel formula.
    maturity (float): The time to maturity in years.

    Returns:
    float: The calculated Nelson-Siegel interest rate.
    """
    rate = params[0] + (params[1] + params[2]) * (1 - np.exp(-maturity/params[3]))/(maturity/params[3]) - params[2] * np.exp(-maturity/params[3])
    
    return rate





def nelson_siegel_extended(params, maturity):
    """
    Calculates the Nelson-Siegel Extended rate for a given set of parameters and maturity.

    Parameters:
    params (list): A list of parameters [param1, param2, param3, param4, param5, param6].
    maturity (float): The time to maturity in years.

    Returns:
    rate (float): The calculated Nelson-Siegel Extended rate.
    """
    rate = params[0] + (params[1] + params[2]) * (1 - np.exp(-maturity/params[3]))/(maturity/params[3]) - params[2] * np.exp(-maturity/params[3]) + params[4] *((1-np.exp(-maturity/params[5]))/(maturity/params[5]) - np.exp(-maturity/params[5]))
    
    return rate




def estimate_curve_ols(CF, prices, interpolate=False):
    """
    Estimates the curve using ordinary least squares (OLS) regression.

    Parameters:
        CF (pd.DataFrame or pd.Series): Cash flows.
        prices (pd.DataFrame or pd.Series or np.ndarray): Prices.
        interpolate (bool, optional): Whether to interpolate the curve. Defaults to False.

    Returns:
        np.ndarray: Estimated curve discounts.
    """

    if isinstance(prices, pd.DataFrame) or isinstance(prices, pd.Series):
        prices = prices[CF.index].values

    mod = LinearRegression(fit_intercept=False).fit(CF.values, prices)

    if interpolate:
        matgrid = get_maturity_delta(CF.columns, CF.columns.min())

        dts_valid = np.logical_and(mod.coef_ < 1.25, mod.coef_ > 0)

        xold = matgrid[dts_valid]
        xnew = matgrid
        yold = mod.coef_[dts_valid]

        f = interpolate.interp1d(xold, yold, bounds_error=False, fill_value='extrapolate')
        discounts = f(xnew)

    else:
        discounts = mod.coef_

    return discounts




def price_with_rate_model(params, CF, t_current, fun_model, convert_to_discount=True, price_coupons=False):
    """
    Calculates the price of a fixed income security using a rate model.

    Parameters:
    params (list): List of parameters for the rate model.
    CF (numpy.ndarray): Cash flow of the fixed income security.
    t_current (float): Current time.
    fun_model (function): Function that models the interest rate.
    convert_to_discount (bool, optional): Flag to convert interest rates to discount factors. Defaults to True.
    price_coupons (bool, optional): Flag to include coupon payments in the price calculation. Defaults to False.

    Returns:
    numpy.ndarray: Price of the fixed income security.
    """

    maturity = get_maturity_delta(CF.columns, t_current)
    
    if convert_to_discount:
        disc = np.zeros(maturity.shape)
        for i, mat in enumerate(maturity):
            disc[i] = intrate_to_discount(fun_model(params,mat),mat)
    else:
        disc = fun(params,mat)
        
        
    if price_coupons:
        price = CF * disc
    else:
        price = CF @ disc
    
    return price




def pricing_errors(params, CF, t_current, fun_model, observed_prices):
    """
    Calculates the pricing errors between the observed prices and the modeled prices.

    Parameters:
    params (list): List of parameters for the rate model.
    CF (list): List of cash flows.
    t_current (float): Current time.
    fun_model (function): Function representing the rate model.
    observed_prices (array-like): Array-like object containing the observed prices.

    Returns:
    float: The sum of squared pricing errors.
    """
    price_modeled = price_with_rate_model(params, CF, t_current, fun_model)

    if isinstance(observed_prices, pd.DataFrame) or isinstance(observed_prices, pd.Series):
        observed_prices = observed_prices.values

    error = sum((observed_prices - price_modeled) ** 2)

    return error


def estimate_rate_curve(model, CF, t_current, prices, x0=None):
    """
    Estimates the rate curve parameters based on the given model.

    Parameters:
        model (str): The model used for estimation. Possible values are 'bootstrap', 'nelson_siegel', and 'nelson_siegel_extended'.
        CF (DataFrame): Cash flow matrix.
        t_current (float): Current time.
        prices (Series): Bond prices.
        x0 (array-like, optional): Initial guess for the optimization algorithm. Default is None.

    Returns:
        array-like: Optimized rate curve parameters.
    """
    if model is bootstrap:
        params = estimate_curve_ols(CF, prices, interpolate=False)
        
        CF_intervals = get_maturity_delta(CF.columns.to_series(), t_current=t_current).values
    
        params_optimized = [CF_intervals, params]

    else:
        if x0 is None:
            if model is nelson_siegel:
                x0 = np.ones(4) / 10
            elif model is nelson_siegel_extended:
                x0 = np.ones(6)
            else:
                x0 = 1        

        mod = minimize(pricing_errors, x0, args=(CF, t_current, model, prices))
        params_optimized = mod.x

    return params_optimized



def extract_spot_curves(quote_date, filepath=None, model=nelson_siegel, delta_maturity = .25, T=30,calc_forward=False, delta_forward_multiple = 1, filter_maturity_dates=False, filter_tips=True):
    """
    Extracts spot curves from treasury quotes data.

    Parameters:
    - quote_date (str): The date of the treasury quotes.
    - filepath (str, optional): The file path of the treasury quotes data. If not provided, a default path will be used.
    - model (function, optional): The model used to estimate the rate curve. Default is nelson_siegel.
    - delta_maturity (float, optional): The increment between maturities in the maturity grid. Default is 0.25.
    - T (int, optional): The maximum maturity in the maturity grid. Default is 30.
    - calc_forward (bool, optional): Whether to calculate forward rates. Default is False.
    - delta_forward_multiple (int, optional): The multiple of delta_maturity used to calculate the delta forward. Default is 1.
    - filter_maturity_dates (bool, optional): Whether to filter maturity dates. Default is False.
    - filter_tips (bool, optional): Whether to filter TIPS (Treasury Inflation-Protected Securities). Default is True.

    Returns:
    - curves (DataFrame): DataFrame containing spot rates, spot discounts, and optionally forward rates and forward discounts.
    """
    if filepath is None:
        filepath = f'../data/treasury_quotes_{quote_date}.xlsx'
        
    rawdata = pd.read_excel(filepath,sheet_name='quotes')
    
    rawdata.columns = rawdata.columns.str.upper()
    rawdata.sort_values('TMATDT',inplace=True)
    rawdata.set_index('KYTREASNO',inplace=True)

    t_check = rawdata['CALDT'].values[0]
    if rawdata['CALDT'].eq(t_check).all():
        t_current = t_check
    else:
        warnings.warn('Quotes are from multiple dates.')
        t_current = None

    rawprices = (rawdata['TDBID'] + rawdata['TDASK'])/2 + rawdata['TDACCINT']
    rawprices.name = 'price'

    ###
    data = filter_treasuries(rawdata, t_date=t_current, filter_tips=filter_tips)

    CF = filter_treasury_cashflows(calc_cashflows(data),filter_maturity_dates=filter_maturity_dates)
    prices = rawprices[CF.index]

    ###
    params = estimate_rate_curve(model,CF,t_current,prices)
    
    if model == nelson_siegel_extended:
        params0 = estimate_rate_curve(nelson_siegel,CF,t_current,prices)
        x0 = np.concatenate((params0,(1,1)))
        params = estimate_rate_curve(model,CF,t_current,prices,x0=x0)
        
    else:
        params = estimate_rate_curve(model,CF,t_current,prices)

    ###
    maturity_grid = np.arange(0,T+delta_maturity,delta_maturity)
    maturity_grid[0] = .01
    
    curves = pd.DataFrame(index = pd.Index(maturity_grid,name='maturity'))
    # adjust earliest maturity from 0 to epsion
    curves.columns.name = quote_date
    
    curves['spot rate']= model(params,maturity_grid)

    curves['spot discount'] = intrate_to_discount(curves['spot rate'].values,curves.index.values)
    
    
    
    if calc_forward:
        delta_forward = delta_forward_multiple * delta_maturity
        
        curves['forward discount'] = curves['spot discount'] / curves['spot discount'].shift(delta_forward_multiple)

        # first value of forward is spot rate
        maturity_init = curves.index[0:delta_forward_multiple]
        curves.loc[maturity_init,'forward discount'] = curves.loc[maturity_init,'spot discount']
        
        curves.insert(2,'forward rate', -np.log(curves['forward discount'])/delta_forward)
        
    return curves



def process_treasury_quotes(quote_date):
    """
    Processes treasury quotes data and returns relevant metrics.

    Parameters:
    - quote_date (str): The date of the treasury quotes.

    Returns:
    - metrics (DataFrame): DataFrame containing metrics such as issue date, maturity date, outstanding, coupon rate, yield, duration, maturity interval, and price.
    """
    
    filepath_rawdata = f'../data/treasury_quotes_{quote_date}.xlsx'
    rawdata = pd.read_excel(filepath_rawdata,sheet_name='quotes')
    rawdata.columns = rawdata.columns.str.upper()
    rawdata.sort_values('TMATDT',inplace=True)
    rawdata.set_index('KYTREASNO',inplace=True)

    t_check = rawdata['CALDT'].values[0]
    if rawdata['CALDT'].eq(t_check).all():
        t_current = t_check
    else:
        warnings.warn('Quotes are from multiple dates.')
        t_current = None

    rawprices = (rawdata['TDBID'] + rawdata['TDASK'])/2 + rawdata['TDACCINT']
    rawprices.name = 'price'

    maturity_delta = get_maturity_delta(rawdata['TMATDT'],t_current)
    maturity_delta.name = 'maturity delta'

    metrics = rawdata.copy()[['TDATDT','TMATDT','TDPUBOUT','TCOUPRT','TDYLD','TDDURATN']]
    metrics.columns = ['issue date','maturity date','outstanding','coupon rate','yld','duration']
    metrics['yld'] *= 365
    metrics['duration'] /= 365
    metrics['outstanding'] *= 1e6
    metrics['maturity interval'] = get_maturity_delta(metrics['maturity date'], t_current)
    metrics['price'] = rawprices
    
    return metrics


def get_bond(quote_date, maturity=None, coupon=None, selection='nearest'):
    """
    Retrieves bond metrics based on the specified criteria.

    Parameters:
    - quote_date (str): The date of the bond quote.
    - maturity (float or list): The maturity interval(s) of the bond(s) to retrieve. If a float is provided, it will be converted to a list.
    - coupon (float): The coupon rate of the bond(s) to retrieve.
    - selection (str): The method used to select bonds when multiple maturities are provided. Options are 'nearest', 'ceil', and 'floor'.

    Returns:
    - metrics (DataFrame): The bond metrics that match the specified criteria.
    """
    
    metrics = process_treasury_quotes(quote_date)

    if coupon is not None:
        metrics = metrics[metrics['coupon rate'] == coupon]
    
    if maturity is not None:
        mats = metrics['maturity interval']

        if type(maturity) is float:
            maturity = [maturity]

        idx = list()

        for m in maturity:
            if selection == 'nearest':
                idx.append(mats.sub(m).abs().idxmin())
            elif selection == 'ceil':
                idx.append(mats.sub(m).where(mats > 0, np.inf).argmin())
            elif selection == 'floor':
                idx.append(mats.sub(m).where(mats < 0, -np.inf).argmax())

        metrics = metrics.loc[idx, :]

    return metrics


def get_bond_raw(quote_date):
    """
    Retrieves raw bond data from an Excel file for a given quote date.

    Parameters:
    quote_date (str): The date of the bond quotes in the format 'YYYY-MM-DD'.

    Returns:
    rawdata (pd.DataFrame): The raw bond data as a pandas DataFrame.
    t_current (str or None): The date of the bond quotes if they are all from the same date, otherwise None.
    """
    
    filepath_rawdata = f'../data/treasury_quotes_{quote_date}.xlsx'
    rawdata = pd.read_excel(filepath_rawdata,sheet_name='quotes')
    rawdata.columns = rawdata.columns.str.upper()
    rawdata.sort_values('TMATDT',inplace=True)
    rawdata.set_index('KYTREASNO',inplace=True)

    t_check = rawdata['CALDT'].values[0]
    if rawdata['CALDT'].eq(t_check).all():
        t_current = t_check
    else:
        warnings.warn('Quotes are from multiple dates.')
        t_current = None
        
    return rawdata, t_current


def forward_discount(spot_discount, T1, T2):
    """
    Calculates the forward discount factor between two time periods.

    Parameters:
    spot_discount (pandas.Series): A pandas Series containing spot discount factors for different time periods.
    T1 (int): The starting time period.
    T2 (int): The ending time period.

    Returns:
    float: The forward discount factor between T1 and T2.
    """
    return spot_discount.loc[T2] / spot_discount.loc[T1]


def calc_npv(rate=0, cashflows=0, maturities=0, price=0):
    """
    Calculates the Net Present Value (NPV) of a series of cashflows.

    Parameters:
    rate (float): The discount rate used to calculate the NPV.
    cashflows (list): List of cashflows.
    maturities (list): List of maturities corresponding to each cashflow.
    price (float): The price of the investment.

    Returns:
    float: The calculated NPV.
    """
        
    temp = cashflows.copy()
    val = sum([cfi/(1+rate)**(maturities[i]) for i, cfi in enumerate(temp)])
    val += - price

    return val


def pv(rate, cashflows, maturities, freq=1):
    """
    Calculates the present value of a series of cashflows.

    Parameters:
    rate (float): The discount rate.
    cashflows (list): List of cashflows.
    maturities (list): List of maturities corresponding to each cashflow.
    freq (int, optional): Number of compounding periods per year. Default is 1.

    Returns:
    float: The present value of the cashflows.
    """
    price = sum([cfi / (1 + rate / freq) ** (maturities[i] * freq) for i, cfi in enumerate(cashflows)])
    return price


def next_business_day(DATE):
    """
    Calculates the next business day given a date.

    Args:
        DATE (datetime.date): The input date.

    Returns:
        datetime.date: The next business day.
    """
    ONE_DAY = datetime.timedelta(days=1)
    HOLIDAYS_US = holidays.US()

    next_day = DATE
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day += ONE_DAY
    return next_day
