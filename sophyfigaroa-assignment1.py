import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
    CS181 Assignment1: Fundamentals with Matplotlib & Plotly

    Name: SOPHY FIGAROA 

    Date: 09/02/25
"""

def load_covid_data ( filename : str ) -> pd . DataFrame:
    """_summary_
    Load COVID-19 data.

    Args:
        filename (str): Path to CSV file

    Returns:
        pd . DataFrame: Processed data
    """
    
    # Load COVID-19 dataset
    df = pd.read_csv(filename)

    return df
    
def clean_missing_values(df : pd.DataFrame) -> pd.DataFrame:
    """
    Cleans Nan values in a pandas DataFrame by replacing them with Nationwide

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing the dataset to clean. The DataFrame 
        contains some Nan values

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame with Nan values replaced by "Nationwide"
    """
    
    # Clean missing values
    # replacement string
    nan_replacement = "Nationwide" 
 
    # Fill in countries with only country wide value with "Nationwide"
    for col in df.columns:
        df[col] = df[col].fillna(nan_replacement)
        
    return df
    
       
def clean_negative_values(df : pd.DataFrame) -> pd.DataFrame:
    """
    Cleans negative values in a pandas DataFrame by replacing them with column median.

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing the dataset to clean. The DataFrame 
        may have numeric columns with negative values.

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame with negative values replaced by the median of 
        their respective columns for numeric columns.
    """
    
    # Select the rows and columns containing numbers, skip first row (colum names are nos considered a row) and first column (contains dates)
    df_numbers = df.iloc[1:, 1:]
    
    # cast objects to numbers for comparison when changing negative values as type is currently object
    for col in df_numbers.columns:
        df_numbers[col] = pd.to_numeric(df_numbers[col], errors='coerce')
        
    # loop over the numeric parts of the columns and replace negative value with median in column
    for col in  df_numbers:
        # compute median value of column
        median =  df_numbers[col].median()
        # if value is negative replace with median value
        df_numbers[col] = df_numbers[col].apply(lambda x: median if x < 0 else x)
        
    #add cleaned numbers back into clean df 
    df.iloc[1:, 1:] = df_numbers
    
    return df

def format_dates(df : pd.DataFrame) -> pd.DataFrame:
    """
    Format date columns into pandas datetime object

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing the COVID19 data, where first column are dates

    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the COVID19 data, where the dates are formatted as datetime object and dates are the index column
    """
    
    # Formatting Dates 
    try:
        # Assume the date comes second, skip first row as that contains province information
        df.loc[1:, "Date"] = pd.to_datetime(
            df.loc[1:, "Date"], format="%m/%d/%y", errors="coerce"
        )
    except ValueError:
        # Error Handeling: Handle mixed or unknown formats using dayfirst=True
        df.loc[1:, "Date"] = pd.to_datetime(
            df.loc[1:, "Date"], errors="coerce", dayfirst=True
        )
        
    # Set date as index column
    df.set_index('Date', inplace=True)
    
    return df 


def aggregate_country_data(df : pd.DataFrame):
    """
    Aggregates country data by grouping duplicate column names and summing up column values

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing COVID19 dataset where column names are countries.

    Returns:
    --------
    pd.DataFrame, pd.DataFram, Series
        df_country_daily_totals: A pandas DataFrame containing countries and their daily number of covid19 cases
        
        country_weekly_average: A pandas DataFrame containing countries and their covid cases per week
        
        country_total_cases: A pandas Series containing countries and their total covid19 cases
        
    """
    # Remove numbers trailing from duplicate column names 
    # CHATGPT was used to fix error of removing trailing numbers
    df.columns = df.columns.str.replace(r'\.\d+$', '', regex=True)
    
    # Aggregate data by country 
    # Skip the first and column row when grouping values, first row contains province info 
    df_skip_rc = df.iloc[1:,]
    # Create daily totals by country
    # This gives us a summary of cases for each country on each day, by summing up the row values of columns with the same name 
    df_country_daily_totals = df_skip_rc.groupby(df_skip_rc.columns, axis=1).sum()
     
    # Note: mean of 0s is NAN
    # Calculates weekly average by taking sections of 7 and calculting the mean
    country_weekly_average = df_country_daily_totals.rolling(window=7).mean()
    
    # Calculates total cases by summing up all the values in the columns, result is a Series
    country_total_cases = df_country_daily_totals.sum()
    
 
    return df_country_daily_totals, country_weekly_average, country_total_cases


def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataset of None, negative and duplicate values.

    Parameters:
    -----------
    df : pd.DataFrame
        A pandas DataFrame containing the dataset to clean. 
    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame with negative values replaced by the median of 
        their respective columns for numeric columns, nan values replaced by Nationwide, and duplicates erased. 
    """

    #create copy as to preserve original dataframe
    df_clean = df.copy()
    
    #Step 1: Rename Country/Region column to Date for easier processing
    df_clean.rename(columns={'Country/Region': 'Date'}, inplace=True)
    
    #Step 2: Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    #Step 3: Remove Nan values and replace with Nationwide
    df_clean = clean_missing_values(df_clean)
    
    #Step 4: Remove negative values and replace with column median
    df_clean = clean_negative_values(df_clean)
    
    #Step 5: format dates, Date is set as index column
    df_clean = format_dates(df_clean)
    
    return df_clean
  
def create_daily_cases_scatter(
    data : pd.DataFrame,
    countries : list[str],
    start_date : str,
    end_date : str
)-> plt.Figure:
    """_summary_
    
    Create scatter plot of daily COVID cases

    Args:
        data (pd.DataFrame): COVID data Country names as columns Cases as elements dates are first column
        countries (List[str]): List of countries
        start_date (str): Starting date
        end_date (str): Ending date

    Returns:
        plt.Figure: Matplotlib figure
    """
    
    #Error Handeling: If country not found in dataset, don't plot and inform that country was not found
    valid_countries = []
    for country in countries:
        if country not in data.columns:
            print(f"Country '{country}' not found in the dataset and will be skipped.")
        else:
            valid_countries.append(country)
            
    #Convert the start and end date strings to datetime objects, to index properly 
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Generate a unique color for each country using a colormap, assuming maximum of 10
    num_countries = len(valid_countries)
    cmap = plt.get_cmap('tab10', num_countries)
    
    # select the rows (label based indexing) from the specified start and end date from only countries in the list
    df_plot = data.loc[start_date_dt:end_date_dt, valid_countries]
    
    # Dates are the indeces of the data frame, making the x-as the dates
    x_dates = df_plot.index
    
    # iterate over countries list to extract cases and store in y-axis 
    for indx, country in enumerate(valid_countries): 
         y_cases = df_plot[country]
         # add alpha so that overlapping values are visible
         ax.scatter(x_dates, y_cases, label=country, color=cmap(indx), alpha=0.6)
         
    # disable scientific formatting for numbers for easier readability
    ax.ticklabel_format(style='plain', axis='y')
    # Add commas between the numbers for clarity ex: 12000000 -> 1,200,000
    #CHATGpt helped with figuring out the formatting syntax
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
    
    # display legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    # setting titles of graph and axes
    ax.set_title('Daily COVID-19 Cases')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Cases')
    
    #display plot
    plt.show()
    
    
def create_total_cases_bar(
    data : pd.Series,
    n_countries : int = 10
) -> plt.Figure:
    """_summary_
    
    Create bar chart of total cases by country

    Args:
        data : COVID data of total cases in country
        n_countries : Number of top countries to show

    Returns:
        plt.Figure: Matplotlib figure
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10)) 
    
    # Order data by top countries showing up first, thus order in descending order
    countries_descending = data.sort_values(ascending=False)
    
    # Y-axes: sample top n counttries from data 
    y_countries = countries_descending.iloc[:n_countries]
    
    # Horizontal Bar plot
    # y: the index of the series are the country names, y: and the values are the total cases
    bar_chart = ax.barh(y_countries.index, y_countries.values, color='red')
    # Invert y-axis so the highest value is at the top
    ax.invert_yaxis()
    
    # Set x-axis limit to prevent text overlap with grid lines
    #CHATGPT: assisted with figuring out how to move x grid line
    ax.set_xlim(0, y_countries.max() * 1.2)
    
    # Add a label to the end of each bar showing the total cases.
    # create an offset so that the label is not on top of the bar but next to it
    offset = 0.01 * max(y_countries.values)
    #loop over bars in barchart
    for bar in bar_chart:
        width = bar.get_width()
        ax.text(
            # x-position of tex 
            width + offset,
            # y-position: calculates to center int the middle
            bar.get_y() + bar.get_height() / 2,
            # formats case numbers to be 1,200,200
            # CHATGPT was used to help format the numbers properly
            f"{int(width):,}",
            va='center',
            ha='left',
            fontsize=12,
            color='black'
        )
        
    # Divides each tick value by 1e6 and formats it, so numbers are displayed in millions 
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M')
    )
    
    
    ax.set_title(f"TOP {n_countries} Countries COVID Cases")
    ax.set_xlabel('Cases')
    ax.set_ylabel('Countries')
    
    #display plot
    plt.tight_layout()
    plt.show()
    
      
def create_interactive_trends(
    data : pd.Series,
    countries: list[str]
) -> go.Figure:
    """_summary_

    Creative interactive time series plot 
    
    Args:
        data (pd.Series): COVID data
        countries (list[str]): Countries to include

    Returns:
        plt.Figure: Plotly Figure
    """
    fig = px.line(
        data, x = data.index, 
        y = countries , 
        labels={
            "x": "Date",
            "variable": "Country", #rename lable to country 
            "value": "COVID-19 Cases"}, 
        title="COVID-19 Trends by Country"
        )
    
    # Update the legend title to "Country"
    fig.update_layout(legend_title_text='Country')
    
    # Display plot
    fig.show()

def create_country_comparison(
    data : pd.Series,
    metric: str = "total_cases"
) -> go.Figure:
    """_summary_

    Creative interactive country comparison
    
    Args:
        data (pd.Series): COVID data
        metric (str): Comparison metric

    Returns:
        plt.Figure: Plotly Figure
    """
    
    # Order data by top countries showing up first
    countries_descending = data.sort_values(ascending=False)
    
    # Convert Series to DataFrame
    country_totals_df = countries_descending.reset_index()
    country_totals_df.columns = ["Country", "Total Cases"]

   
    # Create the bar chart
    fig = px.bar(
        country_totals_df,
        x="Country",  # Country names on the x-axis
        y="Total Cases",  # Total cases on the y-axis
        color="Total Cases",  # Gradient color based on Total Cases
        color_continuous_scale="Plasma",  # Choose a gradient scale
        title="Total COVID Cases by Country",
        labels={"Country": "Country", "Total Cases": "Total Cases"},
        hover_name="Country",  # Show the country name on hover
        hover_data={"Total Cases": ":,"}  # Format Total Cases with commas
    )

    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                   
                    dict(
                        label="All Countries",
                        method="update",
                        args=[{"x": [country_totals_df["Country"]],
                               "y": [country_totals_df["Total Cases"]]}]
                    ), 
                     dict(
                        label="Top 15 Countries",
                        method="update",
                        args=[{"x": [country_totals_df["Country"][:15]],
                               "y": [country_totals_df["Total Cases"][:15]]}]
                    ),
                     dict(
                        label="Top 10 Countries",
                        method="update",
                        args=[{"x": [country_totals_df["Country"][:10]],
                               "y": [country_totals_df["Total Cases"][:10]]}]
                    ),
                     dict(
                        label="Top 5 Countries",
                        method="update",
                        args=[{"x": [country_totals_df["Country"][:5]],
                               "y": [country_totals_df["Total Cases"][:5]]}]
                    ),
                ],
                direction="down",
                showactive=True,
            )
        ],
        xaxis_title="Country",
        yaxis_title="Total Cases",
        template="plotly",  
        coloraxis_showscale=True  # Show the gradient color scale
    )

    # Display Chart
    fig.show()
    
    
def main():
    """
    Plots Graphs
    """
    
    #Load + Clean Data
    clean = clean_data(load_covid_data("global_confirmed_cases.csv"))  
    #Aggregate Data
    df_country_daily_totals, country_weekly_average, country_total_cases = aggregate_country_data(clean) 

    #Matplotlib Scatter Plot
    create_daily_cases_scatter(df_country_daily_totals, ["US","Japan","Canada","United Kingdom"],"2022-01-14","2023-01-14")

    #Matplotlib Bar Chart
    create_total_cases_bar(country_total_cases)  
    
    #Plotly Line Plot
    create_interactive_trends(df_country_daily_totals, ["US","Japan","Kuwait","Sweden"]) 
    
    #Plotly Bar Chart
    create_country_comparison(country_total_cases) 
    
if __name__ == "__main__":
    main()