import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import pandas as pd
import joblib
from snowflake.snowpark import Session
from sklearn.metrics import accuracy_score, classification_report
import re
import plotly.graph_objects as go
import tempfile
import os
import io
import csv
import matplotlib.pyplot as plt

# Function to create a Snowflake session using secrets
def get_snowflake_session():
    try:
        # Set up Snowflake connection using Streamlit secrets
        connection_parameters = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "password": st.secrets["snowflake"]["password"],
            "role": st.secrets["snowflake"]["role"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
            "database": st.secrets["snowflake"]["database"],
            "schema": st.secrets["snowflake"]["schema"]
        }

        # Create and return the Snowflake session
        return Session.builder.configs(connection_parameters).create()
    except Exception as e:
        st.error(f"Error getting Snowflake session: {e}")
        return None

# Snowflake stage
SNOWFLAKE_STAGE = '"STREAMLIT_DEMO"."STREAMLIT_SCHEMA"."STREAMLIT_STAGE"'

# Load the pre-trained model and data from Snowflake
@st.cache_resource
def load_model_and_data():
    session = get_snowflake_session()
    if not session:
        st.error("Failed to create Snowflake session.")
        return None, None, None, None, None, None, None

    try:
        # List of files to load
        files_to_load = [
            "yuge75_experimental2.joblib.gz",
            "yuge75_validfull.joblib.gz",
            "yuge75_xvalid.joblib.gz",
            "yuge75_yvalid.joblib.gz",
            "yuge75_xoob.joblib.gz",
            "yuge75_yoob.joblib.gz",
            "yuge75_oobfull.joblib.gz"
        ]

        loaded_data = {}
        with tempfile.TemporaryDirectory() as tmpdirname:
            for file in files_to_load:
                file_path = f'@{SNOWFLAKE_STAGE}/{file}'
                local_file_path = os.path.join(tmpdirname, file)

                # Download file from Snowflake stage
                session.file.get(file_path, tmpdirname)  # Download to the temporary directory

                # Correct file path after download
                local_file_path = os.path.join(tmpdirname, os.path.basename(file))

                # Load the file using joblib
                loaded_data[file] = joblib.load(local_file_path)

        st.success("Model and data loaded successfully.")
        return (
            loaded_data["yuge75_experimental2.joblib.gz"],
            loaded_data["yuge75_validfull.joblib.gz"],
            loaded_data["yuge75_xvalid.joblib.gz"],
            loaded_data["yuge75_yvalid.joblib.gz"],
            loaded_data["yuge75_oobfull.joblib.gz"],
            loaded_data["yuge75_xoob.joblib.gz"],
            loaded_data["yuge75_yoob.joblib.gz"]
        )
    except Exception as e:
        st.error(f"Error loading model and data from Snowflake: {e}")
        return None, None, None, None, None, None, None

def does_not_end_with_number(column_name):
    return not re.search(r'\d$', column_name)

def split_dataset(df):
    tickerlist = []
    count = sum(does_not_end_with_number(col) for col in df.columns)
    features = count
    num_features_per_ticker = features

    for i in range(int(len(df.columns) / features)):
        start = i * num_features_per_ticker
        end = start + num_features_per_ticker

        ticker = df.iloc[:, start:end]
        if i == 0:
            colnames = ticker.columns.tolist()
        ticker.columns = colnames
        tickerlist.append(ticker)

    return tickerlist

def calculate_technical_indicators(df):
    df['VOLUME'] = df['TURNOVER'] / df['PX_LAST']
    df['PCT_CHANGE_20'] = ((df['PX_LAST'].shift(-20) - df['PX_LAST']) / df['PX_LAST']) * 100

    for short, long in [(10, 20), (20, 40), (40, 80), (80, 120)]:
        df[f'VOL_RATIO_{short}_{long}'] = df['VOLUME'].rolling(window=short).mean() / df['VOLUME'].rolling(window=long).mean()

    df['PE_Ratio'] = 1 / df['EARN_YLD']
    win = 60
    for percentile in [25, 50, 75]:
        df[f'PE_Band_{percentile}'] = df['PE_Ratio'].rolling(win).quantile(percentile / 100)

    ema_12 = df['PX_LAST'].ewm(span=12, adjust=False).mean()
    ema_26 = df['PX_LAST'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    lags = [10, 20, 30, 60, 120]
    for lag in lags:
        df[f'MOMENTUM_{lag}'] = df['PX_LAST'] / df['PX_LAST'].shift(lag)
        df[f'TURNOVER_{lag}'] = df['TURNOVER'].rolling(window=lag).mean()
        df[f'PX_MOMENTUM_{lag}'] = df['PX_LAST'] / df['PX_LAST'].shift(lag)
        df[f'PX_REVERSAL_{lag}'] = df['PX_LAST'].shift(lag) / df['PX_LAST']
        df[f'VOLATILITY_{lag}'] = df['PX_LAST'].rolling(window=lag).std()
        df[f'VOLUME_STD_{lag}'] = df['VOLUME'].rolling(window=lag).std()

    return df

def calculate_validation_stats(model, valid_full, x_valid, y_valid):
    if model is None:
        st.error("Model is not loaded correctly. Please check your Snowflake connection and model file.")
        return None, None

    predicted = model.predict(x_valid)
    predicted_probs = model.predict_proba(x_valid)

    valid_data = pd.DataFrame({
        'Ticker': valid_full['Ticker'],
        'Predicted': predicted,
        'Confidence': predicted_probs[:, 1],
        'Returns': valid_full['PCT_CHANGE_20'],
        'Buy Price': valid_full['PX_LAST'],
    })

    positive_preds = valid_data[valid_data["Predicted"] == 1]

    bins = range(50, 105, 5)
    labels = [f'{i}-{i+5}' for i in bins[:-1]]
    positive_preds['Probbin'] = pd.cut(positive_preds['Confidence'] * 100, bins=bins, labels=labels, right=False)

    stats_df = positive_preds.groupby('Probbin')['Returns'].agg(
        mean='mean',
        top_10=lambda x: x.quantile(0.9),
        bottom_10=lambda x: x.quantile(0.1),
        std='std',
        count='size',
        winrate=lambda x: (x > 0).mean() * 100
    ).reset_index()

    # Initialize the plot
    fig = go.Figure()

    # Plot the PDF for each bin
    for label in labels:
        subset = positive_preds[positive_preds['Probbin'] == label]
        if not subset.empty:
            kde = subset['Returns'].plot.kde()
            x = kde.get_lines()[0].get_xdata()
            y = kde.get_lines()[0].get_ydata()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', name=f'Prob Bin: {label}'))
            kde.remove()

    # Customize the layout
    fig.update_layout(
        title='PDF of Returns by Probability Bin',
        xaxis_title='Return',
        yaxis_title='Density',
        template='plotly_dark',
        legend_title_text='Probability Bins'
    )

    # Embed the plot in Streamlit
    st.plotly_chart(fig)

    return valid_data, stats_df

def prepare_data(uploaded_file):
    file_content = uploaded_file.read()

    # Automatically detect the delimiter
    try:
        dialect = csv.Sniffer().sniff(file_content[:1024].decode('utf-8'))
        delimiter = dialect.delimiter
    except csv.Error:
        # Default to comma if delimiter detection fails
        delimiter = ','

    # Load the data with the detected or default delimiter
    evaluation_data = pd.read_csv(io.BytesIO(file_content), delimiter=delimiter, skiprows=5, index_col='Dates')
    evaluation_data.index = pd.to_datetime(evaluation_data.index)

    evlist = split_dataset(evaluation_data)
    colnames = evlist[0].columns.tolist()
    for ev in evlist:
        ev.columns = colnames

    # Re-read the file to obtain ticker names using the detected delimiter
    tickernames = pd.read_csv(io.BytesIO(file_content), delimiter=delimiter).loc[2, :].dropna().tolist()
    for cov in evlist:
        calculate_technical_indicators(cov)

    evlist2b = [cov.tail(20) for cov in evlist]
    for i, df in enumerate(evlist2b):
        df.loc[:, 'Ticker'] = tickernames[i]

    pred_data = pd.concat(evlist2b, axis=0)
    return pred_data

def run_prediction(model, pred_data):
    pred_nopct = pred_data.drop('PCT_CHANGE_20', axis=1)
    pred_nona = pred_nopct.dropna(axis=0)

    predictions = model.predict(pred_nona.drop(['Ticker'], axis=1))
    prediction_probs = model.predict_proba(pred_nona.drop(['Ticker'], axis=1))

    preds = pd.DataFrame({
        'Predicted': predictions,
        'Confidence': prediction_probs[:, 1],
    })
    preds.index = pred_nona.index

    pred_nona['Predicted'] = preds['Predicted']
    pred_nona['Confidence'] = preds['Confidence']
    return pred_nona.loc[:, ['Ticker', 'Predicted', 'Confidence']]

def main():
    st.title("G Money app")

    model, valid_full, x_valid, y_valid, oob_full, x_oob, y_oob = load_model_and_data()

    if model is None:
        st.error("Model could not be loaded. Please check the Snowflake connection and model file.")
        return

    uploaded_file = st.file_uploader("Upload your CSV file for inference", type=["csv"])

    if uploaded_file is not None:
        pred_data = prepare_data(uploaded_file)
        predictions = run_prediction(model, pred_data)

        st.title("Predictions")
        positive_preds = predictions[predictions['Predicted'] == 1]
        positive_preds.index = positive_preds.index.strftime('%d %B %Y')
        st.write(positive_preds.drop_duplicates(subset=['Confidence', 'Ticker']))

    # Create columns for Out of Distribution and In Distribution Statistics
    col1, col2 = st.columns(2)

    # Out of Distribution Validation Statistics
    st.title("OOD Validation Statistics")
    valid_data, stats_df = calculate_validation_stats(model, oob_full, x_oob, y_oob)
    if valid_data is not None:
        st.write("Statistics by Confidence Bin:")
        st.write(stats_df)

        y_pred = model.predict(x_oob)
        st.write(f"Accuracy: {accuracy_score(y_oob, y_pred)}")

        report_dict = classification_report(y_oob, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.write("Classification Report")
        st.dataframe(report_df)

    # In Distribution Validation Statistics
    st.title("ID Validation Statistics")
    valid_data, stats_df = calculate_validation_stats(model, valid_full, x_valid, y_valid)
    if valid_data is not None:
        st.write("Statistics by Confidence Bin:")
        st.write(stats_df)

        y_pred = model.predict(x_valid)
        st.write(f"Accuracy: {accuracy_score(y_valid, y_pred)}")

        report_dict = classification_report(y_valid, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.write("Classification Report")
        st.dataframe(report_df)

if __name__ == "__main__":
    main()