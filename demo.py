import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from fpdf import FPDF
import google.generativeai as genai
import io
import base64
import time
import os

#--- Page Configuration ---

st.set_page_config(page_title="Groundwater Forecast App", layout="wide")

#--- Gemini API Configuration ---

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") gemini_configured = False

if GEMINI_API_KEY and GEMINI_API_KEY != "Gemini_api_key": try: # Configure Gemini with the API key genai.configure(api_key=GEMINI_API_KEY)

# Load Gemini models with custom generation settings
    generation_config = genai.types.GenerationConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_output_tokens=4000
    )

    gemini_model_report = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        generation_config=generation_config
    )

    gemini_model_chat = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        generation_config=generation_config
    )

    gemini_configured = True

except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. AI features might be limited.")

else: st.warning("Gemini API Key not found or is placeholder. AI features will be disabled. Set GEMINI_API_KEY environment variable or update in code.")



# --- Model Paths & Constants ---
# Ensure this path is relative to the app.py file for deployment
STANDARD_MODEL_DIR = "standard_model.h5"
STANDARD_MODEL_FILENAME = "standard_model.h5"
STANDARD_MODEL_PATH = os.path.join(STANDARD_MODEL_DIR, STANDARD_MODEL_FILENAME)

STANDARD_MODEL_SEQUENCE_LENGTH = 60 # Default, will be updated if model loads
if os.path.exists(STANDARD_MODEL_PATH):
    try:
        # Load model without compiling to avoid issues with custom/missing metrics like 'mse' string
        _std_model_temp = load_model(STANDARD_MODEL_PATH, compile=False)
        STANDARD_MODEL_SEQUENCE_LENGTH = _std_model_temp.input_shape[1]
        del _std_model_temp
        # st.info(f"Standard model structure loaded successfully from {STANDARD_MODEL_PATH} to infer sequence length: {STANDARD_MODEL_SEQUENCE_LENGTH}")
    except Exception as e:
        st.warning(f"Could not load standard model from {STANDARD_MODEL_PATH} to infer sequence length: {e}. Using default {STANDARD_MODEL_SEQUENCE_LENGTH}.")
else:
    st.warning(f"Standard model file not found at relative path: {STANDARD_MODEL_PATH}. Please ensure it exists in the {STANDARD_MODEL_DIR} directory next to app.py.")

# --- Helper Functions ---
@st.cache_data
def load_and_clean_data(uploaded_file_content):
    try:
        df = pd.read_excel(io.BytesIO(uploaded_file_content), engine="openpyxl")
        if df.shape[1] < 2: st.error("File must have at least two columns (Date, Level)."); return None
        date_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["date", "time"])), None)
        level_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["level", "groundwater", "gwl"])), None)
        if not date_col: st.error("Cannot find Date column (e.g., named 'Date', 'Time')."); return None
        if not level_col: st.error("Cannot find Level column (e.g., named 'Level', 'Groundwater Level')."); return None
        st.success(f"Identified columns: Date='{date_col}', Level='{level_col}'. Renaming to 'Date' and 'Level'.")
        df = df.rename(columns={date_col: "Date", level_col: "Level"})[["Date", "Level"]]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
        initial_rows = len(df)
        df.dropna(subset=["Date", "Level"], inplace=True)
        if len(df) < initial_rows: st.warning(f"Dropped {initial_rows - len(df)} rows with invalid/missing date or level values.")
        if df.empty: st.error("No valid data remaining after cleaning."); return None
        df = df.sort_values(by="Date").reset_index(drop=True).drop_duplicates(subset=["Date"], keep="first")
        if df["Level"].isnull().any():
            missing_before = df["Level"].isnull().sum()
            df["Level"] = df["Level"].interpolate(method="linear", limit_direction="both")
            st.warning(f"Filled {missing_before} missing level values using linear interpolation.")
        if df["Level"].isnull().any(): st.error("Could not fill all missing values even after interpolation."); return None
        st.success("Data loaded and cleaned successfully!")
        return df
    except Exception as e: st.error(f"An unexpected error occurred during data loading/cleaning: {e}"); return None

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

@st.cache_resource # Cache model loading
def load_keras_model_from_file(uploaded_file_obj, model_name_for_log):
    temp_model_path = f"temp_{model_name_for_log.replace(' ', '_')}.h5"
    try:
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_file_obj.getbuffer())
        # Load model without compiling to avoid issues with custom/missing metrics like 'mse' string
        model = load_model(temp_model_path, compile=False)
        sequence_length = model.input_shape[1]
        st.success(f"Loaded {model_name_for_log}. Inferred sequence length: {sequence_length}")
        return model, sequence_length
    except Exception as e:
        st.error(f"Error loading Keras model {model_name_for_log}: {e}")
        return None, None
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@st.cache_resource
def load_standard_model_cached(path):
    try:
        # Load model without compiling to avoid issues with custom/missing metrics like 'mse' string
        model = load_model(path, compile=False)
        sequence_length = model.input_shape[1]
        return model, sequence_length
    except Exception as e:
        st.error(f"Error loading standard Keras model from {path}: {e}")
        return None, None

def build_lstm_model(sequence_length, n_features=1):
    model = Sequential([LSTM(40, activation="relu", input_shape=(sequence_length, n_features)), Dropout(0.5), Dense(1)])
    model.compile(optimizer="adam", loss="mean_squared_error") # For training, we compile with loss
    return model

def predict_with_dropout_uncertainty(model, last_sequence_scaled, n_steps, n_iterations, scaler, model_sequence_length):
    all_predictions = []
    current_sequence = last_sequence_scaled.copy().reshape(1, model_sequence_length, 1)
    @tf.function
    def predict_step_training_true(inp): return model(inp, training=True)
    progress_bar = st.progress(0); status_text = st.empty()
    for i in range(n_iterations):
        iteration_predictions_scaled = []
        temp_sequence = current_sequence.copy()
        for _ in range(n_steps):
            next_pred_scaled = predict_step_training_true(temp_sequence).numpy()[0,0]
            iteration_predictions_scaled.append(next_pred_scaled)
            temp_sequence = np.append(temp_sequence[:, 1:, :], np.array([[next_pred_scaled]]).reshape(1,1,1), axis=1)
        all_predictions.append(iteration_predictions_scaled)
        progress_bar.progress((i + 1) / n_iterations); status_text.text(f"MC Dropout Iteration: {i+1}/{n_iterations}")
    progress_bar.empty(); status_text.empty()
    predictions_array_scaled = np.array(all_predictions)
    mean_preds_scaled = np.mean(predictions_array_scaled, axis=0)
    std_devs_scaled = np.std(predictions_array_scaled, axis=0)
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    lower_bound = scaler.inverse_transform((mean_preds_scaled - 1.96 * std_devs_scaled).reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform((mean_preds_scaled + 1.96 * std_devs_scaled).reshape(-1, 1)).flatten()
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.inf
    if np.all(y_true != 0): mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# --- Plotting Functions ---
def create_forecast_plot(historical_df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df["Date"], y=historical_df["Level"], mode="lines", name="Historical Data", line=dict(color="rgb(31, 119, 180)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast", line=dict(color="rgb(255, 127, 14)")))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper_CI"], mode="lines", name="Upper CI (95%)", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower_CI"], mode="lines", name="Lower CI (95%)", line=dict(width=0), fillcolor="rgba(255, 127, 14, 0.2)", fill="tonexty", showlegend=False))
    fig.update_layout(title="Groundwater Level: Historical Data & LSTM Forecast", xaxis_title="Date", yaxis_title="Groundwater Level", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template="plotly_white")
    return fig

def create_loss_plot(history_dict):
    if not history_dict or not isinstance(history_dict, dict) or "loss" not in history_dict or "val_loss" not in history_dict:
        fig = go.Figure()
        fig.update_layout(title="No Training History Available", xaxis_title="Epoch", yaxis_title="Loss")
        fig.add_annotation(text="Training history is not available for pre-trained models or if training did not occur.",xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    history_df = pd.DataFrame(history_dict); history_df["Epoch"] = history_df.index + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["loss"], mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_loss"], mode="lines", name="Validation Loss"))
    fig.update_layout(title="Model Training & Validation Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss (MSE)", hovermode="x unified", template="plotly_white")
    return fig

# --- Gemini API Functions ---
def generate_gemini_report(hist_df, forecast_df, metrics, language):
    if not gemini_configured: return "AI report generation disabled. Configure Gemini API Key."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient data for AI report."
    try:
        prompt = f"""Act as a professional hydrologist. Generate a scientific interpretation of the groundwater level forecast in {language}. 
        Analyze the historical trend (summary below), the forecast trend (from {forecast_df["Forecast"].iloc[0]:.2f} to {forecast_df["Forecast"].iloc[-1]:.2f}), 
        the 95% confidence interval range (e.g., final interval: {forecast_df["Lower_CI"].iloc[-1]:.2f} - {forecast_df["Upper_CI"].iloc[-1]:.2f}), 
        and the LSTM model's performance on the validation set (RMSE: {metrics.get("RMSE", "N/A")}, MAE: {metrics.get("MAE", "N/A")}, MAPE: {metrics.get("MAPE", "N/A")}). 
        Discuss potential aquifer conditions suggested by the data and forecast. 
        Assess the risk of over-pumping or water shortage based on the trends. 
        Provide actionable scientific recommendations for groundwater management or monitoring based on these findings. 
        Ensure the tone is professional and the language is {language}.

        Historical Data Summary:
        {hist_df["Level"].describe().to_string()}

        Forecast Data Summary:
        {forecast_df[["Forecast", "Lower_CI", "Upper_CI"]].describe().to_string()}
        """
        response = gemini_model_report.generate_content(prompt)
        return response.text
    except Exception as e: st.error(f"Error generating AI report: {e}"); return f"Error generating AI report: {e}"

def get_gemini_chat_response(user_query, chat_hist, hist_df, forecast_df, metrics, ai_report):
    if not gemini_configured: return "AI chat disabled. Configure Gemini API Key."
    if hist_df is None or forecast_df is None or metrics is None: return "Error: Insufficient context for AI chat."
    try:
        context = f"""Context for AI Chatbot:
        Historical Data: {hist_df["Level"].describe().to_string()}
        Forecast: {forecast_df.to_string()}
        Metrics: RMSE: {metrics.get("RMSE", "N/A")}, MAE: {metrics.get("MAE", "N/A")}, MAPE: {metrics.get("MAPE", "N/A")}
        AI Report: {ai_report if ai_report else "N/A"}
        Chat History: {chat_hist}
        User: {user_query}
AI:"""
        response = gemini_model_chat.generate_content(context)
        return response.text
    except Exception as e: st.error(f"Error in AI chat: {e}"); return f"Error in AI chat: {e}"

# --- Main Forecasting Pipeline ---
def run_forecast_pipeline(df, model_choice, forecast_horizon, custom_model_file_obj, 
                        sequence_length_train_param, epochs_train_param, 
                        mc_iterations_param, use_custom_scaler_params_flag, custom_scaler_min_param, custom_scaler_max_param):
    st.info(f"Starting forecast pipeline with model: {model_choice}")
    model = None
    model_sequence_length = sequence_length_train_param
    history_data = None
    scaler_obj = MinMaxScaler(feature_range=(0, 1))

    try:
        st.info("Step 1: Preparing Model...")
        if model_choice == "Standard Pre-trained Model":
            if os.path.exists(STANDARD_MODEL_PATH):
                model, model_sequence_length = load_standard_model_cached(STANDARD_MODEL_PATH)
                if model is None: return None, None, None, None
                st.session_state.model_sequence_length = model_sequence_length
            else:
                st.error(f"Standard model not found at {STANDARD_MODEL_PATH}. Check repository structure."); return None, None, None, None
        elif model_choice == "Upload Custom .h5 Model" and custom_model_file_obj is not None:
            model, model_sequence_length = load_keras_model_from_file(custom_model_file_obj, "Custom Model")
            if model is None: return None, None, None, None
            st.session_state.model_sequence_length = model_sequence_length
        elif model_choice == "Train New Model":
            model_sequence_length = sequence_length_train_param
            st.session_state.model_sequence_length = model_sequence_length
        else:
            st.error("Invalid model choice or no custom model uploaded."); return None, None, None, None
        st.info(f"Model preparation complete. Effective sequence length: {model_sequence_length}")

        st.info("Step 2: Preprocessing Data (Scaling)...")
        if model_choice != "Train New Model":
            if use_custom_scaler_params_flag and custom_scaler_min_param is not None and custom_scaler_max_param is not None and custom_scaler_min_param < custom_scaler_max_param:
                st.info(f"Using provided scaler parameters: Original Min={custom_scaler_min_param}, Original Max={custom_scaler_max_param}")
                scaler_obj.fit(np.array([[custom_scaler_min_param], [custom_scaler_max_param]]))
                scaled_data = scaler_obj.transform(df["Level"].values.reshape(-1, 1))
            else:
                st.warning("Scaler parameters not provided or invalid for pre-trained model. Fitting a new scaler to the current data.")
                scaled_data = scaler_obj.fit_transform(df["Level"].values.reshape(-1, 1))
        else: # Train New Model
            st.info("Fitting new scaler for model training.")
            scaled_data = scaler_obj.fit_transform(df["Level"].values.reshape(-1, 1))
        st.info("Data scaling complete.")

        st.info(f"Step 3: Creating sequences with length {model_sequence_length}...")
        if len(df) <= model_sequence_length:
            st.error(f"Not enough data ({len(df)} points) for sequence length {model_sequence_length}. Need > {model_sequence_length}."); return None, None, None, None
        X, y = create_sequences(scaled_data, model_sequence_length)
        if len(X) == 0:
            st.error(f"Could not create sequences."); return None, None, None, None
        st.info(f"Sequence creation complete. Number of sequences: {len(X)}")

        evaluation_metrics = {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan}
        if model_choice == "Train New Model":
            st.info(f"Step 4a: Training New LSTM Model (Epochs: {epochs_train_param})...")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            if len(X_train) == 0 or len(X_val) == 0:
                st.error("Not enough data for train/validation split."); return None, None, None, None
            model = build_lstm_model(model_sequence_length)
            early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            history_obj = model.fit(X_train, y_train, epochs=epochs_train_param, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
            history_data = history_obj.history
            st.success("Model training complete.")
            st.info("Evaluating trained model...")
            val_predictions_scaled = model.predict(X_val)
            val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
            y_val_actual = scaler_obj.inverse_transform(y_val)
            evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
            st.success("Evaluation of trained model complete.")
        else: # Pre-trained model
            st.info("Step 4b: Evaluating Pre-trained Model (on last 20% of available sequences)...")
            if len(X) > 5: # Ensure enough data for a meaningful pseudo-validation
                val_split_idx = int(len(X) * 0.8)
                X_val_pseudo, y_val_pseudo = X[val_split_idx:], y[val_split_idx:]
                if len(X_val_pseudo) > 0:
                    val_predictions_scaled = model.predict(X_val_pseudo)
                    val_predictions = scaler_obj.inverse_transform(val_predictions_scaled)
                    y_val_actual = scaler_obj.inverse_transform(y_val_pseudo)
                    evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
                    st.success("Pseudo-evaluation of pre-trained model complete.")
                else: st.warning("Not enough data for pseudo-validation of pre-trained model (after split).")
            else: st.warning("Not enough data for pseudo-validation of pre-trained model (total sequences too few).")

        st.info(f"Step 5: Forecasting Future {forecast_horizon} Steps (MC Dropout: {mc_iterations_param})...")
        last_sequence_scaled_for_pred = scaled_data[-model_sequence_length:]
        mean_forecast, lower_bound, upper_bound = predict_with_dropout_uncertainty(model, last_sequence_scaled_for_pred, forecast_horizon, mc_iterations_param, scaler_obj, model_sequence_length)
        st.success("Forecasting complete.")

        last_date = df["Date"].iloc[-1]
        try: freq = pd.infer_freq(df["Date"].dropna()); freq = freq if freq else "D"
        except: freq = "D"
        try: date_offset = pd.tseries.frequencies.to_offset(freq)
        except ValueError: date_offset = pd.DateOffset(days=1)
        forecast_dates = pd.date_range(start=last_date + date_offset, periods=forecast_horizon, freq=date_offset)
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": mean_forecast, "Lower_CI": lower_bound, "Upper_CI": upper_bound})
        
        st.info("Forecast pipeline finished successfully.")
        return forecast_df, evaluation_metrics, history_data, scaler_obj

    except Exception as e:
        st.error(f"An error occurred in the forecast pipeline: {e}")
        import traceback; st.error(traceback.format_exc())
        return None, None, None, None

# --- Initialize Session State ---
for key in ["cleaned_data", "forecast_results", "evaluation_metrics", "training_history", 
            "ai_report", "scaler_object", "forecast_plot_fig", "uploaded_data_filename"]:
    if key not in st.session_state: st.session_state[key] = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "chat_active" not in st.session_state: st.session_state.chat_active = False
if "model_sequence_length" not in st.session_state: st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH
if "run_forecast_triggered" not in st.session_state: st.session_state.run_forecast_triggered = False

# --- Sidebar ---
st.sidebar.title("Groundwater Forecast Control Panel")
st.sidebar.header("1. Upload Data")
uploaded_data_file = st.sidebar.file_uploader("Choose an XLSX data file", type="xlsx", key="data_uploader")

st.sidebar.header("2. Model Selection & Configuration")
model_choice = st.sidebar.selectbox("Choose Model Type", ("Standard Pre-trained Model", "Train New Model", "Upload Custom .h5 Model"), key="model_select")

custom_model_file_obj_sidebar = None
custom_scaler_min_sidebar, custom_scaler_max_sidebar = None, None
use_custom_scaler_sidebar = False
sequence_length_train_sidebar = st.session_state.model_sequence_length # Initialize with current session or default
epochs_train_sidebar = 50

if model_choice == "Upload Custom .h5 Model":
    custom_model_file_obj_sidebar = st.sidebar.file_uploader("Upload your .h5 model", type="h5", key="custom_h5_uploader")
    use_custom_scaler_sidebar = st.sidebar.checkbox("Provide custom scaler parameters for uploaded model?", value=False, key="use_custom_scaler_cb")
    if use_custom_scaler_sidebar:
        st.sidebar.markdown("Enter the **original min/max values** your custom model was scaled with (before 0-1 normalization).")
        custom_scaler_min_sidebar = st.sidebar.number_input("Original Data Min Value", value=0.0, format="%.4f", key="custom_scaler_min_in")
        custom_scaler_max_sidebar = st.sidebar.number_input("Original Data Max Value", value=1.0, format="%.4f", key="custom_scaler_max_in")
elif model_choice == "Standard Pre-trained Model":
    st.sidebar.info(f"Using standard model. Default sequence length: {st.session_state.model_sequence_length}")
    use_custom_scaler_sidebar = st.sidebar.checkbox("Provide custom scaler parameters for standard model?", value=False, key="use_std_scaler_cb")
    if use_custom_scaler_sidebar:
        st.sidebar.markdown("Enter the **original min/max values** the standard model was scaled with (if known).")
        custom_scaler_min_sidebar = st.sidebar.number_input("Original Data Min Value (Standard Model)", value=0.0, format="%.4f", key="std_scaler_min_in")
        custom_scaler_max_sidebar = st.sidebar.number_input("Original Data Max Value (Standard Model)", value=1.0, format="%.4f", key="std_scaler_max_in")
elif model_choice == "Train New Model":
    sequence_length_train_sidebar = st.sidebar.number_input("LSTM Sequence Length (for training)", min_value=10, max_value=365, value=st.session_state.model_sequence_length, step=10, key="seq_len_train_in")
    epochs_train_sidebar = st.sidebar.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10, key="epochs_train_in")

mc_iterations_sidebar = st.sidebar.number_input("MC Dropout Iterations (for C.I.)", min_value=10, max_value=500, value=100, step=10, key="mc_iter_in")
forecast_horizon_sidebar = st.sidebar.number_input("Forecast Horizon (steps)", min_value=1, max_value=100, value=12, step=1, key="horizon_in")

if st.sidebar.button("Run Forecast", key="run_forecast_main_btn"):
    st.session_state.run_forecast_triggered = True
    if st.session_state.cleaned_data is not None:
        if model_choice == "Upload Custom .h5 Model" and custom_model_file_obj_sidebar is None:
            st.error("Please upload a custom .h5 model file if that option is selected.")
            st.session_state.run_forecast_triggered = False
        else:
            with st.spinner(f"Running forecast with {model_choice}. This may take a few moments..."):
                forecast_df, metrics, history, scaler_obj = run_forecast_pipeline(
                    st.session_state.cleaned_data, model_choice, forecast_horizon_sidebar, 
                    custom_model_file_obj_sidebar, sequence_length_train_sidebar, epochs_train_sidebar, 
                    mc_iterations_sidebar, use_custom_scaler_sidebar, custom_scaler_min_sidebar, custom_scaler_max_sidebar
                )
            st.session_state.forecast_results = forecast_df
            st.session_state.evaluation_metrics = metrics
            st.session_state.training_history = history
            st.session_state.scaler_object = scaler_obj
            if forecast_df is not None and metrics is not None:
                st.session_state.forecast_plot_fig = create_forecast_plot(st.session_state.cleaned_data, forecast_df)
                st.success("Forecast complete! Results are available in the tabs.")
                st.session_state.ai_report = None; st.session_state.chat_history = []; st.session_state.chat_active = False
            else:
                st.error("Forecast pipeline did not complete successfully. Check messages above.")
                st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
                st.session_state.training_history = None; st.session_state.forecast_plot_fig = None
            st.rerun()
    else:
        st.error("Please upload data first using the sidebar.")
        st.session_state.run_forecast_triggered = False

st.sidebar.header("3. View & Export")
show_report_button = st.sidebar.button("Generate AI Report", key="show_report_btn", disabled=not gemini_configured)
report_language = st.sidebar.selectbox("Report Language", ["English", "French"], key="report_lang_select", disabled=not gemini_configured)
download_report_button = st.sidebar.button("Download Report (PDF)", key="download_report_btn")

st.sidebar.header("4. AI Assistant")
chat_with_ai_button = st.sidebar.button("Activate/Deactivate Chat", key="chat_ai_btn", disabled=not gemini_configured)
if chat_with_ai_button:
    st.session_state.chat_active = not st.session_state.chat_active
    if not st.session_state.chat_active: st.session_state.chat_history = []
    st.rerun()

# --- Main Application Area ---
st.title("💧 Groundwater Level Time Series Forecasting")
st.markdown("Upload data, select/train model, get forecasts with AI insights.")

if uploaded_data_file is not None:
    if st.session_state.get("uploaded_data_filename") != uploaded_data_file.name:
        st.session_state.uploaded_data_filename = uploaded_data_file.name
        with st.spinner("Loading and cleaning data..."):
            cleaned_df_result = load_and_clean_data(uploaded_data_file.getvalue())
        if cleaned_df_result is not None:
            st.session_state.cleaned_data = cleaned_df_result
            st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
            st.session_state.training_history = None; st.session_state.ai_report = None
            st.session_state.chat_history = []; st.session_state.scaler_object = None
            st.session_state.forecast_plot_fig = None
            st.session_state.model_sequence_length = STANDARD_MODEL_SEQUENCE_LENGTH # Reset sequence length on new data
            st.session_state.run_forecast_triggered = False
            st.rerun()
        else:
            st.session_state.cleaned_data = None
            st.error("Data loading failed. Please check the file format and content.")

data_tab, forecast_tab, evaluation_tab, ai_report_tab, chat_tab = st.tabs([
    "Data Preview", "Forecast Results", "Model Evaluation", "AI Report", "AI Chatbot"
])

with data_tab:
    st.header("Uploaded & Cleaned Data Preview")
    if st.session_state.cleaned_data is not None:
        st.dataframe(st.session_state.cleaned_data)
        st.write(f"Shape: {st.session_state.cleaned_data.shape}")
        st.metric("Time Range", f"{st.session_state.cleaned_data['Date'].min():%Y-%m-%d} to {st.session_state.cleaned_data['Date'].max():%Y-%m-%d}")
        st.metric("Data Points", len(st.session_state.cleaned_data))
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(x=st.session_state.cleaned_data["Date"], y=st.session_state.cleaned_data["Level"], mode="lines", name="Level"))
        fig_data.update_layout(title="Historical Groundwater Levels", xaxis_title="Date", yaxis_title="Level", template="plotly_white")
        st.plotly_chart(fig_data, use_container_width=True)
    else:
        st.info("⬆️ Please upload an XLSX data file using the sidebar to begin.")

with forecast_tab:
    st.header("Forecast Results")
    if st.session_state.forecast_results is not None and isinstance(st.session_state.forecast_results, pd.DataFrame) and not st.session_state.forecast_results.empty:
        if st.session_state.forecast_plot_fig is not None:
            st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
        else: st.warning("Forecast plot could not be generated. Displaying data table only.")
        st.subheader("Forecast Data Table"); st.dataframe(st.session_state.forecast_results)
    elif st.session_state.run_forecast_triggered: st.warning("Forecast process was run, but no results are available. Check errors.")
    else: st.info("Run a forecast using the sidebar to see results here.")

with evaluation_tab:
    st.header("Model Evaluation")
    if st.session_state.evaluation_metrics is not None and isinstance(st.session_state.evaluation_metrics, dict):
        st.subheader("Performance Metrics (on validation or pseudo-validation set)")
        col1, col2, col3 = st.columns(3)
        rmse_val = st.session_state.evaluation_metrics.get("RMSE", np.nan)
        mae_val = st.session_state.evaluation_metrics.get("MAE", np.nan)
        mape_val = st.session_state.evaluation_metrics.get("MAPE", np.nan)
        col1.metric("RMSE", f"{rmse_val:.4f}" if not np.isnan(rmse_val) else "N/A")
        col2.metric("MAE", f"{mae_val:.4f}" if not np.isnan(mae_val) else "N/A")
        col3.metric("MAPE", f"{mape_val:.2f}%" if not np.isnan(mape_val) and mape_val != np.inf else ("N/A" if np.isnan(mape_val) else "Inf"))
        st.subheader("Training Loss (if model was trained in this session)")
        if st.session_state.training_history:
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else: st.info("No training history available (e.g., using a pre-trained model or training failed).")
    elif st.session_state.run_forecast_triggered: st.warning("Forecast process was run, but no evaluation metrics are available. Check errors.")
    else: st.info("Run a forecast to see model evaluation metrics here.")

if show_report_button:
    if not gemini_configured: st.error("AI Report disabled. Configure Gemini API Key.")
    elif st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
        with st.spinner(f"Generating AI report ({report_language})..."):
            st.session_state.ai_report = generate_gemini_report(
                st.session_state.cleaned_data, st.session_state.forecast_results,
                st.session_state.evaluation_metrics, report_language
            )
        if st.session_state.ai_report and not st.session_state.ai_report.startswith("Error:"):
            st.success("AI report generated.")
        else: st.error(f"Failed to generate AI report. {st.session_state.ai_report}")
        st.rerun()
    else: st.error("Cleaned data, forecast results, and evaluation metrics must be available. Run a successful forecast first.")

with ai_report_tab:
    st.header("AI-Generated Scientific Report")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    if st.session_state.ai_report: st.markdown(st.session_state.ai_report)
    else: st.info("Click \"Generate AI Report\" in the sidebar after a successful forecast.")

with chat_tab:
    st.header("AI Chatbot Assistant")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    elif st.session_state.chat_active:
        if st.session_state.cleaned_data is not None and st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
            st.info("Chat activated. Ask about the results.")
            for sender, message in st.session_state.chat_history:
                with st.chat_message(sender.lower()): st.markdown(message)
            user_input = st.chat_input("Ask the AI assistant:")
            if user_input:
                st.session_state.chat_history.append(("User", user_input))
                with st.chat_message("user"): st.markdown(user_input)
                with st.spinner("AI thinking..."):
                    ai_response = get_gemini_chat_response(
                        user_input, st.session_state.chat_history, st.session_state.cleaned_data,
                        st.session_state.forecast_results, st.session_state.evaluation_metrics, st.session_state.ai_report
                    )
                st.session_state.chat_history.append(("AI", ai_response))
                with st.chat_message("ai"): st.markdown(ai_response)
                st.rerun()
        else:
            st.warning("Run a successful forecast to provide context for the chatbot.")
            st.session_state.chat_active = False; st.rerun()
    else:
        st.info("Click \"Activate/Deactivate Chat\" in sidebar (requires forecast results)." if gemini_configured else "AI Chat disabled.")

if download_report_button:
    if st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None and st.session_state.ai_report is not None and st.session_state.forecast_plot_fig is not None:
        with st.spinner("Generating PDF report..."):
            try:
                pdf = FPDF()
                pdf.add_page()
                font_path_dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                report_font = "Arial" # Fallback font
                if os.path.exists(font_path_dejavu):
                    try: 
                        pdf.add_font("DejaVu", fname=font_path_dejavu, uni=True)
                        report_font = "DejaVu"
                    except RuntimeError as font_err:
                        st.warning(f"Failed to add DejaVu font ({font_err}), using Arial.")
                else: 
                    st.warning(f"DejaVu font not found at {font_path_dejavu}, using Arial. For better PDF character support on Render, consider adding a .ttf font to your repo and referencing it, or installing fonts via a build script.")
                
                pdf.set_font(report_font, size=12)
                pdf.cell(0, 10, txt="Groundwater Level Forecast Report", new_x="LMARGIN", new_y="NEXT", align="C"); pdf.ln(5)

                plot_filename = "forecast_plot.png"
                try:
                    st.session_state.forecast_plot_fig.write_image(plot_filename, scale=2)
                    img_width_mm = 190 
                    pdf.image(plot_filename, x=pdf.get_x(), y=pdf.get_y(), w=img_width_mm)
                    pdf.ln(125) # Adjust based on typical plot height
                except Exception as img_err: 
                    st.warning(f"Could not save/embed plot image: {img_err}. Plot omitted from PDF.")
                finally:
                    if os.path.exists(plot_filename): os.remove(plot_filename)

                pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Model Evaluation Metrics", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                pdf.set_font(report_font, size=10)
                for key, value in st.session_state.evaluation_metrics.items():
                    val_str = f"{value:.4f}" if isinstance(value, (float, np.floating)) and not np.isnan(value) else str(value)
                    pdf.cell(0, 8, txt=f"{key}: {val_str}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(5)

                pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt="Forecast Data (First 10 points)", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                pdf.set_font(report_font, size=8); col_widths = [35, 35, 35, 35]
                pdf.cell(col_widths[0], 7, txt="Date", border=1)
                pdf.cell(col_widths[1], 7, txt="Forecast", border=1)
                pdf.cell(col_widths[2], 7, txt="Lower CI", border=1)
                pdf.cell(col_widths[3], 7, txt="Upper CI", border=1, new_x="LMARGIN", new_y="NEXT")
                for _, row in st.session_state.forecast_results.head(10).iterrows():
                    pdf.cell(col_widths[0], 6, txt=str(row["Date"].date()), border=1)
                    pdf.cell(col_widths[1], 6, txt=f"{row['Forecast']:.2f}", border=1)
                    pdf.cell(col_widths[2], 6, txt=f"{row['Lower_CI']:.2f}", border=1)
                    pdf.cell(col_widths[3], 6, txt=f"{row['Upper_CI']:.2f}", border=1, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(5)

                pdf.set_font(report_font, "B", size=11); pdf.cell(0, 10, txt=f"AI-Generated Report ({report_language})", new_x="LMARGIN", new_y="NEXT"); pdf.ln(1)
                pdf.set_font(report_font, size=10)
                pdf.multi_cell(0, 5, txt=st.session_state.ai_report)
                pdf.ln(5)

                pdf_output_bytes = pdf.output(dest="S").encode("latin-1")
                # b64_pdf = base64.b64encode(pdf_output_bytes).decode() # Not needed for direct download
                st.sidebar.download_button(
                    label="Download PDF Report Now", # Changed label to be more direct
                    data=pdf_output_bytes,
                    file_name="groundwater_forecast_report.pdf",
                    mime="application/octet-stream",
                    key="pdf_download_final_btn" # Added a unique key
                )
                st.success("PDF report ready. Click the download button in the sidebar.")
            except Exception as pdf_err:
                st.error(f"Failed to generate PDF report: {pdf_err}")
                import traceback; st.error(traceback.format_exc())
    else:
        st.error("Required data for PDF report is missing. Run a forecast and generate AI report first.")

