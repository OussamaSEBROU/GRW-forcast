import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from fpdf import FPDF
import google.generativeai as genai
import io
import base64
import time
import os

# --- Page Configuration ---
st.set_page_config(page_title="Groundwater Forecast App", layout="wide")

# --- Gemini API Configuration ---
# IMPORTANT: Replace 'Gemini_api_key' with your actual API key
# It's recommended to use Streamlit secrets or environment variables for API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "Gemini_api_key") # Placeholder
try:
    if GEMINI_API_KEY == "Gemini_api_key" or not GEMINI_API_KEY:
        st.warning("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable or replace the placeholder in the code. AI features will be disabled.")
        gemini_configured = False
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model_report = genai.GenerativeModel("gemini-pro")
        gemini_model_chat = genai.GenerativeModel("gemini-pro") # Can use the same or different
        gemini_configured = True
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. AI features might not work.")
    gemini_configured = False

# --- Helper Functions (Data, LSTM, Metrics, Plotting - Keep as before) ---

@st.cache_data
def load_and_clean_data(uploaded_file_content):
    try:
        # Explicitly use openpyxl engine for reading Excel files
        df = pd.read_excel(io.BytesIO(uploaded_file_content), engine='openpyxl')
        if df.shape[1] < 2: st.error("Error: File must have at least two columns (Date, Level)."); return None
        date_col, level_col = None, None
        potential_date_cols = [col for col in df.columns if any(kw in col.lower() for kw in ["date", "time"])]
        potential_level_cols = [col for col in df.columns if any(kw in col.lower() for kw in ["level", "groundwater", "gwl"])]
        if not potential_date_cols: st.error("Error: Cannot find Date column (containing 'date' or 'time')."); return None
        if not potential_level_cols: st.error("Error: Cannot find Level column (containing 'level', 'groundwater', 'gwl')."); return None
        date_col, level_col = potential_date_cols[0], potential_level_cols[0]
        st.success(f"Identified: Date=\'{date_col}\', Level=\'{level_col}\'. Renaming to 'Date', 'Level'.")
        df = df.rename(columns={date_col: 'Date', level_col: 'Level'})[['Date', 'Level']]
        try: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception as e: st.error(f"Error parsing Date: {e}."); return None
        df['Level'] = pd.to_numeric(df['Level'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['Date', 'Level'], inplace=True)
        if len(df) < initial_rows: st.warning(f"Dropped {initial_rows - len(df)} rows with invalid values.")
        if df.empty: st.error("Error: No valid data left after cleaning."); return None
        df = df.sort_values(by='Date').reset_index(drop=True).drop_duplicates(subset=['Date'], keep='first')
        if df['Level'].isnull().any():
            missing_count = df['Level'].isnull().sum()
            st.warning(f"{missing_count} missing values found. Applying linear interpolation.")
            df['Level'] = df['Level'].interpolate(method='linear', limit_direction='both')
        if df['Level'].isnull().any(): st.error("Error: Could not fill all missing values."); return None
        st.success("Data loaded and cleaned.")
        return df
    except Exception as e: st.error(f"Error loading/cleaning data: {e}"); return None

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(sequence_length, n_features=1):
    model = Sequential([LSTM(40, activation='relu', input_shape=(sequence_length, n_features)), Dropout(0.5), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_with_mc_dropout(model, last_sequence, n_steps, n_iterations, scaler, sequence_length):
    all_predictions = []
    current_sequence = last_sequence.copy().reshape(1, sequence_length, 1)
    @tf.function
    def predict_step(inp): return model(inp, training=True)
    progress_bar = st.progress(0); status_text = st.empty()
    for i in range(n_iterations):
        iteration_predictions = []
        temp_sequence = current_sequence.copy()
        for _ in range(n_steps):
            next_pred_scaled = predict_step(temp_sequence).numpy()[0, 0]
            iteration_predictions.append(next_pred_scaled)
            next_pred_scaled_reshaped = np.array([[next_pred_scaled]])
            temp_sequence = np.append(temp_sequence[:, 1:, :], next_pred_scaled_reshaped.reshape(1, 1, 1), axis=1)
        all_predictions.append(iteration_predictions)
        progress = (i + 1) / n_iterations
        progress_bar.progress(progress); status_text.text(f"MC Dropout Iteration: {i+1}/{n_iterations}")
    progress_bar.empty(); status_text.empty()
    predictions_array = np.array(all_predictions)
    mean_preds_scaled = np.mean(predictions_array, axis=0)
    std_devs_scaled = np.std(predictions_array, axis=0)
    mean_preds = scaler.inverse_transform(mean_preds_scaled.reshape(-1, 1)).flatten()
    lower_bound_scaled = mean_preds_scaled - 1.96 * std_devs_scaled
    upper_bound_scaled = mean_preds_scaled + 1.96 * std_devs_scaled
    lower_bound = scaler.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
    upper_bound = scaler.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten()
    return mean_preds, lower_bound, upper_bound

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    try: mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except: mape = np.inf
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def create_forecast_plot(historical_df, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_df['Date'], y=historical_df['Level'], mode='lines', name='Historical', line=dict(color='rgb(31, 119, 180)')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='rgb(255, 127, 14)')))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper_CI'], mode='lines', name='Upper CI', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Lower_CI'], mode='lines', name='Lower CI', line=dict(width=0), fillcolor='rgba(255, 127, 14, 0.2)', fill='tonexty', showlegend=False))
    fig.update_layout(title='Groundwater Level: Historical & Forecast', xaxis_title='Date', yaxis_title='Level', hovermode='x unified', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), template='plotly_white')
    return fig

def create_loss_plot(history_dict):
    history_df = pd.DataFrame(history_dict); history_df['Epoch'] = history_df.index + 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df['Epoch'], y=history_df['loss'], mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(x=history_df['Epoch'], y=history_df['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(title='Model Training & Validation Loss', xaxis_title='Epoch', yaxis_title='Loss (MSE)', hovermode='x unified', template='plotly_white')
    return fig

# --- Gemini API Functions ---

def generate_gemini_report(hist_df, forecast_df, metrics, language):
    if not gemini_configured: return "AI report generation disabled. Please configure the Gemini API Key."
    try:
        prompt = f"""Act as a professional hydrologist. Generate a scientific interpretation of the groundwater level forecast in {language}. 
        Analyze the historical trend (summary below), the forecast trend (from {forecast_df['Forecast'].iloc[0]:.2f} to {forecast_df['Forecast'].iloc[-1]:.2f}), 
        the 95% confidence interval range (e.g., final interval: {forecast_df['Lower_CI'].iloc[-1]:.2f} - {forecast_df['Upper_CI'].iloc[-1]:.2f}), 
        and the LSTM model's performance on the validation set (RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.1f}%). 
        Discuss potential aquifer conditions suggested by the data and forecast. 
        Assess the risk of over-pumping or water shortage based on the trends. 
        Provide actionable scientific recommendations for groundwater management or monitoring based on these findings. 
        Ensure the tone is professional and the language is {language}.

        Historical Data Summary:
        {hist_df['Level'].describe().to_string()}

        Forecast Data Summary:
        {forecast_df[['Forecast', 'Lower_CI', 'Upper_CI']].describe().to_string()}
        """
        response = gemini_model_report.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating AI report: {e}")
        return f"Error generating AI report: {e}"

def get_gemini_chat_response(user_query, chat_hist, hist_df, forecast_df, metrics, ai_report):
    if not gemini_configured: return "AI chat disabled. Please configure the Gemini API Key."
    try:
        # Construct context, limiting length if necessary
        context = f"""Context for the AI Chatbot:
        You are an AI assistant helping a user understand groundwater level forecasts.
        Historical Data Summary:
        {hist_df['Level'].describe().to_string()}

        Forecast Results (next {len(forecast_df)} steps):
        {forecast_df.to_string()}

        Model Evaluation Metrics (Validation Set):
        RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.2f}%

        AI-Generated Scientific Report:
        {ai_report if ai_report else 'Not generated yet.'}

        Chat History (User and AI turns):
        """
        history_text = "\n".join([f"{sender}: {msg}" for sender, msg in chat_hist])
        full_prompt = context + history_text + f"\nUser: {user_query}\nAI:"

        # Limit prompt length if needed (simple truncation here)
        max_prompt_length = 8000 # Adjust as needed based on model limits
        if len(full_prompt) > max_prompt_length:
            cutoff = len(full_prompt) - max_prompt_length
            full_prompt = context[:max(0, len(context)-cutoff)] + history_text + f"\nUser: {user_query}\nAI:"
            st.warning("Chat context truncated due to length limits.")

        response = gemini_model_chat.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting chat response: {e}")
        return f"Error getting chat response: {e}"

# --- Main Forecasting Pipeline (Keep as before) --- 
def run_lstm_forecast(df, forecast_horizon, sequence_length=60, epochs=50, batch_size=32, mc_iterations=100):
    try:
        status_text = st.empty(); status_text.info("1. Preprocessing..."); time.sleep(0.5)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df['Level'].values.reshape(-1, 1))
        X, y = create_sequences(scaled_data, sequence_length)
        if len(X) == 0: st.error(f"Error: Not enough data ({len(df)}) for sequence length {sequence_length}. Need > {sequence_length}."); return None, None, None, None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        status_text.info("2. Building Model..."); time.sleep(0.5)
        model = build_lstm_model(sequence_length)
        status_text.info(f"3. Training Model (Epochs: {epochs})...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
        st.success("Training complete.")
        status_text.info("4. Evaluating..."); time.sleep(0.5)
        val_predictions_scaled = model.predict(X_val)
        val_predictions = scaler.inverse_transform(val_predictions_scaled)
        y_val_actual = scaler.inverse_transform(y_val)
        evaluation_metrics = calculate_metrics(y_val_actual, val_predictions)
        st.success("Evaluation complete.")
        status_text.info(f"5. Forecasting (MC Iterations: {mc_iterations})...")
        last_sequence_scaled = scaled_data[-sequence_length:]
        mean_forecast, lower_bound, upper_bound = predict_with_mc_dropout(model, last_sequence_scaled, forecast_horizon, mc_iterations, scaler, sequence_length)
        st.success("Forecasting complete.")
        last_date = df['Date'].iloc[-1]
        try: freq = pd.infer_freq(df['Date']); freq = freq or 'D'
        except: freq = 'D'
        try: date_offset = pd.tseries.frequencies.to_offset(freq)
        except ValueError: date_offset = pd.DateOffset(days=1)
        forecast_dates = pd.date_range(start=last_date + date_offset, periods=forecast_horizon, freq=date_offset)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': mean_forecast, 'Lower_CI': lower_bound, 'Upper_CI': upper_bound})
        status_text.empty()
        return forecast_df, evaluation_metrics, history.history, scaler
    except Exception as e:
        st.error(f"Error during forecast: {e}")
        import traceback; st.error(traceback.format_exc())
        if 'status_text' in locals(): status_text.empty()
        return None, None, None, None

# --- Initialize Session State --- 
if 'cleaned_data' not in st.session_state: st.session_state.cleaned_data = None
if 'forecast_results' not in st.session_state: st.session_state.forecast_results = None
if 'evaluation_metrics' not in st.session_state: st.session_state.evaluation_metrics = None
if 'training_history' not in st.session_state: st.session_state.training_history = None
if 'ai_report' not in st.session_state: st.session_state.ai_report = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = [] # List of tuples (sender, message)
if 'scaler' not in st.session_state: st.session_state.scaler = None
if 'forecast_plot_fig' not in st.session_state: st.session_state.forecast_plot_fig = None
if 'chat_active' not in st.session_state: st.session_state.chat_active = False # Control chat display

# --- Sidebar --- 
st.sidebar.title("Groundwater Forecast Control Panel")
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose an XLSX file", type="xlsx")
st.sidebar.header("2. Configure Forecast")
sequence_length_input = st.sidebar.number_input("LSTM Sequence Length", min_value=10, max_value=200, value=60, step=10)
epochs_input = st.sidebar.number_input("Training Epochs", min_value=10, max_value=500, value=50, step=10)
mc_iterations_input = st.sidebar.number_input("MC Dropout Iterations", min_value=10, max_value=500, value=100, step=10)
forecast_horizon = st.sidebar.number_input("Forecast Horizon", min_value=1, max_value=100, value=12, step=1)
run_forecast_button = st.sidebar.button("Run Forecast", key="run_forecast", disabled=not gemini_configured and False) # Keep enabled even if Gemini fails for now

st.sidebar.header("3. View & Export")
show_report_button = st.sidebar.button("Generate AI Report", key="show_report", disabled=not gemini_configured)
report_language = st.sidebar.selectbox("Report Language", ["English", "French"], key="report_lang", disabled=not gemini_configured)
download_report_button = st.sidebar.button("Download Report (PDF)", key="download_report")

st.sidebar.header("4. AI Assistant")
chat_with_ai_button = st.sidebar.button("Activate/Deactivate Chat", key="chat_ai", disabled=not gemini_configured)
if chat_with_ai_button:
    st.session_state.chat_active = not st.session_state.chat_active
    if not st.session_state.chat_active:
        st.session_state.chat_history = [] # Clear history when deactivated
    st.rerun()

# --- Main Application Area --- 
st.title("💧 Groundwater Level Time Series Forecasting")
st.markdown("Upload groundwater level data (.xlsx), configure LSTM, and get forecasts with AI insights.")

# --- Data Upload Logic ---
if uploaded_file is not None:
    uploaded_file_content = uploaded_file.getvalue()
    cleaned_df_result = load_and_clean_data(uploaded_file_content)
    if cleaned_df_result is not None:
        st.session_state.cleaned_data = cleaned_df_result
        st.session_state.forecast_results = None; st.session_state.evaluation_metrics = None
        st.session_state.training_history = None; st.session_state.ai_report = None
        st.session_state.chat_history = []; st.session_state.scaler = None
        st.session_state.forecast_plot_fig = None; st.session_state.chat_active = False
    else: st.session_state.cleaned_data = None

# --- Tabs --- 
data_tab, forecast_tab, evaluation_tab, ai_report_tab, chat_tab = st.tabs([
    "Data Preview & Upload", "Forecast Results", "Model Evaluation", "AI Report", "AI Chatbot"
])

with data_tab:
    st.header("Cleaned Data Preview")
    if st.session_state.cleaned_data is not None:
        st.dataframe(st.session_state.cleaned_data)
        st.write(f"Shape: {st.session_state.cleaned_data.shape}")
        st.metric("Range", f"{st.session_state.cleaned_data['Date'].min():%Y-%m-%d} to {st.session_state.cleaned_data['Date'].max():%Y-%m-%d}")
        st.metric("Points", len(st.session_state.cleaned_data))
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(x=st.session_state.cleaned_data['Date'], y=st.session_state.cleaned_data['Level'], mode='lines', name='Level'))
        fig_data.update_layout(title='Historical Levels', xaxis_title='Date', yaxis_title='Level', template='plotly_white')
        st.plotly_chart(fig_data, use_container_width=True)
    else: st.info("⬆️ Upload XLSX file via sidebar.")

# --- Forecasting Logic ---
if run_forecast_button:
    if st.session_state.cleaned_data is not None:
        if len(st.session_state.cleaned_data) > sequence_length_input:
            with st.spinner(f"Running forecast... (Seq: {sequence_length_input}, Epochs: {epochs_input}, Horizon: {forecast_horizon})"):
                forecast_df, metrics, history, scaler_obj = run_lstm_forecast(
                    st.session_state.cleaned_data, forecast_horizon, sequence_length_input, epochs_input, mc_iterations=mc_iterations_input
                )
            if forecast_df is not None:
                st.session_state.forecast_results = forecast_df
                st.session_state.evaluation_metrics = metrics
                st.session_state.training_history = history
                st.session_state.scaler = scaler_obj
                st.session_state.forecast_plot_fig = create_forecast_plot(st.session_state.cleaned_data, forecast_df)
                st.success("Forecast complete! Results updated.")
                st.session_state.ai_report = None; st.session_state.chat_history = []; st.session_state.chat_active = False
                st.rerun()
            else: st.error("Forecast failed. See errors.")
        else: st.error(f"Not enough data ({len(st.session_state.cleaned_data)}) for sequence length {sequence_length_input}. Need > {sequence_length_input}.")
    else: st.error("Upload data first.")

# --- Display Forecast Results ---
with forecast_tab:
    st.header("Forecast Results")
    if st.session_state.forecast_results is not None and st.session_state.forecast_plot_fig is not None:
        st.plotly_chart(st.session_state.forecast_plot_fig, use_container_width=True)
        st.subheader("Forecast Data")
        st.dataframe(st.session_state.forecast_results)
    else: st.info("Run forecast to see results.")

# --- Display Evaluation Metrics ---
with evaluation_tab:
    st.header("Model Evaluation (Validation Set)")
    if st.session_state.evaluation_metrics is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{st.session_state.evaluation_metrics['RMSE']:.4f}")
        col2.metric("MAE", f"{st.session_state.evaluation_metrics['MAE']:.4f}")
        col3.metric("MAPE", f"{st.session_state.evaluation_metrics['MAPE']:.2f}%")
        st.subheader("Training Loss")
        if st.session_state.training_history:
            loss_fig = create_loss_plot(st.session_state.training_history)
            st.plotly_chart(loss_fig, use_container_width=True)
        else: st.info("No training history.")
    else: st.info("Run forecast to see metrics.")

# --- AI Report Logic ---
if show_report_button:
    if not gemini_configured: st.error("AI Report disabled. Configure Gemini API Key.")
    elif st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None:
        with st.spinner(f"Generating AI report ({report_language})..."):
            st.session_state.ai_report = generate_gemini_report(
                st.session_state.cleaned_data,
                st.session_state.forecast_results,
                st.session_state.evaluation_metrics,
                report_language
            )
        st.success("AI report generated.")
        st.rerun()
    else: st.error("Run forecast first.")

# --- Display AI Report ---
with ai_report_tab:
    st.header("AI-Generated Scientific Report")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    if st.session_state.ai_report: st.markdown(st.session_state.ai_report)
    else: st.info("Click 'Generate AI Report' after running forecast.")

# --- AI Chatbot Logic ---
with chat_tab:
    st.header("AI Chatbot Assistant")
    if not gemini_configured: st.warning("AI features disabled. Configure Gemini API Key.")
    elif st.session_state.chat_active:
        if st.session_state.forecast_results is not None:
            st.info("Chat activated. Ask about the results.")
            # Display chat history using st.chat_message
            for sender, message in st.session_state.chat_history:
                with st.chat_message(sender.lower()): # Use 'user' or 'ai'
                    st.markdown(message)

            # Get user input via st.chat_input
            user_input = st.chat_input("Ask the AI assistant:")
            if user_input:
                # Add user message to history and display
                st.session_state.chat_history.append(("User", user_input))
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Get AI response
                with st.spinner("AI thinking..."):
                    ai_response = get_gemini_chat_response(
                        user_input,
                        st.session_state.chat_history,
                        st.session_state.cleaned_data,
                        st.session_state.forecast_results,
                        st.session_state.evaluation_metrics,
                        st.session_state.ai_report
                    )
                # Add AI response to history and display
                st.session_state.chat_history.append(("AI", ai_response))
                with st.chat_message("ai"):
                    st.markdown(ai_response)
                # Streamlit reruns automatically with chat_input, no explicit rerun needed here
        else:
            st.warning("Run forecast first to provide context.")
            st.session_state.chat_active = False # Deactivate if no context
            st.rerun()
    else:
        st.info("Click 'Activate/Deactivate Chat' in sidebar to use the chatbot (requires forecast results)." if gemini_configured else "AI Chat disabled.")

# --- Report Export Logic ---
if download_report_button:
    if st.session_state.forecast_results is not None and st.session_state.evaluation_metrics is not None and st.session_state.ai_report is not None and st.session_state.forecast_plot_fig is not None:
        with st.spinner("Generating PDF report..."):
            try:
                pdf = FPDF()
                pdf.add_page()
                # Add DejaVu font for broader character support (especially if AI report uses non-Latin chars)
                try:
                    # Ensure the font file exists in the sandbox
                    # Note: This path might differ, adjust if needed or install font
                    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                    if os.path.exists(font_path):
                        pdf.add_font("DejaVu", fname=font_path, uni=True)
                        pdf.set_font("DejaVu", size=12)
                        report_font = "DejaVu"
                    else:
                        st.warning("DejaVu font not found at expected path. Using Arial (may cause issues with special characters).")
                        pdf.set_font("Arial", size=12)
                        report_font = "Arial"
                except RuntimeError:
                     st.warning("Error adding DejaVu font. Using Arial.")
                     pdf.set_font("Arial", size=12)
                     report_font = "Arial"

                pdf.cell(0, 10, txt="Groundwater Level Forecast Report", ln=1, align='C')
                pdf.ln(10)

                # Add Plot
                plot_filename = "/tmp/forecast_plot.png"
                try:
                    st.session_state.forecast_plot_fig.write_image(plot_filename, scale=2)
                    img_width_mm = 190
                    pdf.image(plot_filename, x=10, y=pdf.get_y(), w=img_width_mm)
                    # Estimate image height based on common aspect ratios (e.g., 16:9 or 4:3)
                    img_height_mm = img_width_mm * (9/16) # Adjust ratio if needed
                    pdf.ln(img_height_mm + 5) # Add some padding
                    os.remove(plot_filename)
                except Exception as img_err:
                    st.warning(f"Could not save/embed plot: {img_err}. Adding placeholder.")
                    pdf.set_font(report_font, 'I', size=10)
                    pdf.cell(0, 10, txt="[Forecast plot error]", ln=1, align='C'); pdf.ln(5)

                # Add Metrics
                pdf.set_font(report_font, 'B', size=11)
                pdf.cell(0, 10, txt="Model Evaluation Metrics", ln=1)
                pdf.set_font(report_font, size=10)
                for key, value in st.session_state.evaluation_metrics.items():
                    pdf.cell(0, 8, txt=f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}", ln=1)
                pdf.ln(5)

                # Add Forecast Table (Sample)
                pdf.set_font(report_font, 'B', size=11)
                pdf.cell(0, 10, txt="Forecast Data (Sample)", ln=1)
                pdf.set_font(report_font, size=9)
                col_widths = [35, 35, 35, 35]
                pdf.cell(col_widths[0], 8, txt="Date", border=1)
                pdf.cell(col_widths[1], 8, txt="Forecast", border=1)
                pdf.cell(col_widths[2], 8, txt="Lower CI", border=1)
                pdf.cell(col_widths[3], 8, txt="Upper CI", border=1, ln=1)
                pdf.set_font(report_font, size=8)
                for _, row in st.session_state.forecast_results.head(10).iterrows():
                    pdf.cell(col_widths[0], 7, txt=str(row['Date'].date()), border=1)
                    pdf.cell(col_widths[1], 7, txt=f"{row['Forecast']:.2f}", border=1)
                    pdf.cell(col_widths[2], 7, txt=f"{row['Lower_CI']:.2f}", border=1)
                    pdf.cell(col_widths[3], 7, txt=f"{row['Upper_CI']:.2f}", border=1, ln=1)
                pdf.ln(5)

                # Add AI Report
                pdf.set_font(report_font, 'B', size=11)
                pdf.cell(0, 10, txt=f"AI-Generated Report ({report_language})", ln=1)
                pdf.set_font(report_font, size=10)
                # Use multi_cell with UTF-8 handling if using DejaVu
                if report_font == "DejaVu":
                    pdf.multi_cell(0, 5, txt=st.session_state.ai_report)
                else: # Fallback for Arial (basic encoding)
                    pdf.multi_cell(0, 5, txt=st.session_state.ai_report.encode('latin-1', 'replace').decode('latin-1'))
                pdf.ln(10)

                pdf_output = pdf.output(dest='S').encode('latin-1') # Still use latin-1 for FPDF output bytes
                b64 = base64.b64encode(pdf_output).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="groundwater_forecast_report.pdf">Download PDF Report</a>'
                st.sidebar.markdown(href, unsafe_allow_html=True)
                st.success("PDF generated. Click link in sidebar.")
            except Exception as pdf_error:
                st.error(f"Failed to generate PDF: {pdf_error}")
                import traceback; st.error(traceback.format_exc())
    else:
        st.error("Run forecast, generate AI report, and ensure plot exists before downloading PDF.")


