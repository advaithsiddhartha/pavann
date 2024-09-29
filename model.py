import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def show_page():

    # Title of the app
    st.title("Air Quality Data Analysis with LSTM")
    st.info("`TEAM SIRIUS _ SMART INDIA HACKATHON`")
    st.success("R. Krishna Advaith Siddhartha, S. Ravi Teja, V. Subhash, R. Bhoomika, K.R. Nakshathra, M. Abhinav")

    # Step 1: File uploader for CSV file
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Step 2: Load and display the data
        try:
            data = pd.read_csv(uploaded_file, delimiter=';')  # assuming a semicolon delimiter
            st.write("Data loaded successfully!")
            
            # Display the first few rows of the dataframe
            st.write("First few rows of the data:")
            st.write(data.head())
            
            # Clean up column names by stripping spaces
            data.columns = data.columns.str.strip()

            # Step 3: Check for the NO₂ column
            if 'NO2(GT)' in data.columns:
                st.write("Found the 'NO2(GT)' column. Proceeding with preprocessing.")
                
                # Preprocess the data
                data['NO2(GT)'] = data['NO2(GT)'].ffill()  # Fill missing values using forward fill
                data = data.dropna(subset=['NO2(GT)'])  # Drop any remaining rows with missing NO₂ values
                
                # Step 4: Normalize the NO₂ data
                no2_data = data['NO2(GT)'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                no2_data_normalized = scaler.fit_transform(no2_data)
                
                st.write("Normalized NO₂ data (first 5 values):")
                st.write(no2_data_normalized[:5])
                
                # Plot 1: Time Series of Original NO₂ Levels
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=data.index, y=data['NO2(GT)'], mode='lines', name='Original NO₂ Levels', line=dict(color='blue')))
                fig1.update_layout(title='Time Series of NO₂ Levels', xaxis_title='Time', yaxis_title='NO₂ Levels')
                st.plotly_chart(fig1)

                # Step 5: Train/Test Split
                def create_dataset(data, time_step=1):
                    X, Y = [], []
                    for i in range(len(data) - time_step - 1):
                        X.append(data[i:(i + time_step), 0])
                        Y.append(data[i + time_step, 0])
                    return np.array(X), np.array(Y)

                time_step = 10
                X, Y = create_dataset(no2_data_normalized, time_step)
                X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input

                # Split into training and testing datasets
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

                # Step 6: Define the LSTM Model
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(25))
                model.add(Dense(1))

                model.compile(optimizer='adam', loss='mean_squared_error')

                # Step 7: Train the model
                st.write("Training the model... This may take a while.")
                history = model.fit(X_train, Y_train, batch_size=64, epochs=10, verbose=1)
                st.success("Model training completed!")

                # Step 8: Model Evaluation
                Y_pred = model.predict(X_test)

                # Inverse transform to get back to original scale
                Y_test_rescaled = scaler.inverse_transform(Y_test.reshape(-1, 1))
                Y_pred_rescaled = scaler.inverse_transform(Y_pred)

                # Calculate mean squared error
                mse = mean_squared_error(Y_test_rescaled, Y_pred_rescaled)
                st.write(f"Mean Squared Error on Test Data: {mse}")

                # Step 9: Plot Actual vs Predicted
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=np.arange(len(Y_test_rescaled)), y=Y_test_rescaled, mode='lines', name='Actual NO₂ Levels', line=dict(color='orange')))
                fig2.add_trace(go.Scatter(x=np.arange(len(Y_pred_rescaled)), y=Y_pred_rescaled, mode='lines', name='Predicted NO₂ Levels', line=dict(color='green')))
                fig2.update_layout(title='Actual vs Predicted NO₂ Levels', xaxis_title='Samples', yaxis_title='NO₂ Levels')
                st.plotly_chart(fig2)

                # Step 10: Histogram of NO₂ Levels
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(x=data['NO2(GT)'], nbinsx=30, name='NO₂ Levels', marker_color='purple'))
                fig3.update_layout(title='Histogram of NO₂ Levels', xaxis_title='NO₂ Levels', yaxis_title='Frequency')
                st.plotly_chart(fig3)

                # Step 11: Predicted vs Actual Scatter Plot
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=Y_test_rescaled.flatten(), y=Y_pred_rescaled.flatten(), mode='markers', name='Predicted vs Actual', marker=dict(color='red', size=5)))
                fig4.add_trace(go.Scatter(x=[Y_test_rescaled.min(), Y_test_rescaled.max()], y=[Y_test_rescaled.min(), Y_test_rescaled.max()], mode='lines', name='Perfect Prediction', line=dict(color='blue', dash='dash')))
                fig4.update_layout(title='Predicted vs Actual NO₂ Levels', xaxis_title='Actual NO₂ Levels', yaxis_title='Predicted NO₂ Levels')
                st.plotly_chart(fig4)

                # Step 12: Loss Plot
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=np.arange(len(history.history['loss'])), y=history.history['loss'], mode='lines', name='Loss', line=dict(color='teal')))
                fig5.update_layout(title='Model Loss over Epochs', xaxis_title='Epochs', yaxis_title='Loss')
                st.plotly_chart(fig5)

            else:
                st.error("NO₂ column not found in the dataset.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

