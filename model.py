import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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
                fig1, ax1 = plt.subplots()
                ax1.plot(data['NO2(GT)'], label='Original NO₂ Levels', color='blue')
                ax1.set_title('Time Series of NO₂ Levels')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('NO₂ Levels')
                ax1.legend()
                st.pyplot(fig1)

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
                fig2, ax2 = plt.subplots()
                ax2.plot(Y_test_rescaled, label="Actual NO₂ Levels", color='orange')
                ax2.plot(Y_pred_rescaled, label="Predicted NO₂ Levels", color='green')
                ax2.set_title("Actual vs Predicted NO₂ Levels")
                ax2.legend()
                st.pyplot(fig2)

                # Step 10: Histogram of NO₂ Levels
                fig3, ax3 = plt.subplots()
                ax3.hist(data['NO2(GT)'], bins=30, color='purple', alpha=0.7)
                ax3.set_title("Histogram of NO₂ Levels")
                ax3.set_xlabel("NO₂ Levels")
                ax3.set_ylabel("Frequency")
                st.pyplot(fig3)

                # Step 11: Box Plot of NO₂ Levels
                fig4, ax4 = plt.subplots()
                ax4.boxplot(data['NO2(GT)'], vert=False)
                ax4.set_title("Box Plot of NO₂ Levels")
                ax4.set_xlabel("NO₂ Levels")
                st.pyplot(fig4)

                # Step 12: Predicted vs Actual Scatter Plot
                fig5, ax5 = plt.subplots()
                ax5.scatter(Y_test_rescaled, Y_pred_rescaled, color='red')
                ax5.set_title("Predicted vs Actual NO₂ Levels")
                ax5.set_xlabel("Actual NO₂ Levels")
                ax5.set_ylabel("Predicted NO₂ Levels")
                ax5.plot([Y_test_rescaled.min(), Y_test_rescaled.max()], [Y_test_rescaled.min(), Y_test_rescaled.max()], color='blue', linestyle='--')
                st.pyplot(fig5)

                # Step 13: Loss Plot
                fig6, ax6 = plt.subplots()
                ax6.plot(history.history['loss'], label='Loss', color='teal')
                ax6.set_title("Model Loss over Epochs")
                ax6.set_xlabel("Epochs")
                ax6.set_ylabel("Loss")
                ax6.legend()
                st.pyplot(fig6)

            else:
                st.error("NO₂ column not found in the dataset.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

