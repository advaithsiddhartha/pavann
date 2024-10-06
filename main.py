import streamlit as st

# Set the title of the app
st.title("App Update Notice")

# Add a message about the update
st.write(
    "We're updating our app for enhanced functionalities. Here is a video of the prototype of the first version:"
)

# Embed the updated YouTube video
video_url = "https://www.youtube.com/embed/2FBeQwbQXPA"
st.markdown(
    f'<iframe width="80%" height="315" src="{video_url}" frameborder="0" allowfullscreen></iframe>',
    unsafe_allow_html=True
)

# Add information about Team Sirius and Pavan.AI
st.subheader("Team Sirius Presents Pavan.AI")
st.write(
    "An innovative platform for real-time air quality monitoring in India, utilizing AI/ML for accurate NO2 data visualization."
)

# Highlights section
st.subheader("Highlights")
highlights = [
    "🌍 **Real-time Data:** Access live air quality data across various Indian cities.",
    "🤖 **AI Integration:** Utilizes Pavan.AI for accurate NO2 level predictions.",
    "📊 **User-Friendly:** Easy city selection and data retrieval interface.",
    "🌐 **Advanced Visualization:** Downloadable visualizations for enhanced usability.",
    "📈 **Data Analysis:** Users can upload CSV files for custom air quality analysis.",
    "📍 **Dynamic Mapping:** Interactive AQI map to explore air quality nationwide.",
    "🔒 **Secure Access:** Features a login and notification system for personalized updates."
]
for item in highlights:
    st.write(item)

# Key Insights section
st.subheader("Key Insights")
key_insights = [
    "🌬️ **Real-time Monitoring:** The platform provides immediate access to air quality data, crucial for researchers and the public to stay informed about pollution levels.",
    "📉 **Predictive Modeling:** By employing LSTM models within TensorFlow, the system effectively predicts NO2 levels, enhancing accuracy and reliability of data.",
    "🗺️ **Geospatial Integration:** The use of GeoPandas for downscaling satellite data allows for higher resolution air quality insights, making them more relevant to users.",
    "🔄 **Dynamic User Experience:** Users interact with a seamless interface that allows them to retrieve specific air quality information based on geographic coordinates.",
    "📊 **Data Visualization:** The site offers comprehensive visual tools, such as AQI maps and detailed graphs, which facilitate better understanding of air quality trends.",
    "📥 **Custom Data Uploads:** Users can upload their own CSV files for analysis, allowing for personalized insights tailored to specific datasets.",
    "🔔 **User Engagement:** The inclusion of a login feature and alerts ensures that users are kept updated with the latest air quality information relevant to their selected areas."
]
for insight in key_insights:
    st.write(insight)

# Run the app using: streamlit run your_script.py
