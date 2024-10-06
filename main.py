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
    f'<iframe width="560" height="315" src="{video_url}" frameborder="0" allowfullscreen></iframe>',
    unsafe_allow_html=True
)

# Run the app using: streamlit run your_script.py
