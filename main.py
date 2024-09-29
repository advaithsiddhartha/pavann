# import streamlit as st
# import api
# import lat
# import model
# from streamlit_navigation_bar import st_navbar
# import base64

# # Function to load and encode image as base64
# def get_base64_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# # Set page layout to wide
# st.set_page_config(layout="wide")
# st.write(" ")
# page = st_navbar(["API WINDOW", "LATITUDE AND LONGITUDE TRACING", "LSTM MODEL WINDOW", "LOGIN", "PROFILE"])

# # Load the banner image
# banner_image_base64 = get_base64_image('./1.png')  # Make sure the path is correct

# # Add a banner-type header with a colored stripe background using the base64-encoded image
# st.markdown(f"""
#     <style>
#     .banner-image {{
#         width: 100%;
#         height: auto;
#     }}
#     </style>
#     <div>
#         <img src="data:image/png;base64,{banner_image_base64}" class="banner-image">
#     </div>
# """, unsafe_allow_html=True)

# # Create a navbar for page navigation

# # Set title below the banner
# st.write(" ")
# st.write(" ")
# st.title("SIH - Team Sirius - 1734")
# st.success("Select the navigation item to switch between pages")

# # Page routing logic based on navbar selection
# if page == "API WINDOW":
#     api.show_page()
# elif page == "LATITUDE AND LONGITUDE TRACING":
#     lat.show_page()
# elif page == "LSTM MODEL WINDOW":
#     model.show_page()

# # Ensure each module (`api`, `lat`, `model`) has a `show_page()` function to render the respective page.


import streamlit as st
import api
import lat
import model
from streamlit_navigation_bar import st_navbar
import base64

# Function to load and encode image as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Set page layout to wide
st.set_page_config(layout="wide")
page = st_navbar(["API WINDOW", "LATITUDE AND LONGITUDE TRACING", "LSTM MODEL WINDOW" , "LOGIN" ])

# Load the banner image
banner_image_base64 = get_base64_image('./1.png')  # Make sure the path is correct

# Add a banner-type header with a colored stripe background using the base64-encoded image
st.markdown(f"""
    <style>
    .banner-image {{
        width: 100%;
        height: auto;
    }}
    </style>
    <div>
        <img src="data:image/png;base64,{banner_image_base64}" class="banner-image">
    </div>
""", unsafe_allow_html=True)

# Inject CSS to style the navbar items (make them bigger and more prominent)
st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"] > button {
        font-size: 24px;
        padding: 10px 20px;
        text-align: center;
        color: white;
        background-color: #333;
        border-radius: 5px;
        margin: 5px;
        transition: background-color 0.3s ease;
    }
    div[data-testid="stHorizontalBlock"] > button:hover {
        background-color: #007bff;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Create a navbar for page navigation

# Set title below the banner
st.write(" ")
st.write(" ")
# Page routing logic based on navbar selection
if page == "API WINDOW":
    api.show_page()
elif page == "LATITUDE AND LONGITUDE TRACING":
    lat.show_page()
elif page == "LSTM MODEL WINDOW":
    model.show_page()

# Ensure each module (`api`, `lat`, `model`) has a `show_page()` function to render the respective page.