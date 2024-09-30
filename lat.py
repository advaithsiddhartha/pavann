import streamlit as st
from PIL import Image, ImageDraw

def pixelate(image, pixel_size):
    """Pixelate the input image by reducing its size and scaling it back up."""
    small = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        resample=Image.NEAREST
    )
    return small.resize(image.size, Image.NEAREST)

def get_no2_value(color):
    """Determine NO2 value based on the average color of the cropped image."""
    # Example color to NO2 value mapping
    color_mapping = {
        (255, 0, 0): 100,  # Red for high NO2
        (0, 255, 0): 50,   # Green for medium NO2
        (0, 0, 255): 10,   # Blue for low NO2
        (255, 255, 255): 0  # White for no NO2
    }
    return color_mapping.get(color, 0)  # Default to 0 if color not found

def show_page():
    st.title("Map Latitude and Longitude Selection Window")
    st.info("`TEAM SIRIUS _ SMART INDIA HACKATHON`")
    st.success("R. Krishna Advaith Siddhartha ,  V. Subhash  , S. Ravi Teja   ,  K.R. Nakshathra , R. Bhoomika        , M. Abhinav")
    # Initialize session state
    if 'clicked_coords' not in st.session_state:
        st.session_state.clicked_coords = (0, 0)
    if 'city_selected' not in st.session_state:
        st.session_state.city_selected = None
    if 'use_existing' not in st.session_state:
        st.session_state.use_existing = False

    # City images mapping
    city_images = {
        "Hyderabad": "hyd.png",
        "Mumbai": "mum.png",
        "Chennai": "chn.png",
        "Delhi": "del.png",
        "Ahmedabad": "ahm.png",
        "Visakhapatnam": "vskp.png",
        "Bengaluru": "bng.png",
        "Bhopal":"bpl.png",
        "Indore":"ind.png",
        "Jaipur":"jpr.png",
        "Kanpur":"knp.png",
        "Lucknow":"lkn.png",
        "Patna":"ptn.png",
        "Surat":"srt.png",
    }

    # Button to continue with existing shape files
    if st.button("Continue with existing shape files"):
        st.session_state.use_existing = True

    # If using existing shape files, display city selection
    if st.session_state.use_existing:
        city_options = list(city_images.keys())
        city = st.selectbox("Select a City", city_options, key="city_selector")

        if city:
            st.session_state.city_selected = city
            image_path = city_images.get(city)

            if image_path:
                # Load and display the city image
                image = Image.open(image_path)
                st.image(image, caption=f"Image of {city}", use_column_width=True)

                # Input fields for selecting X and Y coordinates
                width, height = image.size
                default_x = min(width // 2, width - 25)
                default_y = min(height // 2, height - 25)

                x = st.number_input("X Coordinate", min_value=0, max_value=width, value=default_x, key="x_coord_city")
                y = st.number_input("Y Coordinate", min_value=0, max_value=height, value=default_y, key="y_coord_city")

                st.session_state.clicked_coords = (x, y)

                # Draw selection lines on the image
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)
                draw.line((x, 0, x, height), fill='red', width=2)  # Vertical line
                draw.line((0, y, width, y), fill='red', width=2)   # Horizontal line
                st.image(draw_image, caption="Image with Selection Lines", use_column_width=True)

                # Crop a 10x10 box around the selected coordinates
                left = max(0, x - 5)
                top = max(0, y - 5)
                right = min(width, x + 5)
                bottom = min(height, y + 5)

                cropped_image = image.crop((left, top, right, bottom)).resize((10, 10))
                st.image(cropped_image.resize((100, 100)), caption="Cropped Image (10x10)", use_column_width=False)

                # Get the average color of the cropped image
                avg_color = cropped_image.getcolors(cropped_image.size[0] * cropped_image.size[1])
                avg_color = max(avg_color)[1] if avg_color else (0, 0, 0)

                # Get the NO2 value based on the average color
                no2_value = get_no2_value(avg_color)
                st.write(f"Estimated NO2 Value: {no2_value} µg/m³")

                # Pixelate the cropped image
                pixelated_image = pixelate(cropped_image, pixel_size=1)
                st.image(pixelated_image.resize((100, 100)), caption="Pixelated Cropped Image", use_column_width=False)

    # Allow users to upload their own image if not using existing shape files
    else:
        uploaded_file = st.file_uploader("Upload a map image (up to 5000x5000 pixels)", type=["jpg", "png"], key="file_uploader")

        if uploaded_file:
            # Read and process the uploaded image
            image = Image.open(uploaded_file)

            if image.size[0] <= 5000 and image.size[1] <= 5000:
                st.image(image, caption="Uploaded Map", use_column_width=True)

                # Input fields for selecting X and Y coordinates
                width, height = image.size
                default_x = min(width // 2, width - 25)
                default_y = min(height // 2, height - 25)

                x = st.number_input("X Coordinate", min_value=0, max_value=width, value=default_x, key="x_coord_upload")
                y = st.number_input("Y Coordinate", min_value=0, max_value=height, value=default_y, key="y_coord_upload")

                st.session_state.clicked_coords = (x, y)

                # Draw selection lines on the image
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)
                draw.line((x, 0, x, height), fill='red', width=2)  # Vertical line
                draw.line((0, y, width, y), fill='red', width=2)   # Horizontal line
                st.image(draw_image, caption="Image with Selection Lines", use_column_width=True)

                # Crop a 10x10 box around the selected coordinates
                left = max(0, x - 5)
                top = max(0, y - 5)
                right = min(width, x + 5)
                bottom = min(height, y + 5)

                cropped_image = image.crop((left, top, right, bottom)).resize((10, 10))
                st.image(cropped_image.resize((100, 100)), caption="Cropped Image (10x10)", use_column_width=False)

                # Get the average color of the cropped image
                avg_color = cropped_image.getcolors(cropped_image.size[0] * cropped_image.size[1])
                avg_color = max(avg_color)[1] if avg_color else (0, 0, 0)

                # Get the NO2 value based on the average color
                no2_value = get_no2_value(avg_color)
                st.write(f"Estimateue: 3 µg/m³")

                # Pixelate the cropped image
                pixelated_image = pixelate(cropped_image, pixel_size=1)
                st.image(pixelated_image.resize((100, 100)), caption="Pixelated Cropped Image", use_column_width=False)

# Call the function to run the Streamlit app
if __name__ == "__main__":
    show_page()
