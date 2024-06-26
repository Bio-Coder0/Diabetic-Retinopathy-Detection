import streamlit as st
import numpy as np
import cv2
import imutils

# Function to show images in Streamlit
def show_image(image, title="Processed Image"):
    st.image(image, use_column_width=True, caption=title)

# Function to process the input image
def process_image(input_image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for red color (in HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    # Mask the red color
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    # Find contours of the red areas
    cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Find the contour of the square (assuming it's the largest contour)
    square_contour = max(cnts, key=cv2.contourArea)
    
    # Filter out the square contour from the list of contours
    cnts = [c for c in cnts if c is not square_contour]
    
    # Calculate the minimum enclosing circle of the square contour
    ((_, _), square_radius) = cv2.minEnclosingCircle(square_contour)
    
    # Calculate the pixel-to-millimeter ratio based on the size of the square (2 mm by 2 mm)
    square_size_mm = 10  # Size of the square in millimeters
    pixel_per_mm = square_radius / square_size_mm
    
    # Initialize list to store measurements
    measurements = []
    
    # Image dimensions
    height, width = input_image.shape[:2]
    
    # Iterate over contours to find red circles
    for c in cnts:
        # Find the centroid of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Check if the centroid is too close to the image border
            border_distance = min(cX, cY, width - cX, height - cY)
            if border_distance > 20:  # Adjust the distance threshold as needed
                # Find the minimum enclosing circle of the contour
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                # Check if the contour meets the criteria for a red circle
                if 20 < radius < 100:  # Adjust the radius range as needed
                    # Convert radius from pixels to millimeters
                    radius_mm = radius / pixel_per_mm
                    
                    # Append measurements to the list
                    measurements.append((x, y, radius_mm))
    
    # Draw the circles on the input image
    for (x, y, radius_mm) in measurements:
        cv2.circle(input_image, (int(x), int(y)), int(radius_mm * pixel_per_mm), (0, 0, 255), 2)
        cv2.putText(input_image, "{:.1f}mm".format(radius_mm), (int(x - radius_mm), int(y - radius_mm - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return input_image

# Streamlit app
st.title("Skin Prick Test")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    # Perform image processing
    processed_image = process_image(image)
    # Display the processed image
    show_image(processed_image, title="Processed Image")
