import re
import pandas as pd
import json
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from geopy.distance import geodesic

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Define a function to navigate between pages
def navigate_to(page):
    st.session_state.page = page

# Sidebar: Add navigation button for "Back to Home"
with st.sidebar:
    if st.session_state.page != 'home':  # Show the Back to Home button only if we're not already on the home page
        if st.button("Back to Home"):
            navigate_to('home')

# Logic to control which page to display
if st.session_state.page == 'home':
    # Home page with Vendor Management System title and buttons
    st.title("Vendor Management System")

    # Create two buttons for navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Vendor Selection"):
            navigate_to('vendor_selection')

    with col2:
        if st.button("BOQ"):
            navigate_to('boq')

elif st.session_state.page == 'vendor_selection':
    # Vendor Selection page content
    st.title("Vendor Selection")

    # Coordinates for major cities
    city_coordinates = {
        'Gurgaon': (28.4595, 77.0266),'Delhi': (28.7041, 77.1025),'Mumbai': (19.0760, 72.8777), 'Chennai': (13.0827, 80.2707),
        'Kolkata': (22.5726, 88.3639), 'Bangalore': (12.9716, 77.5946),'Hyderabad': (17.3850, 78.4867), 'Pune': (18.5204, 73.8567),
        'Ahmedabad': (23.0225, 72.5714),'Jaipur': (26.9124, 75.7873),'Lucknow': (26.8467, 80.9462),'Bhopal': (23.2599, 77.4126),
        'Patna': (25.5941, 85.1376),'Indore': (22.7196, 75.8577),'Kanpur': (26.4499, 80.3319),'Nagpur': (21.1458, 79.0882),
        'Visakhapatnam': (17.6868, 83.2185),'Vadodara': (22.3072, 73.1812), 'Surat': (21.1702, 72.8311),'Varanasi': (25.3176, 82.9739),
        'Amritsar': (31.6340, 74.8723),'Ludhiana': (30.9010, 75.8573), 'Agra': (27.1767, 78.0081),'Meerut': (28.9845, 77.7064),
        'Rajkot': (22.3039, 70.8022),'Coimbatore': (11.0168, 76.9558),'Madurai': (9.9252, 78.1198),'Nashik': (19.9975, 73.7898),
        'Jodhpur': (26.2389, 73.0243), 'Ranchi': (23.3441, 85.3096),'Guwahati': (26.1445, 91.7362),'Chandigarh': (30.7333, 76.7794),
        'Mysore': (12.2958, 76.6394),'Thrissur': (10.5276, 76.2144),'Thiruvananthapuram': (8.5241, 76.9366),'Vijayawada': (16.5062, 80.6480),
        'Gwalior': (26.2183, 78.1828),'Kochi': (9.9312, 76.2673),'Faridabad': (28.4089, 77.3178),'Noida': (28.5355, 77.3910),
        'Ghaziabad': (28.6692, 77.4538),'Dehradun': (30.3165, 78.0322),'Shimla': (31.1048, 77.1734),'Jammu': (32.7266, 74.8570),
        'Panaji': (15.4909, 73.8278),'Bhubaneswar': (20.2961, 85.8245),'Raipur': (21.2514, 81.6296),'Bilaspur': (22.0796, 82.1391),
        'Jabalpur': (23.1815, 79.9864),'Aurangabad': (19.8762, 75.3433),'Tirupati': (13.6288, 79.4192),'Rourkela': (22.2270, 84.8524),
        'Durgapur': (23.5204, 87.3119),'Silchar': (24.8333, 92.7789),'Shillong': (25.5788, 91.8933),'Kozhikode': (11.2588, 75.7804),
        'Alappuzha': (9.4981, 76.3388),'Navi Mumbai': (19.0330, 73.0297),'Chennai': (13.0827, 80.2707),'Coimbatore': (11.0168, 76.9558),
        'Madurai': (9.9252, 78.1198),'Tiruchirappalli': (10.7905, 78.7047),'Salem': (11.6643, 78.1460),'Tirunelveli': (8.7139, 77.7567),
        'Erode': (11.3410, 77.7172),'Vellore': (12.9165, 79.1325),'Thoothukudi': (8.7642, 78.1348),'Tiruppur': (11.1085, 77.3411),
        'Dindigul': (10.3673, 77.9803),'Thanjavur': (10.7870, 79.1378),'Sivagangai': (9.8477, 78.4815),'Virudhunagar': (9.5810, 77.9624),
        'Nagapattinam': (10.7672, 79.8420),'Ramanathapuram': (9.3762, 78.8308),'Namakkal': (11.2189, 78.1677),'Cuddalore': (11.7447, 79.7689),
        'Karur': (10.9571, 78.0792),'Theni': (10.0104, 77.4777),'Kanyakumari': (8.0883, 77.5385),'Krishnagiri': (12.5186, 78.2137),
        'Perambalur': (11.2320, 78.8806),'Ariyalur': (11.1428, 79.0782),'Nilgiris': (11.4916, 76.7337),'Ranipet': (12.9224, 79.3326),
        'Tiruvannamalai': (12.2253, 79.0747),'Villupuram': (11.9395, 79.4924),'Kallakurichi': (11.7376, 78.9597),'Chengalpattu': (12.6921, 79.9707),
        'Tenkasi': (8.9604, 77.3152),'Tirupattur': (12.4967, 78.5730),'Pudukkottai': (10.3797, 78.8205),'Thiruvarur': (10.7668, 79.6345),
        'Mayiladuthurai': (11.1036, 79.6491),'Dharmapuri': (12.1357, 78.1602)
    }

    # Function to calculate distance between Gurgaon and user-specified location
    def calculate_distance_to_gurgaon(user_place):
        gurgaon_coords = city_coordinates['Gurgaon']
        if user_place in city_coordinates:
            user_place_coords = city_coordinates[user_place]
            return geodesic(gurgaon_coords, user_place_coords).kilometers
        else:
            st.error(f"Coordinates for '{user_place}' not found.")
            st.stop()

    # Load CSV files (dummy paths, replace with actual paths)
    vendordata = pd.read_csv('vendor_data.csv')
    technicalrating = pd.read_csv('tech_rating.csv')

    # Clean column names
    vendordata.columns = vendordata.columns.str.strip()
    technicalrating.columns = technicalrating.columns.str.strip()

    # Merge datasets on 'FACADE VENDOR' column
    merged_data = pd.merge(vendordata, technicalrating, on='FACADE VENDOR')
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

    # Preserve original vendor names and factory locations
    vendor_names = merged_data['FACADE VENDOR']
    factory_locations = merged_data['Factory Location']

    # Convert categorical columns to numerical format
    le = LabelEncoder()
    for column in merged_data.select_dtypes(include=['object']).columns:
        merged_data[column] = le.fit_transform(merged_data[column])

    # Convert 'Recommended' column to numerical values
    if 'Recommended' in merged_data.columns:
        merged_data['Recommended'] = merged_data['Recommended'].apply(lambda x: 1 if x == 'YES' else 0)

    # Sidebar: City selection for calculating distance from Gurgaon
    city_options = sorted(city_coordinates.keys())
    user_place = st.sidebar.selectbox("Select your location", city_options)

    if st.sidebar.button("Calculate and Rank Vendors"):
        # Calculate the distance from Gurgaon
        distance_to_user_place = calculate_distance_to_gurgaon(user_place)
        st.write(f"**Distance between Gurgaon and {user_place}: {distance_to_user_place:.2f} km**")

        # Update the 'Distance of Factory from Gurgaon (Kms)' column
        merged_data['Distance of Factory from Gurgaon (Kms)'] = distance_to_user_place

        # Define features and target
        X = merged_data.drop(columns=['FACADE VENDOR', 'Recommended'])
        y = merged_data['Recommended']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = xgb_model.predict(X_test)

        # Adjust the ranking formula
        merged_data['Ranking_Score'] = (merged_data['Production Capacity'] * 0.3) + \
                                       (merged_data['Turn Over'] * 0.2) + \
                                       (merged_data['Past Similar projects'] * 0.2) + \
                                       (merged_data['Average Rating'] * 0.1) - \
                                       (merged_data['Distance of Factory from Gurgaon (Kms)'] / 100 * 0.2)

        # Rank vendors based on the ranking score
        ranked_vendors = merged_data.sort_values(by='Ranking_Score', ascending=False)
        ranked_vendors['FACADE VENDOR'] = vendor_names.iloc[ranked_vendors.index]
        ranked_vendors['Factory Location'] = factory_locations.iloc[ranked_vendors.index]

        # Get the top vendors (you can adjust how many vendors to display)
        top_vendors = ranked_vendors.head(2)  # Adjust to show more vendors if needed

        for index, vendor in top_vendors.iterrows():
            # Create a Streamlit container for vendor information
            with st.container():
                # Create two columns for the image and the details
                col1, col2 = st.columns([1, 3])

                # Vendor image in the first column
                with col1:
                    st.image("icon.jpg", width=150)

                # Vendor details in the second column
                with col2:
                    st.subheader(vendor['FACADE VENDOR'])
                    st.write(f"**Factory Location:** {vendor['Factory Location']}")
                    st.write(f"**Production Capacity:** {vendor['Production Capacity']}")
                    st.write(f"**Turn Over:** {vendor['Turn Over']}")
                    st.write(f"**Past Similar Projects:** {vendor['Past Similar projects']}")
                    st.write(f"**Distance from Gurgaon:** {vendor['Distance of Factory from Gurgaon (Kms)']:.2f} km")
                    st.write(f"**Average Rating:** {vendor['Average Rating']}")
                    st.write(f"**Ranking Score:** {vendor['Ranking_Score']:.2f}")

                # Divider between vendors
                st.divider()

elif st.session_state.page == 'boq':
    # BOQ Page content
    st.title("Bill of Quantities (BOQ) Page")

    # BOQ feature extraction code here
    def extract_features_from_row(row: pd.Series) -> dict:
        features = {}

        # Extract features based on the product description in 'Unnamed: 0' column
        product_description = row['Unnamed: 0']

        # Series (e.g., SKN, Alucobond, Framing Member)
        series_match = re.search(r'\bSKN[-\s]\d+\b|\b154\s*SKN\b|\bST[-\s]\d+\b|\bAlucobond\b|\bFraming Member\b', product_description, re.IGNORECASE)
        features['Series'] = series_match.group(0).strip() if series_match else None

        # Min Thickness (e.g., wind pressure or thickness mentions)
        thickness_match = re.search(r'(\b\d+\s*mm\b|\bminimum thickness of \d+\s*mm\b|\bminimum \d+\s*mm\b|\d+\s*mm thk\.|\bwind pressure\b)', product_description, re.IGNORECASE)
        features['Min Thickness'] = re.sub(r'(\bminimum thickness of\b|\bminimum\b|\sthk\.)', '', thickness_match.group(0)).strip() if thickness_match else None

        # Approved Makes (e.g., Saint Gobain, Alucobond, Hilti, Fischer)
        makes_match = re.search(r'(Saint\s*Gobain|Alucobond|Hilti|Fischer|Secondary\s*Steel\s*-?\s*M\.S\. structure)', product_description, re.IGNORECASE)
        features['Approved makes'] = makes_match.group(0).strip() if makes_match else None

        # Reflective Coating (e.g., surface mentions)
        reflective_coating_match = re.search(r'(surface\s*#\d+|Framing\s*member\s*-?\s*Finish)', product_description, re.IGNORECASE)
        features['Reflective coating'] = f"reflective coating shall be on {reflective_coating_match.group(0)}" if reflective_coating_match else None

        # Glass (e.g., insulated glass, laminated glass)
        glass_match = re.search(r'(insulated\s*glass\s*unit|insulated\s*glass|Glass\s*type\s*-?\s*Single\s*glass\s*/\s*Laminated|Laminated\s*Glass)', product_description, re.IGNORECASE)
        features['Glass'] = glass_match.group(0).strip() if glass_match else None

        # Special Seal (e.g., hermetically sealed, hardware)
        special_seal_match = re.search(r'(hermetically\s*sealed(?:\s*with\s*the\s*two\s*lites\s*of\s*glass)?|hardware)', product_description, re.IGNORECASE)
        features['Special seal'] = special_seal_match.group(0).strip() if special_seal_match else None

        # Air Gap (e.g., air space, glass type)
        air_gap_match = re.search(r'(\b\d+\s*mm\s*air\s*gap\b|\b\d+\s*mm\s*air\s*space\b|Glass\s*type\s*-?\s*Single\s*glass\s*/\s*Laminated)', product_description, re.IGNORECASE)
        features['Air Gap'] = re.sub(r'\s*air\s*space', ' air gap', air_gap_match.group(0).strip()) if air_gap_match else None

        # Spacer (e.g., black aluminum spacers, SGP/PVB)
        spacer_match = re.search(r'(black\s*aluminum\s*spacers|black\s*aluminium\s*spacers|SGP\s*/\s*PVB)', product_description, re.IGNORECASE)
        features['Spacer'] = spacer_match.group(0).strip() if spacer_match else None

        # Special Bend/Shape (e.g., Bent at corners, toughened clear glass)
        special_bend_shape_match = re.search(r'(Bent\s*at\s*corners|toughened\s*clear\s*glass)', product_description, re.IGNORECASE)
        features['Special bend/shape'] = special_bend_shape_match.group(0).strip() if special_bend_shape_match else None

        # Primary Sealant (e.g., Poly-Isobutylene, Spider fittings)
        primary_sealant_match = re.search(r'(Poly[-\s]*Isobutylene|Spider\s*fittings)', product_description, re.IGNORECASE)
        features['Primary Sealant'] = primary_sealant_match.group(0).strip() if primary_sealant_match else None

        # Secondary Silicon Sealant (e.g., Dow Corning, DC 991H, Silicone)
        secondary_sealant_match = re.search(r'(Dow\s*Corning\s*3362|DC\s*991H|Silicone|Sillicone)', product_description, re.IGNORECASE)
        features['Secondary silicon sealant'] = secondary_sealant_match.group(0).strip() if secondary_sealant_match else None

        # Codal Reference (e.g., BS EN, IS standards)
        codal_reference_match = re.search(r'(BS\s*EN\s*\d+|IS\s*875\s*part\s*III|AS\s*1288)', product_description, re.IGNORECASE)
        features['Codal reference'] = codal_reference_match.group(0).strip() if codal_reference_match else None

        # Special Treatment (e.g., heat strengthened, thermoplastic core, type of canopy)
        special_treatment_match = re.search(r'(heat\s*strengthened|thermoplastic\s*core\s*of\s*anti\s*oxidant\s*LDPE|Type\s*of\s*canopy\s*-?\s*Glass\s*canopy)', product_description, re.IGNORECASE)
        features['Special treatment'] = special_treatment_match.group(0).strip() if special_treatment_match else None

        return features

    # Sidebar: Input field for product description
# Use Markdown to make the label bold
    st.sidebar.markdown("**Enter Product Description**")

    # The text area remains as usual without a label
    product_description_input = st.sidebar.text_area("", placeholder="Enter the product description here...")

    if st.sidebar.button("Process BOQ"):
        # Extract features from the user input
        features = extract_features_from_row(pd.Series({'Unnamed: 0': product_description_input}))

        # Display the extracted features in a container
        with st.container():
            st.subheader("Extracted Features")
            for feature, value in features.items():
                st.write(f"**{feature}:** {value}")

