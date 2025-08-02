# import pandas as pd
# import time
# import pickle
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut

# # --------------------- Load your Data ----------------------
# df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\Capstone Project\Data Preprocessing New\data_recommendation_v2.csv")  # Update path as needed

# # Drop rows where NearbyPlaces is missing
# df.dropna(subset=['NearbyPlaces'], inplace=True)

# # If column is stored as stringified lists, convert them to real lists
# df['NearbyPlaces'] = df['NearbyPlaces'].apply(eval)

# # ------------------- Setup Geolocator -----------------------
# geo_locator = Nominatim(user_agent="property_recommender")

# # -------------------- Load Previous Cache -------------------
# try:
#     with open(r"C:\Users\aryan\Desktop\Capstone Project\Joblib\geo_cache.pkl", "rb") as f:
#         coord_cache = pickle.load(f)
# except FileNotFoundError:
#     coord_cache = {}

# # ------------------ Geocoding Function ----------------------
# def geocode_address(addr):
#     addr = addr.strip().lower()
    
#     if addr in coord_cache:
#         return coord_cache[addr]

#     try:
#         location = geo_locator.geocode(addr + ", Gurgaon, India", timeout=10)
#         if location:
#             coord = (location.latitude, location.longitude)
#             coord_cache[addr] = coord
#             return coord
#     except GeocoderTimedOut:
#         print("Timeout on:", addr)
#     except Exception as e:
#         print(f"Failed: {addr} — {e}")

#     coord_cache[addr] = None
#     return None

# # ------------------ Get Unique Nearby Places ----------------------
# nearby_places = set(
#     place.strip().lower() 
#     for sublist in df['NearbyPlaces'] 
#     for place in sublist if place
# )

# all_addresses = sorted(nearby_places)

# print(f"🧠 Total unique nearby places to geocode: {len(all_addresses)}")

# # ------------------- Geocode All Addresses -------------------
# for i, address in enumerate(all_addresses):
#     if address not in coord_cache or coord_cache[address] is None:
#         print(f"[{i+1}/{len(all_addresses)}] Geocoding: {address}")
#         geocode_address(address)
#         time.sleep(0.5)  # Rate limiting to avoid being blocked

# # ---------------------- Save Cache --------------------------
# with open(r"C:\Users\aryan\Desktop\Capstone Project\Joblib\geo_cache.pkl", "wb") as f:
#     pickle.dump(coord_cache, f)

# print("✅ Geocoding complete. Cache saved to geo_cache.pkl.")


import pickle

with open(r"C:\Users\aryan\Desktop\Capstone Project\Joblib\geo_cache.pkl", "rb") as f:
    cache = pickle.load(f)

print("\n🔍 Sample cache keys:")
for i, k in enumerate(list(cache.keys())[:20]):
    print(f"{i}: {k}")
