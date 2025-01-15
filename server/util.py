import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

__location = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0],2)

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __location

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __location = __data_columns[3:]

    global __model 
    with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __location

def get_data_columns():
    return __data_columns

# sqft (Square Feet), bhk (Bedroom, Hall, Kitchen), bath (Bathroom)
# 1 lakh = Rp 20.535.000

if __name__ == "__main__":
    load_saved_artifacts()
    print("Note : sqft (Square Feet), bhk (Bedroom, Hall, Kitchen), bath (Bathroom), 1 lakh = Rp 20.535.000")
    print(get_location_names())
    print(get_estimated_price('Yelahanka New Town', 1500, 4, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2)) # other location
