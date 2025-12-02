import json
import logging
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
logging.debug(f"script_dir: {script_dir}")


def get_db(db_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, f"{db_name}/{db_name}.json")
    try:
        with open(file_path, "r") as f:
            db_schema = json.load(f)
    except FileNotFoundError:
        logging.error(f"Schema file not found: {file_path}")
        return None
    except PermissionError:
        logging.error(f"Permission denied when accessing file: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in schema file {file_path}: {e}")
        return None
    return db_schema


# sql-eval datasets
academic = get_db("academic")
advising = get_db("advising")
atis = get_db("atis")
geography = get_db("geography")
restaurants = get_db("restaurants")
scholar = get_db("scholar")
yelp = get_db("yelp")

# sql-eval-instruct datasets
broker = get_db("broker")
car_dealership = get_db("car_dealership")
derm_treatment = get_db("derm_treatment")
ewallet = get_db("ewallet")

dbs = {
    # sql-eval datasets
    "academic": academic,
    "advising": advising,
    "atis": atis,
    "geography": geography,
    "restaurants": restaurants,
    "scholar": scholar,
    "yelp": yelp,
    # sql-eval-instruct datasets
    "broker": broker,
    "car_dealership": car_dealership,
    "derm_treatment": derm_treatment,
    "ewallet": ewallet,
}
