# recommender.py

import pickle
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Load model safely
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError("❌ model.pkl not found in project folder")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# Get expected feature names (SAFE)
# --------------------------------------------------
if hasattr(model, "feature_names_in_"):
    EXPECTED_FEATURES = list(model.feature_names_in_)
else:
    # fallback for pipelines / old sklearn
    EXPECTED_FEATURES = None


# --------------------------------------------------
# Default values (Ames Housing safe defaults)
# --------------------------------------------------
DEFAULTS = {
    "LotFrontage": 0,
    "LotArea": 0,
    "OverallQual": 5,
    "OverallCond": 5,
    "YearBuilt": 2000,
    "YearRemodAdd": 2000,
    "MasVnrArea": 0,
    "BsmtFinSF1": 0,
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    "TotalBsmtSF": 0,
    "1stFlrSF": 0,
    "2ndFlrSF": 0,
    "GrLivArea": 0,
    "FullBath": 1,
    "HalfBath": 0,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "TotRmsAbvGrd": 6,
    "Fireplaces": 0,
    "GarageYrBlt": 2000,
    "GarageCars": 0,
    "GarageArea": 0,
    "WoodDeckSF": 0,
    "OpenPorchSF": 0,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2010,

    # categorical
    "MSSubClass": "20",
    "MSZoning": "RL",
    "Street": "Pave",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "BsmtFinType2": "Unf",
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "KitchenQual": "TA",
    "Functional": "Typ",
    "FireplaceQu": "None",
    "GarageType": "Attchd",
    "GarageFinish": "Unf",
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "SaleType": "WD",
    "SaleCondition": "Normal"
}


# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_house_price(
    GrLivArea,
    OverallQual,
    YearBuilt,
    FullBath,
    GarageCars,
    GarageArea,
    LotArea,
    Neighborhood
):
    # If model expects fixed features
    if EXPECTED_FEATURES:
        data = {col: DEFAULTS.get(col, 0) for col in EXPECTED_FEATURES}
    else:
        data = DEFAULTS.copy()

    # Override user inputs
    data.update({
        "GrLivArea": GrLivArea,
        "OverallQual": OverallQual,
        "YearBuilt": YearBuilt,
        "FullBath": FullBath,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "LotArea": LotArea,
        "Neighborhood": Neighborhood
    })

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    return round(float(prediction), 2)
