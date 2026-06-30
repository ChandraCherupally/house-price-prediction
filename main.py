import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path

#creating object for FastAPI
app = FastAPI(
    title="California House Price Prediction API",
    description="Predict California house prices using a Random Forest Regressor.",
    version="0.1.0",
)

# Allow the browser frontend (file:// or any localhost port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# Load Model once when server started
# ------------------------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "data/house_model.joblib")
features =  joblib.load(BASE_DIR / "data/house_features.joblib")

#input schema

class HouseFeatures(BaseModel):
    MedInc :     float = Field(gt=0, description="Medium Income of Neibhour hood")
    HouseAge :   float = Field(gt=0, description="Average Age of Household")
    AveRooms :   float = Field(gt=0, description="Average rooms in the House")
    AveBedrms :  float = Field(gt=0, description="Average bedrooms in the House")
    Population : float = Field(gt=0, description="Average population of the area")
    AveOccup :   float = Field(gt=0, description="Average members in the house")
    Latitude :   float = Field(ge=32, le=42, description="Lattitude of the area")
    Longitude :  float = Field(ge=-125,le=-114, description="Langitude of the area")

#home — serves the frontend UI
@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(BASE_DIR / "index.html")



#Health check
@app.get("/health")
def health():
    return{
        "status" :"running",
        "model" : "RandomForestRegressor",
        "features" : features,
        "avg_error" : "$32,773",
        "R2 Score" : "0.805",
    }

#Prediction
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc" :  house.MedInc,
            "HouseAge" : house.HouseAge,
            "AveRooms" : house.AveRooms,
            "AveBedrms" : house.AveBedrms,
            "Population" : house.Population,
            "AveOccup" : house.AveOccup,
            "Latitude" : house.Latitude,
            "Longitude" : house.Longitude
        }])
        
        predicted = model.predict(input_data)[0]
        price_usd = predicted*100000

        return {
            "predicted_price" : f"${price_usd:,.0f}",
            "predicted_price_short" : f"${predicted:.2f} hundred thousands",
            "fidence range" : f"${price_usd-32773:,.0f} to ${price_usd+32773:,.0f}"
        }



    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )



@app.post("/predictfile")
async def predict_file(file: UploadFile = File(...)):
    ## Exception 1
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="please uplaod csv file only"
        )
    
    contents = await file.read()
    #b'name,age'\nSagar,25\nRahul,35
    df = pd.read_csv(io.BytesIO(contents))

    required_columns = [
        "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    ## Exception 2
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are  missing from your file {missing_columns}"
        )
    
    ## Exception 3
    if len(df)==0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file has no data rows"
        )

    ######### Predicting the "Predicted Price", 
    try:
        df["predicted_columns_usd"] = model.predict(df[required_columns])
        df["predicted_columns_usd"] = df["predicted_columns_usd"].apply(lambda x:f"${x*100000:,.0f}")
        output = df.to_csv(index=False)

        return StreamingResponse(
             io.StringIO(output),
             media_type="text/csv",
             headers={
                 "Content-Disposition":"attachment;filename=predictions.csv"
             }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )



    
