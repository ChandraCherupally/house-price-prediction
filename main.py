import io
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

# ------------------------------------
# Load Model once when server started
# ------------------------------------

model = joblib.load("\data\house_model.joblib")
features =  joblib.load("\data\house_features.joblib")

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

#home
@app.get("/")
def home():
    return {
        "message" : "California house price prediction api",
        "status" : "running",
        "endpoint" : "send post request to /predict"

    }

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
    
    print(df.head())

    required_columns = [
        "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    print(missing_columns)
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



    
