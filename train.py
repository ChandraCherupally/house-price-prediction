from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd
import joblib
from pathlib import Path
import logging 


# --------------------------------------------------------------
# Log configuration
# --------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"


logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

log = logging.getLogger("house-price-prediction")

# --------------------------------------------------------------
# train - test split
# --------------------------------------------------------------

log.info("loading datasets")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

log.info(f"total records: {X.shape[0]}")
log.info("train and test splitting of data started....")
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
log.info("train and test splitting of data completed....")


# --------------------------------------------------------------
# Model training
# --------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)

log.info(f"average error: ${mae*100000:,.0f}")
log.info(f"R2 score: {r2:,.3f}")

joblib.dump(model, "house_model.joblib")
joblib.dump(list(X.columns), "house_features.joblib")
