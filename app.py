import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Al-Mizan: House Price Predictor", page_icon="🏠", layout="wide")

LOCATIONS = ["Jakarta Selatan", "Jakarta Barat", "Depok"]
LOCATION_MULTIPLIER = {"Jakarta Selatan": 3.0, "Jakarta Barat": 1.8, "Depok": 1.0}
RANDOM_STATE = 42
N_SAMPLES = 200

# ==========================================
# DATA GENERATION (synthetic)
# ==========================================
@st.cache_data
def generate_data():
    rng = np.random.default_rng(RANDOM_STATE)
    locations = rng.choice(LOCATIONS, size=N_SAMPLES)
    area = rng.integers(40, 300, size=N_SAMPLES)
    bedrooms = rng.integers(1, 6, size=N_SAMPLES)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, size=N_SAMPLES), 1, None).astype(int)

    multiplier = np.array([LOCATION_MULTIPLIER[loc] for loc in locations])
    noise = rng.normal(0, 80, size=N_SAMPLES)
    price = (area * 8 + bedrooms * 120 + bathrooms * 60) * multiplier + noise
    price = np.clip(price, 100, None)

    return pd.DataFrame({
        "Area (m2)": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Location": locations,
        "Price (Million Rp)": price.round(0),
    })

# ==========================================
# MODEL TRAINING (cached per model choice)
# ==========================================
@st.cache_resource
def train_model(model_name: str):
    df = generate_data()
    X = pd.get_dummies(df.drop(columns=["Price (Million Rp)"]),
                       columns=["Location"], drop_first=True)
    y = np.log(df["Price (Million Rp)"])  # log target stabilizes multiplicative structure

    feature_columns = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    else:  # Linear Regression needs scaling
        model = make_pipeline(StandardScaler(), LinearRegression())

    model.fit(X_train, y_train)

    # Metrics back on the Rupiah scale
    pred_test = np.exp(model.predict(X_test))
    true_test = np.exp(y_test)
    mae = mean_absolute_error(true_test, pred_test)
    r2 = r2_score(true_test, pred_test)

    return model, feature_columns, mae, r2

# ==========================================
# UI
# ==========================================
st.title("🏠 Al-Mizan: House Price Predictor")
st.caption("Compare Random Forest vs Linear Regression on synthetic Jakarta housing data.")

# --- Model toggle ---
model_name = st.radio("Choose a model", ["Random Forest", "Linear Regression"], horizontal=True)
model, feature_columns, mae, r2 = train_model(model_name)

col_a, col_b = st.columns(2)
col_a.metric("MAE (Million Rp)", f"{mae:,.1f}")
col_b.metric("R² Score", f"{r2:.3f}")

st.divider()

# --- Prediction form ---
st.subheader("Predict a price")
c1, c2, c3, c4 = st.columns(4)
area_input = c1.number_input("Area (m²)", 20, 500, 100)
bedrooms = c2.number_input("Bedrooms", 1, 10, 3)
bathrooms = c3.number_input("Bathrooms", 1, 10, 2)
location_input = c4.selectbox("Location", LOCATIONS)

if st.button("Predict", type="primary"):
    df_user = pd.DataFrame([{
        "Area (m2)": area_input,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Location": location_input,
    }])
    df_user = pd.get_dummies(df_user, columns=["Location"], drop_first=True)
    df_user = df_user.reindex(columns=feature_columns, fill_value=0)

    pred = np.exp(model.predict(df_user))[0]
    st.success(f"Estimated price: **Rp {pred:,.0f} million** (using {model_name})")

st.divider()

# --- Transparency panel ---
with st.expander("🔍 Model transparency"):
    if model_name == "Random Forest":
        importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
        fig, ax = plt.subplots()
        importances.plot.barh(ax=ax, color="#4c78a8")
        ax.set_title("Feature Importances")
        st.pyplot(fig)
        plt.close(fig)
    else:
        coefs = pd.Series(model.named_steps["linearregression"].coef_,
                          index=feature_columns).sort_values()
        fig, ax = plt.subplots()
        coefs.plot.barh(ax=ax, color="#f58518")
        ax.set_title("Linear Coefficients (on scaled, log-price)")
        st.pyplot(fig)
        plt.close(fig)

# --- Data distribution ---
df = generate_data()
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Area (m2)", y="Price (Million Rp)",
                hue="Location", palette="viridis", ax=ax)
ax.set_title("Actual House Price Distribution")
st.pyplot(fig)
plt.close(fig)

st.markdown("---")
st.info("💡 **Portfolio tip:** Push this `app.py` to GitHub and deploy free on "
        "**Streamlit Community Cloud** to share your work live.")
