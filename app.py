import streamlit as st
import joblib
import numpy as np

model = joblib.load("Diabetset.pkl")

st.title("ระบบพยากรณ์โรคเบาหวาน")

st.subheader("ข้อมูลพื้นฐาน")

# อายุ
age = st.number_input("อายุ (ปี) (ถ้ามี)", min_value=0, max_value=120, value=None, placeholder="กรอกถ้ามี")

# น้ำหนัก/ส่วนสูง → BMI คำนวณเฉพาะเมื่อกรอกครบ
weight = st.number_input("น้ำหนัก (kg) (ถ้ามี)", min_value=0.0, max_value=200.0, value=None, placeholder="กรอกถ้ามี")
height = st.number_input("ส่วนสูง (cm) (ถ้ามี)", min_value=0.0, max_value=220.0, value=None, placeholder="กรอกถ้ามี")

if weight and height:
    bmi = weight / ((height/100)**2)
    st.write(f"**BMI ของคุณ:** {bmi:.2f}")
else:
    bmi = None


# Glucose
glucose = st.number_input(
    "ระดับน้ำตาล (Glucose) (ถ้ามี)",
    min_value=0,
    max_value=300,
    value=None,
    placeholder="กรอกถ้ามี"
)

# Blood Pressure
bp = st.number_input(
    "ความดันโลหิต (Diastolic BP) (ถ้ามี)",
    min_value=0,
    max_value=200,
    value=None,
    placeholder="กรอกถ้ามี"
)

# SkinThickness
skin = st.number_input(
    "Skin Thickness (mm) (ถ้ามี)",
    min_value=0,
    max_value=100,
    value=None,
    placeholder="กรอกถ้ามี"
)

# Insulin
insulin = st.number_input(
    "ระดับ Insulin (ถ้ามี)",
    min_value=0,
    max_value=400,
    value=None,
    placeholder="กรอกถ้ามี"
)


# DPF
st.subheader("ประวัติครอบครัว (ถ้ามี)")
family = st.selectbox(
    "เลือกสถานะ (ถ้ามี)",
    ["", "ไม่มีเลย", "มีญาติห่างๆเป็น", "มีพ่อ/เเม่/พี่น้องเป็น1คน", "มีหลายคนในครอบครัวเป็น", "พ่อ/เเม่เป็นทั้งคู่ หรือหลายคนในครอบครัวใกล้ชิด"]
)

DPF_MAP = {
    "ไม่มีเลย": 0.1,
    "มีญาติห่างๆเป็น": 0.3,
    "มีพ่อ/เเม่/พี่น้องเป็น1คน": 0.6,
    "มีหลายคนในครอบครัวเป็น": 1.0,
    "พ่อ/เเม่เป็นทั้งคู่ หรือหลายคนในครอบครัวใกล้ชิด": 2.0,
}

dpf = DPF_MAP.get(family, None)


# --------------------------------------------------
# ปุ่มคำนวณ
# --------------------------------------------------
if st.button("คำนวณความเสี่ยง"):

    # ทำ array จากค่าที่กรอก (ไม่มี = np.nan)
    features = np.array([[ 
        glucose if glucose is not None else np.nan,
        bmi if bmi is not None else np.nan,
        age if age is not None else np.nan,
        bp if bp is not None else np.nan,
        insulin if insulin is not None else np.nan,
        dpf if dpf is not None else np.nan,
        skin if skin is not None else np.nan,
    ]])

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.subheader("ผลการประเมิน")

    if prediction == 1:
        st.error(f"⚠ คุณมีความเสี่ยงสูงเป็นเบาหวาน ({prob*100:.2f}%)")
    else:
        st.success(f"คุณมีความเสี่ยงต่ำ ({prob*100:.2f}%)")






