import sys
import os
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import cv2
from model_utils import recognize_face  # Must exist in the same folder

app = FastAPI()

# ---------------------- CORS ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXCEL_FILE = "students.xlsx"
DATASET_FOLDER = "dataset"  # Folder containing training images for students

# ---------------------- Helper: Get all student names from dataset ----------------------
def get_all_dataset_names():
    if not os.path.exists(DATASET_FOLDER):
        return []
    names = [name for name in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, name))]
    return names

# ---------------------- Helper: Ensure Excel file exists and is valid ----------------------
def create_or_update_excel():
    dataset_names = get_all_dataset_names()
    if os.path.exists(EXCEL_FILE):
        try:
            df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
        except Exception:
            df = pd.DataFrame(columns=["Name"])
    else:
        df = pd.DataFrame(columns=["Name"])
    
    # Add any new names from dataset that are not in Excel
    for name in dataset_names:
        if name not in df["Name"].values:
            df = pd.concat([df, pd.DataFrame({"Name": [name], "Attendance": ["Absent"], "Percentage": [0.0]})], ignore_index=True)

    # Ensure Attendance and Percentage columns exist
    if "Attendance" not in df.columns:
        df["Attendance"] = "Absent"
    if "Percentage" not in df.columns:
        df["Percentage"] = 0.0

    # Save back
    df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
    return df

# ---------------------- Root ----------------------
@app.get("/")
def root():
    return {"message": "✅ Face Recognition API running successfully!"}

# ---------------------- Recognize Face ----------------------
@app.post("/recognize/")
async def recognize(
    file: UploadFile = File(...),
    date: str = Form(...),
    time_slot: str = Form(...),
):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        student_name = recognize_face(rgb_image)

        # Read or create Excel
        df = create_or_update_excel()

        # Update attendance for current date & time slot
        col_name = f"{date}_{time_slot}"
        if col_name not in df.columns:
            df[col_name] = "Absent"

        if student_name in df["Name"].values:
            df.loc[df["Name"] == student_name, col_name] = "Present"

        # Update Attendance and Percentage safely
        attendance_cols = [c for c in df.columns if "_" in c]
        if attendance_cols:
            df["Attendance"] = df[attendance_cols].apply(
                lambda row: "Present" if "Present" in row.values else "Absent", axis=1
            )
            df["Percentage"] = df[attendance_cols].apply(
                lambda row: float(np.mean(row == "Present") * 100) if len(row) > 0 else 0.0, axis=1
            )
        else:
            df["Attendance"] = "Absent"
            df["Percentage"] = 0.0

        # ---------------------- Sanitize Percentage ----------------------
        df["Percentage"] = df["Percentage"].apply(lambda x: 0.0 if (pd.isna(x) or not math.isfinite(x)) else float(x))

        df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
        return {"result": student_name}

    except Exception as e:
        return {"error": str(e)}

# ---------------------- Reset Attendance ----------------------
@app.post("/reset-attendance/")
def reset_attendance():
    try:
        df = create_or_update_excel()
        for col in df.columns:
            if "_" in col:
                df[col] = "Absent"
        df["Attendance"] = "Absent"
        df["Percentage"] = 0.0

        # ---------------------- Sanitize Percentage ----------------------
        df["Percentage"] = df["Percentage"].apply(lambda x: 0.0 if (pd.isna(x) or not math.isfinite(x)) else float(x))

        df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
        return {"message": "✅ Attendance reset successfully!"}
    except Exception as e:
        return {"error": str(e)}

# ---------------------- Download Attendance ----------------------
@app.get("/download-attendance/")
def download_attendance():
    return FileResponse(
        EXCEL_FILE,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=EXCEL_FILE,
    )

# ---------------------- View Attendance ----------------------
@app.get("/view-attendance/")
def view_attendance():
    try:
        df = create_or_update_excel()

        # ---------------------- Sanitize all data for JSON ----------------------
        # Convert all numeric columns to native float and replace NaN/inf
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].apply(lambda x: float(x) if (pd.notna(x) and math.isfinite(x)) else 0.0)

        # Convert all remaining columns to Python native types
        data = df.where(pd.notna(df), None).to_dict(orient="records")
        for record in data:
            for key, value in record.items():
                # Ensure Python native types
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    record[key] = float(value)
                elif pd.isna(value):
                    record[key] = None

        return {"attendance": data}

    except Exception as e:
        return {"error": str(e)}

