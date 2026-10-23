
import os
from pathlib import Path

import joblib
import mysql.connector
from mysql.connector import Error
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "email_fraud_model.pkl"
CA_PATH = BASE_DIR / "isrgrootx1.pem"

# TiDB Cloud configuration
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get(
        "DB_NAME", "email_fraud_detection"
    ),
    "port": int(os.environ.get("DB_PORT", "4000")),
    "ssl_ca": str(CA_PATH),
    "ssl_verify_cert": True,
}

# Load the trained model
model = joblib.load(MODEL_PATH)


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


# Save email analysis
def save_analysis(email_text, prediction, risk_level):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO analysis_history
            (email_text, prediction, risk_level)
            VALUES (%s, %s, %s)
            """,
            (email_text, prediction, risk_level),
        )

        connection.commit()
        print("Analysis saved successfully!")

    except Error as e:
        print("Database save error:", e)

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# Home page and email analysis
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    email_text = ""
    error = None

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()

        if not email_text:
            error = "Please enter an email to analyze."
        else:
            prediction = int(model.predict([email_text])[0])

            if prediction == 1:
                category = "Potentially Spam / Phishing"
                risk = "High - Review Carefully"
            else:
                category = "Likely Legitimate"
                risk = "Lower Risk - Still Verify"

            result = {
                "category": category,
                "risk": risk,
            }

            save_analysis(email_text, category, risk)

    return render_template(
        "index.html",
        result=result,
        email_text=email_text,
        error=error,
    )


# Analysis history page
@app.route("/history")
def history():
    connection = None
    cursor = None
    records = []
    history_error = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, email_text, prediction, risk_level, analyzed_at
            FROM analysis_history
            ORDER BY id DESC
            """
        )

        records = cursor.fetchall()

    except Error as e:
        history_error = "Unable to load history. Please check the database."
        print("History database error:", e)

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

    return render_template(
        "history.html",
        records=records,
        history_error=history_error,
    )


# Analytics dashboard
@app.route("/dashboard")
def dashboard():
    connection = None
    cursor = None

    stats = {
        "total": 0,
        "phishing": 0,
        "legitimate": 0,
    }

    try:
        connection = get_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(
                    CASE
                        WHEN prediction = 'Potentially Spam / Phishing'
                        THEN 1 ELSE 0
                    END
                ), 0),
                COALESCE(SUM(
                    CASE
                        WHEN prediction = 'Likely Legitimate'
                        THEN 1 ELSE 0
                    END
                ), 0)
            FROM analysis_history
            """
        )

        row = cursor.fetchone()

        stats = {
            "total": row[0],
            "phishing": row[1],
            "legitimate": row[2],
        }

    except Error as e:
        print("Dashboard database error:", e)

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

    return render_template("dashboard.html", stats=stats)


if __name__ == "__main__":
    app.run(debug=True)
