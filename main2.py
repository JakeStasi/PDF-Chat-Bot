from question_system import setup_qa_system
from flask import Flask, render_template, redirect, url_for, request, session
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
qa_chain = None


@app.route("/", methods=["GET", "POST"])
def index():
    global qa_chain
    answer = None
    upload_success = False

    if "history" not in session:
        session["history"] = []

    if "pdf_uploaded" not in session:
        session["pdf_uploaded"] = False

    if request.method == "POST":
        # Handle PDF upload
        if 'pdf' in request.files and request.files['pdf'].filename:
            pdf = request.files['pdf']
            pdf.save("static/uploaded.pdf")  # Save to static so it can be served
            qa_chain = setup_qa_system("static/uploaded.pdf")
            session["history"] = []
            session["pdf_uploaded"] = True
            upload_success = True

        # Handle question
        elif 'question' in request.form:
            question = request.form['question']
            if qa_chain:
                answer = qa_chain.run(question)
                session["history"].append((question, answer))
                session.modified = True
            else:
                answer = "❗ Please upload a PDF first."

    return render_template(
        "practice.html",
        answer=answer,
        history=session["history"],
        upload_success=upload_success,
        pdf_uploaded=session["pdf_uploaded"]
    )

@app.route("/reset", methods=["POST"])
def reset():
    global qa_chain
    qa_chain = None
    session.clear()
    try:
        os.remove("static/uploaded.pdf")
    except FileNotFoundError:
        pass
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
