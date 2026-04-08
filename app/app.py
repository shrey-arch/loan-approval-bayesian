from flask import Flask, render_template, request
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader

app = Flask(__name__)

# Load model once
reader = BIFReader("models/bayesian_model.bif")
model = reader.get_model()
inference = VariableElimination(model)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    decision = None

    if request.method == "POST":
        # Get inputs from form
        income = request.form["Income"]
        cibil = request.form["CIBIL"]
        loan_amount = request.form["Loan_Amount"]
        assets = request.form["Assets"]
        loan_term = request.form["Loan_Term"]
        dependents = request.form["Dependents"]
        education = request.form["Education"]
        self_employed = request.form["Self_Employed"]

        # Relevant evidence
        evidence = {
            'Income': income,
            'CIBIL': cibil,
            'Loan_Amount': loan_amount,
            'Assets': assets
        }

        # Inference
        query = inference.query(
            variables=['Loan_Status'],
            evidence=evidence
        )

        probs = query.values
        states = query.state_names['Loan_Status']

        # Extract probabilities (raw)
        approved = probs[states.index('Approved')]
        rejected = probs[states.index('Rejected')]

        # Rounded for display
        result = {
            "Approved": round(approved, 2),
            "Rejected": round(rejected, 2)
        }

        # 🔥 Decision logic with manual review
        if abs(approved - rejected) < 0.05:
            decision = "Manual Review Required"
        elif approved > rejected:
            decision = "Approved"
        else:
            decision = "Rejected"


    return render_template(
       "index.html",
        result=result,
        decision=decision,
        form_data=request.form
    )



if __name__ == "__main__":
    app.run(debug=True)