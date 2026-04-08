import pandas as pd
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import BIFReader


def load_model(model_path="models/bayesian_model.bif"):
    reader = BIFReader(model_path)
    model = reader.get_model()
    return model


def predict_loan_status(evidence):
    # Load trained model
    model = load_model()

    # Create inference object
    inference = VariableElimination(model)

    # Select only important features for inference
    relevant_evidence = {
    'Income': evidence['Income'],
    'CIBIL': evidence['CIBIL'],
    'Loan_Amount': evidence['Loan_Amount'],
    }

    if 'Assets' in evidence:
        relevant_evidence['Assets'] = evidence['Assets']

    result = inference.query(
    variables=['Loan_Status'],
    evidence=relevant_evidence
    )

    return result


if __name__ == "__main__":
    evidence = {
        'Income': 'Medium',
        'CIBIL': 'Poor',
        'Assets': 'Low',
        'Loan_Amount': 'Medium',
        'Loan_Term': 'Long',
        'Dependents': 'Few',
        'Education': 'Graduate',
        'Self_Employed': 'No'
    }

    result = predict_loan_status(evidence)
    decision_index = result.values.argmax()
    states = result.state_names['Loan_Status']
    confidence = max(result.values)


    print("\n🔮 Prediction Result:")
    print(result)
    print("\n✅ Final Decision:", states[decision_index])
    print(f"📊 Confidence: {confidence:.2f}")