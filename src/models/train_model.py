import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.readwrite import BIFWriter
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.models.structure import create_model


def train_model(data_path, save_path="models/bayesian_model.bif"):
    # Load data
    data = pd.read_csv(data_path)

    # 🔥 Train-Test Split
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

    print(f"📊 Training samples: {len(train_data)}")
    print(f"📊 Testing samples : {len(test_data)}")

    # Create model structure
    model = create_model()

    # Train model on TRAIN data
    model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")

    print("✅ Model trained successfully!")

    # Save model
    writer = BIFWriter(model)
    writer.write_bif(save_path)

    print(f"✅ Model saved at {save_path}")

    # 🔥 Evaluate on TEST data
    evaluate_model(model, test_data)

    return model


def evaluate_model(model, data):
    inference = VariableElimination(model)

    y_true = []
    y_pred = []

    for _, row in data.iterrows():
        evidence = {
            'Income': row['Income'],
            'CIBIL': row['CIBIL'],
            'Loan_Amount': row['Loan_Amount'],
            'Assets': row['Assets']
        }

        try:
            result = inference.query(
                variables=['Loan_Status'],
                evidence=evidence,
                show_progress=False
            )

            states = result.state_names['Loan_Status']
            pred = states[result.values.argmax()]

            y_true.append(row['Loan_Status'])
            y_pred.append(pred)

        except:
            continue

    # 🔥 Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Approved')
    recall = recall_score(y_true, y_pred, pos_label='Approved')
    f1 = f1_score(y_true, y_pred, pos_label='Approved')

    print("\n📊 Model Evaluation Metrics (Test Data):")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # 🔥 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=['Approved', 'Rejected'])

    print("\n📊 Confusion Matrix:")
    print("                Predicted")
    print("              Approved  Rejected")
    print(f"Actual Approved   {cm[0][0]:<8}  {cm[0][1]:<8}")
    print(f"Actual Rejected   {cm[1][0]:<8}  {cm[1][1]:<8}")


if __name__ == "__main__":
    train_model("data/loan_bn_ready.csv")