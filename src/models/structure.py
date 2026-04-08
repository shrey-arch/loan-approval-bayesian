from pgmpy.models import DiscreteBayesianNetwork

def create_model():
    model = DiscreteBayesianNetwork([
        ('Education', 'Income'),
        ('Self_Employed', 'Income'),

        ('Income', 'Loan_Amount'),
        ('Income', 'Loan_Status'),

        ('CIBIL', 'Loan_Status'),
        ('Assets', 'Loan_Status'),
        ('Dependents', 'Loan_Status'),
        ('Loan_Amount', 'Loan_Status'),
        ('Loan_Term', 'Loan_Status')
    ])

    return model