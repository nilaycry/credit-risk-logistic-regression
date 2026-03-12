import csv
import random
import os

os.makedirs('data', exist_ok=True)
with open('data/credit_risk_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['CompanyID','DebtRatio','ProfitMargin','CurrentRatio','RetainedEarnings','Defaulted'])
    
    for i in range(1, 1001):
        debt = random.uniform(0.1, 0.9)
        profit = random.uniform(-0.1, 0.3)
        current = random.uniform(0.5, 2.5)
        retained = random.uniform(-0.5, 0.8)
        
        # High debt increases risk, high profit/current/retained decreases risk
        # This will create a dataset that Logistic Regression can learn
        score = 8.0 * debt - 15.0 * profit - 2.0 * current - 3.0 * retained
        
        prob = 1.0 / (1.0 + 2.71828**(-score))
        
        # Add some stochasticity
        defaulted = 1 if (random.random() < prob) else 0
        
        writer.writerow([i, debt, profit, current, retained, defaulted])

print("Successfully generated data/credit_risk_dataset.csv with 1000 records!")
