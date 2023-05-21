import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Citirea datelor de antrenare și test
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separarea caracteristicilor și a variabilei țintă din datele de antrenare
train_x = train_data.drop(columns=['Id', 'Open Date', 'City', 'City Group', 'Type', 'revenue'])
train_y = train_data['revenue']

# Inițializarea și antrenarea modelului de regresie liniară
regressor = LinearRegression()
regressor.fit(train_x, train_y)

# Realizarea predicțiilor pe datele de test
test_x = test_data.drop(columns=['Id', 'Open Date', 'City', 'City Group', 'Type'])
predictions = regressor.predict(test_x)

# Calcularea erorii medii pătratice (MSE) pe datele de antrenare
train_predictions = regressor.predict(train_x)
train_mse = mean_squared_error(train_y, train_predictions)
print("Mean Squared Error (Train):", train_mse)

# Salvarea rezultatelor în fișierul 'sampleSubmission.csv'
submission = pd.DataFrame({'Id': test_data['Id'], 'Prediction': predictions})
submission.to_csv('sampleSubmission2.csv', index=False)
