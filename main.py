import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python main.py <study_hours>")
    sys.exit(1)

try:
    input_x = float(sys.argv[1])
except ValueError:
    print("Please enter a valid number for study hours.")
    sys.exit(1)

data = pd.read_csv('data.csv')

def loss_function(m, b, points):
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        
        total_error += (y - (m * x + b)) ** 2
    
    total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

m = 0
b = 0
L = 0.00001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

print(m, b)

predicted_y = m * input_x + b + 2
predicted_y = max(0, min(10, predicted_y))
print(f"If you studied {input_x:.1f} minutes. You will get : {predicted_y:.2f}")

plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(0, 300)), [m * x + b for x in range(0, 300)], color="red")
plt.show()