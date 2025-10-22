def perceptron(x1, x2):
    w1 = -1
    w2 = -1
    b = 1
    z = w1 * x1 + w2 * x2 + b
    if z >= 0:
        return 1
    else:
        return 0


inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
print("A B | NOT(A AND B)")
for x1, x2 in inputs:
    y = perceptron(x1, x2)
    print(f"{x1} {x2} | {y}")
