# LINEAR REGRESSION by Damian Diaz
# last edited 5/31/19
#
# https://github.com/damiandiaz212


# get gradient of the intercept
def get_gradient_at_b(x, y, b, m):
    N = len(x)
    SUM = 0
    for i in range(N):
        SUM += (y[i] - ((m * x[i]) + b))
    b_gradient = -(2 / N) * SUM
    return b_gradient


# get gradient of the slope
def get_gradient_at_m(x, y, b, m):
    N = len(x)
    SUM = 0
    for i in range(N):
        SUM += x[i] * (y[i] - ((m * x[i]) + b))
    m_gradient = -(2 / N) * SUM
    return m_gradient


# step function for gradient descent
def step_gradient(b_current, m_current, x, y, learning_rate):
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]


# gradient descent function
def gradient_descent(x, y, learning_rate, num_iterations):
    b = 0
    m = 0
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return [b, m]
