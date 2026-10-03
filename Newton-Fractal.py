import numpy as np
import matplotlib.pyplot as plt


# Asking for a simple polynomial in normal notation.
print("Newton Fractal Generator")
print("Enter a simple polynomial using z and ^ for powers.")
print("For example: z^3 - 1")

equation = input("f(z) = ").replace(" ", "")


# Splitting the polynomial into separate terms.
# Turning - into +- makes negative terms easy to separate as well.
terms = equation.replace("-", "+-").split("+")
powers = {}


# Reading the coefficient and power from each term.
for term in terms:

    if not term:
        continue

    # Anything without z is a constant, so its power is 0.
    if "z" not in term:
        coefficient = float(term)
        power = 0

    else:
        coefficient = term.split("z")[0]

        # z and -z have implied coefficients of 1 and -1.
        if coefficient == "":
            coefficient = 1
        elif coefficient == "-":
            coefficient = -1
        else:
            coefficient = float(coefficient)

        # z on its own has a power of 1.
        # Otherwise the number after ^ gives the power.
        power = int(term.split("^")[1]) if "^" in term else 1

    powers[power] = powers.get(power, 0) + coefficient


# Converting the polynomial into a coefficient array.
# z^3 - 2z + 4 becomes [1, 0, -2, 4].
degree = max(powers)

coefficients = np.array(
    [powers.get(p, 0) for p in range(degree, -1, -1)],
    dtype=float
)


# Applying the power rule to get the derivative.
# az^n becomes na z^(n-1).
derivative = coefficients[:-1] * np.arange(degree, 0, -1)


# Turning the coefficient arrays into functions.
f = np.poly1d(coefficients)
df = np.poly1d(derivative)


# Finding the roots so the final points can be sorted
# by which root they converged towards.
roots = np.roots(coefficients)


print("\nf(z) =")
print(f)

print("\nf'(z) =")
print(df)

print("\nRoots:")
print(roots)


# Creating a 500 x 500 grid across the complex plane.
x = np.linspace(-2, 2, 500)
y = np.linspace(-2, 2, 500)

X, Y = np.meshgrid(x, y)

# Combining the real and imaginary coordinates into z = x + iy.
Z = X + 1j * Y


# Applying Newton's equation 30 times to every point in the grid.
#
#                 f(z_n)
# z_(n+1) = z_n - -------
#                 f'(z_n)
#
# NumPy applies the equation to the whole grid at once.
for i in range(30):

    # Some points can have f'(z) = 0, which causes division by zero.
    # The warnings can be ignored while the other points continue.
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = Z - f(Z) / df(Z)


# Comparing every final value with each root.
# The closest root decides which basin the starting point belongs to.
distances = np.abs(Z[..., None] - roots)

fractal = np.argmin(distances, axis=2)


# Displaying the Newton fractal.
fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(
    fractal,
    extent=[-2, 2, -2, 2],
    origin="lower"
)

ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")
ax.set_title("Newton Fractal for f(z) = " + equation)

plt.show()

# Leaving the figure as the final output also helps it display in Juno.
fig