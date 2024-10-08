from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from math import sin, cos, radians

N = 100000
G = 10
DT = 0.01

LINES_TO_CROSS = np.array(
    [
        [[2, 2], [2, 11]],
        [[7, 2], [7, 11]],
    ]
)

LINES_TO_MISS = np.array(
    [
        [[3, 5], [6, 5]],
        [[3, 8], [6, 8]],
    ]
)

alpha_min = 1
alpha_max = 89
velocity_min = 1
velocity_max = 30

alpha = np.random.uniform(alpha_min, alpha_max, N)
velocity = np.random.uniform(velocity_min, velocity_max, N)


def solve_quadratic(a, b, c):
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return []
    sqrt_discriminant = np.sqrt(discriminant)
    x1 = (-b + sqrt_discriminant) / (2 * a)
    x2 = (-b - sqrt_discriminant) / (2 * a)
    return x1, x2


def collide(alpha, velocity, line):
    v_x = velocity * cos(radians(alpha))
    v_y = velocity * sin(radians(alpha))

    x1, y1 = line[0]
    x2, y2 = line[1]

    if x1 == x2:
        t = x1 / v_x
        if t < 0:
            return False
        y = v_y * t - 0.5 * G * t**2
        return min(y1, y2) <= y <= max(y1, y2)
    else:
        # y = ax + b
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        # t = x/v_x
        # y = v_y/v_x * x - (0.5G/v_x**2) * x**2
        # ax + b = v_y/v_x * x - (0.5G/v_x**2) * x**2
        # 0 = - (G/2v_x**2) * x**2 + (v_y/v_x - a) * x  - b

        A = -G / (2 * v_x**2)
        B = (v_y / v_x) - a
        C = -b

        x_min = min(x1, x2)
        x_max = max(x1, x2)

        for x in solve_quadratic(A, B, C):
            if x_min <= x <= x_max:
                t = x / v_x
                if t >= 0:
                    return True

        return False


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for line in LINES_TO_MISS:
    ax1.plot(
        line[:, 0],
        line[:, 1],
        "r",
        linewidth=2,
    )

for line in LINES_TO_CROSS:
    ax1.plot(
        line[:, 0],
        line[:, 1],
        "g",
        linewidth=2,
    )

xmax = max(np.max(LINES_TO_CROSS[:, :, 0]), np.max(LINES_TO_MISS[:, :, 0]))
ymax = max(np.max(LINES_TO_CROSS[:, :, 1]), np.max(LINES_TO_MISS[:, :, 1]))

ax1.set_xlim(0, xmax + 1)
ax1.set_ylim(0, ymax + 1)
ax1.set_xticks(range(int(xmax) + 2))
ax1.set_yticks(range(int(ymax) + 2))

results = []
for a, v in zip(alpha, velocity):
    to_be_all_true = [collide(a, v, line) for line in LINES_TO_CROSS]
    to_be_all_false = [collide(a, v, line) for line in LINES_TO_MISS]
    results.append(all(to_be_all_true) and not any(to_be_all_false))

all_results = np.array(list(zip(alpha, velocity, results)))
valid = np.array([[a, v] for a, v, r in all_results if r])

for a, v in valid[:5]:
    t_max = 2 * v * sin(radians(a)) / G
    T = np.arange(0, t_max + DT, DT)
    X = v * cos(radians(a)) * T
    Y = v * sin(radians(a)) * T - 0.5 * G * T**2
    ax1.plot(X, Y, "b")

ax1.set_title("Projectile Trajectories")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.grid(True)

ax2.scatter(
    valid[:, 0],
    valid[:, 1],
    s=10,
    c="none",
    edgecolors="b",
    linewidths=1,
)

rect_x = alpha_min
rect_y = velocity_min
rect_width = alpha_max - alpha_min
rect_height = velocity_max - velocity_min

# Create a Rectangle patch
rect = Rectangle(
    (rect_x, rect_y), rect_width, rect_height, fill=False, edgecolor="r", linewidth=2
)

# Add the rectangle to ax2
ax2.add_patch(rect)

ax2.set_title("Allowed and forbidden regions")
ax2.set_xlabel(r"$\alpha$ (degrees)")
ax2.set_ylabel(r"$v_0$ (m/s)")

plt.tight_layout()

print(len(valid) / N)

plt.show()

# save figure
fig.savefig("plots.png")
