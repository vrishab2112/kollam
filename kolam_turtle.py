import turtle as T
import math

# Window setup
screen = T.Screen()
screen.title("Kolam Rosette - Turtle")
screen.setup(width=900, height=900)
screen.bgcolor('#6b1f23')  # reddish background similar to photo

pen = T.Turtle(visible=False)
pen.speed(0)
pen.color('white')
pen.pensize(6)
pen.hideturtle()
T.tracer(False)


def move_to(x: float, y: float):
    pen.up(); pen.goto(x, y); pen.down()


def rounded_diamond_grid(n: int = 4, s: float = 32.0, r_factor: float = 0.5):
    """Draw a diamond of n×n squares (rotated 45°) using rounded corners.
    s: square size; r_factor controls corner radius as fraction of s.
    """
    r = s * r_factor
    # Build diamond indices in square lattice coordinates
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            if abs(i) + abs(j) == n:  # border squares only for the outer diamond
                # four corners before rotation
                pts = [
                    (i * s, j * s),
                    ((i+1) * s, j * s),
                    ((i+1) * s, (j+1) * s),
                    (i * s, (j+1) * s)
                ]
                # Draw rounded rectangle edges
                draw_rounded_poly(pts, r)


def draw_rounded_poly(pts, r):
    # pts are axis-aligned square corners; we draw straight segments minus r and quarter arcs
    def L(a, b):
        ax, ay = a; bx, by = b
        vx, vy = bx - ax, by - ay
        Llen = math.hypot(vx, vy)
        ux, uy = vx / Llen, vy / Llen
        sx, sy = ax + ux * r, ay + uy * r
        ex, ey = bx - ux * r, by - uy * r
        pen.up(); pen.goto(sx, sy); pen.down(); pen.goto(ex, ey)

    for k in range(4):
        a = pts[k]
        b = pts[(k+1) % 4]
        c = pts[(k+2) % 4]
        # straight part of side ab
        L(a, b)
        # arc at vertex b from direction ab to bc
        bx, by = b
        start = math.degrees(math.atan2(by - a[1], bx - a[0]))
        end = math.degrees(math.atan2(c[1] - by, c[0] - bx))
        # normalize sweep direction clockwise
        sweep = (end - start) % 360
        if sweep <= 0:
            sweep += 360
        # turtle draws left-turning arcs with circle; use negative radius for clockwise
        pen.up(); pen.goto(bx, by); pen.setheading(start); pen.forward(r); pen.right(90)
        pen.down(); pen.circle(r, sweep)


def rotated_frame(angle_deg: float):
    pen.up(); pen.home(); pen.setheading(0)
    pen.left(angle_deg)


def petals(m: int = 8, R: float = 170.0, r: float = 70.0):
    """Draw m circular petals around a center.
    Each petal is an arc of radius r whose center lies on a circle of radius R.
    """
    for k in range(m):
        theta = 360.0 * k / m
        cx = R * math.cos(math.radians(theta))
        cy = R * math.sin(math.radians(theta))
        move_to(cx, cy - r)
        pen.setheading(0)
        pen.circle(r, 180)


def center_ring(R_outer: float = 35.0, R_inner: float = 14.0):
    pen.up(); pen.goto(0, -R_outer); pen.setheading(0); pen.down(); pen.circle(R_outer)
    pen.up(); pen.goto(0, -R_inner); pen.down(); pen.color('#6b1f23'); pen.circle(R_inner)
    pen.color('white')


def small_dots(m: int = 16, radius: float = 4.0, ring_R: float = 110.0):
    for k in range(m):
        th = 360.0 * k / m
        x = ring_R * math.cos(math.radians(th))
        y = ring_R * math.sin(math.radians(th))
        pen.up(); pen.goto(x, y - radius); pen.down()
        pen.color('white'); pen.begin_fill(); pen.circle(radius); pen.end_fill()


def main():
    rotated_frame(45)  # rotate so the grid is diamond-like
    rounded_diamond_grid(n=4, s=40, r_factor=0.55)
    rotated_frame(0)
    petals(m=12, R=155, r=55)
    center_ring(34, 14)
    small_dots(m=16, radius=4.5, ring_R=112)
    T.tracer(True)
    T.done()


if __name__ == '__main__':
    main()


