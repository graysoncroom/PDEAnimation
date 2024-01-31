from manim import *

import numpy as np
from scipy.optimize import fsolve

def u(t, x):
    return 1 / (1 + ((beta_inverse(beta(x) - t) + 3)**2))

def c(t, x):
    return 1 / (x**2 + 1)

def beta(x):
    return (1/3) * x**3 + x

def beta_inverse(y):
    def fixed_point_fn(x, *args):
        return beta(x) - y
    x_guess = np.sign(y)
    x_inverse = fsolve(fixed_point_fn, x_guess, args=(y))
    return x_inverse[0]

CONFIG = {
    "include_numbers": True,
    "color": GREEN,
}

class PDEAnimation(Scene):
    def construct(self):

        # The array Y is the set of points we are tracking along each
        # characteristic curve.
        Y = []

        # This is the number of Ys we want to track
        # Change this and the animation will generate with
        # a different number of tracked points.
        number_of_y_points = 7

        # The interval in which we place our evenly spaced initial Ys
        Y_bounds_start = -4
        Y_bounds_end = 2

        # Generate evenly spaced Ys
        for i in range(number_of_y_points):
            delta = i*(Y_bounds_end - Y_bounds_start + 1) 
            Y.append(Y_bounds_start + delta / number_of_y_points)

        # Comes from u(t,x) = f(x)
        # where x(t) = \beta^{-1}(t + \beta(x))
        # in lecture 4
        def x(t, x0):
            return beta_inverse(t + beta(x0))

        # We create a new array X to build and destroy as the original
        # points coming from Y move along the x-axis in the graph of
        # x vs. u(t,x)
        X = []
        for y_i in Y:
            X.append(x(0, y_i))
 
        # Define the axes for the graphs
        # axes1 - axes for x vs. u(t,x) graph
        axes1 = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1, .1],
            x_length=6,
            #y_length=6,
            tips=False,
            axis_config=CONFIG
        )

        # cc_axes - axes for t vs. x characteristic curve graphs
        cc_axes = Axes(
            x_range=[0, 13, 1],
            y_range=[-5, 5, 1],
            x_length=8,
            tips=False,
            axis_config=CONFIG
        )

        # TODO: Deal with positioning / scaling and alignment of labels
        # in a cleaner way

        label1 = Tex(r"x vs. $u(t,x) = \frac{1}{1 + [\beta^{-1}(\frac{1}{3}x^3 + x - t) + 3]}$", font_size=20)
        label1.to_edge(7*DOWN)
        cc_label = Tex(r"t vs. x", font_size=20)
        cc_label.to_edge(10*UP)

        # Scale / Position the axes
        axes1.scale(0.6)
        cc_axes.scale(0.6)
        axes1.to_edge(8*DOWN)
        cc_axes.to_edge(9*UP)

        graph1 = axes1.plot(lambda x: u(0,x), color=WHITE)

        Y_dots = []
        for y_i in Y:
            Y_dots.append(Dot(axes1.coords_to_point(y_i, 0), color = GREEN))

        Y_sticks = []
        for i in range(len(Y)):
            Y_sticks.append(
                DashedLine(
                    Y_dots[i].get_center(), 
                    Dot(
                        axes1.coords_to_point(Y[i], u(0, Y[i]))
                    ).get_center(),
                    stroke_width = 2
                )
            ) 

        graph1_group = VGroup(
            graph1,
            label1,
            *Y_dots,
            *Y_sticks
        )

        # Recall that Y is the array of initial points
        # not the array of moving points (X / Xcc)
        #
        # We take points x0 from Y since the characteristic curves
        # are more easily found from those initial points.
        characteristic_curves = []
        for x0 in Y:
            characteristic_curves.append(
                cc_axes.plot(lambda t: x(t, x0), color=WHITE)
            )

        # Now, we define an array Xcc_dots to store the
        # mobjects (manim objects) representing our points X.
        Xcc_dots = []
        for x_i in X:
            Xcc_dots.append(
                Dot(cc_axes.coords_to_point(0, x_i), color = GREEN)
            )

        ccs_group = VGroup(
            *characteristic_curves,
            cc_label,
            *Xcc_dots
        )

        # Add axes and graphs to the scene
        self.add(axes1, cc_axes)
        self.add(graph1_group, ccs_group)
        self.play(Create(graph1))
        
        # Note: final_t_value is closely related to the
        # new_t multiplier in the first line of the loop.
        # Changing one likely means you need to change the other.
        final_t_value = 13
        for i in range(final_t_value*10 + 1):
            new_t = 0.1 * i

            # First we update the x vs. u(t,x) graph for the updated t value

            graph1_transform = always_redraw(
                lambda: 
                    axes1.plot(
                        lambda x: u(new_t, x), 
                        color=WHITE
                )
            )

            new_X = []
            for y_i in Y:
                new_X.append(
                    x(new_t, y_i)
                )

            new_X_dots = []
            for x_i in new_X:
                new_X_dots.append(
                    always_redraw(
                        lambda: Dot(axes1.coords_to_point(x_i, 0), color = GREEN)
                    )
                )

            new_X_sticks = []
            for i in range(len(X)):
                new_X_sticks.append(
                    always_redraw(
                        lambda:
                            DashedLine(
                                new_X_dots[i].get_center(),
                                Dot(
                                    axes1.coords_to_point(
                                        new_X[i], 
                                        u(new_t, new_X[i]))
                                ).get_center(),
                                stroke_width = 2
                            )
                    )
                )

            transformed_graph1_group = VGroup(
                graph1_transform,
                label1,
                *new_X_dots,
                *new_X_sticks
            )

            # Now we update the cc points for the new t value:

            # Note that len(new_X) == len(Xcc_dots)
            for i in range(len(new_X)):
                Xcc_dots[i] = always_redraw(
                    lambda:
                        Dot(cc_axes.coords_to_point(new_t, new_X[i]), color = GREEN)
                )
            
            # Note that len(characteristic_curves) == len(Y)
            for i in range(len(characteristic_curves)):
                characteristic_curves[i] = always_redraw(
                    lambda: 
                        cc_axes.plot(lambda t: x(t, Y[i]), color=WHITE)
                )

            # Create new group with our transformed charateristic curve points
            transformed_ccs_group = VGroup(
                *characteristic_curves,
                cc_label,
                *Xcc_dots,
            )

            self.play(
                Transform(graph1_group, transformed_graph1_group),
                run_time=0.025
            )

            self.play(
                Transform(ccs_group, transformed_ccs_group), 
                run_time=0.025
            )
        
        self.wait(5)
