from manim import *
from sage.all import *
from sage.rings.factorint import factor_trial_division
from sage.schemes.elliptic_curves.mod_poly import classical_modular_polynomial
from SupersingularIsogenyGraphs import *


BLUE_B_35_OPACITY = "#354D50"
PRIME_COLOR = LIGHT_PINK
ALICE_COLORS = (BLUE_B, GREEN_D)
BOB_COLORS = (ORANGE, RED_D)


def prime_overlay_label(p):
    prime_string = None
    if p < (10**7):
        prime_string = str(int(p))
    # Attempt to factor p + 1 with small factors
    else:
        factorization = factor_trial_division(p + 1, 10000)
        for prime_factor, mult in list(factorization):
            if prime_factor >= 1000:
                # If no good rep, just write entire prime
                prime_string = p
                break
        if prime_string is None:
            prime_string = latex(factorization) + r" - 1"
    prime_tex = MathTex("p = ", prime_string)
    prime_tex[-1].set_color(PRIME_COLOR)
    return prime_tex


def get_initial_sage_data(p):
    Fq = FiniteField(p**2, "w")
    w = Fq.gen()
    if p % 4 == 3:
        E0 = EllipticCurve(Fq, [1, 0])
    elif p % 3 == 2:
        E0 = EllipticCurve(Fq, [0, 1])
    else:
        raise Exception("p must be congruent to 3 mod 4 or 2 mod 3")
    initial_j_invariant = E0.j_invariant()

    return Fq, w, E0, initial_j_invariant


def get_entire_sage_data_by_level(p, ell, maximum_depth=100):
    get_initial_sage_data(p)
    Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
    levels = [[initial_j_invariant.to_integer()]]
    all_vertex_ids = [initial_j_invariant.to_integer()]
    adjacency_list = (
        []
    )  # pairs of vertex ids in the graph ordered so that the first < second w.r.t jinv.to_integer()

    for i in range(1, maximum_depth):
        current_level = levels[i - 1]
        next_level = []
        # Get the neighbors of each vertex in the current level
        # and add them to the next level
        # Also, add the edges to the adjacency list
        # for each vertex in the current level
        for vert_id in current_level:
            j_inv = Fq.from_integer(vert_id)
            modular_polynomial_evaluated = classical_modular_polynomial(ell, j_inv)
            modular_roots = modular_polynomial_evaluated.roots(multiplicities=False)
            all_neighbor_ids = [
                neighbor_j.to_integer()
                for neighbor_j in modular_roots
                if neighbor_j != j_inv
            ]
            unique_neighbor_ids = [
                neighbor_id
                for neighbor_id in all_neighbor_ids
                if neighbor_id not in all_vertex_ids
            ]

            for neighbor_id in all_neighbor_ids:
                # Create the unique edge ID
                edge_id = tuple(sorted((vert_id, neighbor_id)))
                if edge_id not in adjacency_list:
                    adjacency_list.append(edge_id)

            next_level.extend(unique_neighbor_ids)
            all_vertex_ids.extend(unique_neighbor_ids)
            all_vertex_ids = list(set(all_vertex_ids))  # Remove duplicates

        # Double check- Remove duplicates from the next level
        next_level = list(set(next_level))
        if len(next_level) == 0:
            break
        levels.append(next_level)

    return levels, adjacency_list


def get_curves_real_data(E, num_points=100, randomize_coeff_signs=True):
    """
    Get a list of randomly sampled points on the elliptic curve E in the given x_range.
    There is significant sacrifices made here to visualize a curve defined over Fp^2
    in a way that looks like a curve defined over R. There in general is no canonical map Fp^2 -> R.
    The map we use on the coefficients is the following:
    a + b * w -> sqrt(a^2 + b^2)
    """
    j_inv = E.j_invariant()
    F = j_inv.parent()
    if F.degree() == 1:
        E = EllipticCurve_from_j(RR(j_inv))
        if randomize_coeff_signs:
            ainvs = [RR(np.random.choice([-1, 1])) * a_inv for a_inv in E.ainvs()]
        else:
            ainvs = E.ainvs()
        a1, a2, a3, a4, a6 = ainvs

    else:
        real_as = []
        for a_inv in E.ainvs():
            coeffs = a_inv.polynomial().coefficients(sparse=False)
            if len(coeffs) == 0:
                a = 0
                b = 0
            elif len(coeffs) == 1:
                a = coeffs[0]
                b = 0
            else:
                a, b = coeffs
            real_as.append(
                RR(np.random.choice([-1, 1])) * sqrt(RR(a) ** 2 + RR(b) ** 2)
            )
        E = EllipticCurve(real_as)
        a1, a2, a3, a4, a6 = real_as

    d = E.division_polynomial(2)

    def f1(z):
        # Internal function for plotting first branch of the curve
        return (-(a1 * z + a3) + sqrt(abs(d(z)))) / 2

    def f2(z):
        # Internal function for plotting second branch of the curve
        return (-(a1 * z + a3) - sqrt(abs(d(z)))) / 2

    r = sorted(d.roots(RR, multiplicities=False))
    xmins = []
    xmaxs = []
    if len(r) > 1:
        xmins.append(r[0])
        xmaxs.append(r[1])
    if len(r) == 1:
        flex = sorted(E.division_polynomial(3).roots(RR, multiplicities=False))
        flex = flex[-1]
        xmins.append(r[-1])
        # Arbitrary x value
        x_cutoff = flex + 2 * (flex - r[-1])
        xmaxs.append(x_cutoff)
        ymin = -abs(f1(x_cutoff))
        ymax = -ymin
    else:
        xmins.append(r[-1])
        R = RR["x"]
        x = R.gen()
        if a1 == 0:
            # a horizontal tangent line can only occur at a root of
            Ederiv = 3 * x**2 + 2 * a2 * x + a4
        else:
            # y' = 0  ==>  y = (3*x^2 + 2*a2*x + a4) / a1
            y = (3 * x**2 + 2 * a2 * x + a4) / a1
            Ederiv = y**2 + a1 * x * y + a3 * y - (x**3 + a2 * x**2 + a4 * x + a6)
        critx = [a for a in Ederiv.roots(RR, multiplicities=False) if r[0] < a < r[1]]
        if not critx:
            raise RuntimeError("No horizontal tangent lines on bounded component")
        # The 2.5 here is an aesthetic choice
        ymax = 2.5 * max([f1(a) for a in critx])
        ymin = 2.5 * min([f2(a) for a in critx])
        top_branch = (
            ymax**2 + a1 * x * ymax + a3 * ymax - (x**3 + a2 * x**2 + a4 * x + a6)
        )
        bottom_branch = (
            ymin**2 + a1 * x * ymin + a3 * ymin - (x**3 + a2 * x**2 + a4 * x + a6)
        )
        xmaxs.append(
            max(
                top_branch.roots(RR, multiplicities=False)
                + bottom_branch.roots(RR, multiplicities=False)
            )
        )
    xmin = min(xmins)
    xmax = max(xmaxs)
    span = xmax - xmin
    xmin = xmin - 0.2 * span
    xmax = xmax + 0.2 * span

    implicit_function = (
        lambda x_val, y_val: y_val**2
        + a1 * x_val * y_val
        + a3 * y_val
        - (x_val**3 + a2 * x_val**2 + a4 * x_val + a6)
    )

    points = []
    # Generate random points on the elliptic curve, that sum to a third
    while len(points) < num_points:
        x1 = RR(np.random.uniform(xmin, xmax))
        possible_points = E.lift_x(x1, all=True)
        for P in possible_points:
            if ymin <= P.y() <= ymax:
                points.append(P)

    points = [[P.x(), P.y()] for P in points if P != 0]
    return (
        points,
        implicit_function,
        [float(xmin), float(xmax)],
        [float(ymin), float(ymax)],
    )


class BreadthFirstPrime(Scene):
    """
    Visualizes the breadth-first traversal of a supersingular isogeny graph.
    - Animates the graph construction level by level:
        - Highlights vertices at the current level.
        - Creates edges connecting to the next level.
        - Dynamically scales and translates the graph to fit the scene.

    SIKE:
    manim_configuration["output_file"] = "BreadthFirstBigPrimel2"
    scene = BreadthFirstPrime(p = (2**216)*(3**137)-1, ell=2)
    scene.render()
    manim_configuration["output_file"] = "BreadthFirstBigPrimel3"
    scene = BreadthFirstPrime(p = (2**216)*(3**137)-1, ell=3)
    scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel2"
    scene = BreadthFirstPrime(ell=2)
    scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel3"
    scene = BreadthFirstPrime(ell=3)
    scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel5"
    scene = BreadthFirstPrime(ell=5)
    scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel7"
    scene = BreadthFirstPrime(ell=7)
    scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel11"
    scene = BreadthFirstPrime(ell=11)
    scene.render()
    """

    def __init__(self, p=2063, ell=2):
        self.ell = ell
        self.p = p
        super().__init__()

    def construct(self):
        ell = self.ell
        p = self.p
        p_label = prime_overlay_label(p).scale(1.3)
        l_label = MathTex(
            rf"\ell = ", ell, tex_to_color_map={f"{ell}": ALICE_COLORS[0]}
        ).scale(1.3)
        overlay = (
            VGroup(p_label, l_label, z_index=2)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UL, buff=0.5)
        ).add_background_rectangle(
            color=BLACK, opacity=0.6, buff=0.2, corner_radius=0.1
        )
        self.add(overlay)

        # Note to self: The flow of manim animation should always go somewhat like what is below:
        # 1. Perform any sage calculations neccesary to geenrate all mobjects that will be used in the scene
        # 2. Create all intermediate mobjects that will be used in the scene .animate should be avoided when possible. Always opt for Transform in these situations.
        # 3. Animate the mobjects in the scene
        print("Performing sage calculations...")
        Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
        if p < 10**7:
            maximum_depth = 20
        else:
            maximum_depth = 10
        levels, adjacency_list = get_entire_sage_data_by_level(
            p, ell, maximum_depth=maximum_depth
        )

        print("Creating graph mobjects...")
        # Create the entire graph
        ss_graph = SupersingularIsogenyGraph(Fq, levels, adjacency_list)

        # Create the intermediate mobjects (those that need be created each level)
        new_mobjects_on_level = [
            ss_graph.get_including_level(i) for i in range(len(levels))
        ]

        # Play the scene
        graph_mobjects_introduced = [*new_mobjects_on_level[0]]
        self.add(*new_mobjects_on_level[0])
        for i in range(1, len(new_mobjects_on_level)):
            prev_vertices, prev_edges = new_mobjects_on_level[i - 1]
            vertices, edges = new_mobjects_on_level[i]

            # Creation animations
            self.play(
                LaggedStart(
                    VGroup(vertex[0] for vertex in prev_vertices)
                    .animate(rate_func=rate_functions.there_and_back_with_pause)
                    .set_color(ALICE_COLORS[1]),
                    AnimationGroup(*[Create(edge) for edge in edges]),
                    lag_ratio=0.5,
                )
            )
            self.play(
                DrawBorderThenFill(vertices, run_time=0.5),
            )
            graph_mobjects_introduced.extend([vertices, edges])

            # Group together mobjects that are in the scene compute translation and scalling that would center the graph
            graph_mobject_in_scene = VGroup(*graph_mobjects_introduced)
            translation_vector = (
                -graph_mobject_in_scene.get_center()
            )  # Centers the graph
            aspect_ratio = graph_mobject_in_scene.length_over_dim(
                0
            ) / graph_mobject_in_scene.length_over_dim(1)
            desired_aspect_ratio = self.camera.frame_width / self.camera.frame_height
            # We inflate scaling_factor to add padding to graph. We do this by effectively having frame dimensions be 0.9x the actual frame dimensions
            if desired_aspect_ratio > aspect_ratio:
                scaling_factor = (
                    0.9 * self.camera.frame_height / graph_mobject_in_scene.height
                )
            else:
                scaling_factor = (
                    0.9 * self.camera.frame_width / graph_mobject_in_scene.width
                )

            if scaling_factor < 1 and i < len(new_mobjects_on_level) - 1:
                # Scale and transform every mob individually
                self.play(
                    *[
                        mob.animate()
                        .shift(translation_vector)
                        .scale(scaling_factor, about_point=ORIGIN, scale_stroke=True)
                        for group in graph_mobjects_introduced
                        for mob in group
                    ],
                )
                # Scale unscene graph levels to match
                for j in range(i + 1, len(new_mobjects_on_level)):
                    unsceene_vertices, unscreen_edges = new_mobjects_on_level[j]
                    for mob in [*unsceene_vertices, *unscreen_edges]:
                        mob.shift(translation_vector).scale(
                            scaling_factor, about_point=ORIGIN, scale_stroke=True
                        )
        self.wait(3)


class VerticesExplained(ThreeDScene):
    """
    A Manim scene that visualizes the vertices of a supersingular isogeny graph.
    The scene begins by introducing the initial real elliptic curve and its points,
    then transforms it into its finite field representation. It then constructs
    the supersingular isogeny graph level by level, displaying hovering grids
    with points and equations for each vertex.

    SIKE:
    manim_configuration["output_file"] = "VerticesExplainedp2063l2"
    scene = VerticesExplained(p=2063, ell=2)
    """

    def __init__(self, p=2063, ell=2):
        self.ell = ell
        self.p = p
        super().__init__()

    def construct(self):
        ell = self.ell
        p = self.p
        p_label = prime_overlay_label(p).scale(1.3)
        l_label = MathTex(
            rf"\ell = ", ell, tex_to_color_map={f"{ell}": ALICE_COLORS[0]}
        ).scale(1.3)
        overlay = (
            VGroup(p_label, l_label, z_index=2)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UL, buff=0.5)
        ).add_background_rectangle(
            color=BLACK, opacity=0.6, buff=0.2, corner_radius=0.1
        )
        self.add_fixed_in_frame_mobjects(overlay)
        self.remove(overlay)

        print("Performing sage calculations...")
        Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
        Fp = Fq.prime_subfield()
        if p > 10**7:
            raise NotImplementedError("p > 10^7 not implemented yet")
        levels, adjacency_list = get_entire_sage_data_by_level(p, ell, maximum_depth=5)

        # Some display parameters
        prime_resolution = 100  # These primes will simply be too big to display all points over Fp or Fp2, so this is the number of distinct x and y coordinate values we will display for x, y in Fp
        max_display_points = prime_resolution * 10  # This looks good. just heuristic
        every_curves_fp_points = {}
        every_curves_real_data = {}
        for level in levels:
            for vertex_id in level:
                j_inv = Fq.from_integer(vertex_id)
                # Here we are trying to express the fact that E is a curve over Fp^2, but also an elliptic curve.
                # I have descided for quick expository purposes that the details pertaining to the curve being defined over Fp^2 are not important
                # I simply want to emphasise that there is the real interpretation of the curve, and the finite field interpretation of the curve,
                # So, I will be displaying max_display_points points over Fp when E is defined over Fp, if E is over Fp^2, the points will be random.
                # While this is a cop-out, I could not find a way to get 100 points in E(F_p) in reasonable time. I tried finding one, then using group law. Didn't work since E is not defined over Fp. I tried randomly generating points, but since the densisity of Fp points is about 1/p, this is not a good idea.
                # Real points on the curve are honestly generated, sampling random points from methods in sageMaths E.plot().
                # However, an arbitrary mapping of the coefficients of the curve (Fp^2) to real numbers when E is defined over Fp^2.
                Fp_points = []
                if j_inv in Fp:
                    E = EllipticCurve_from_j(Fp(j_inv))
                    for _ in range(2 * p):
                        if len(Fp_points) >= max_display_points:
                            break
                        P = E.random_point()
                        x = Fp.from_integer(np.random.randint(0, p))
                        for P in E.lift_x(x, all=True):
                            P = [int(P.x()), int(P.y())]
                            if P not in Fp_points and P[0] < p and P[1] < p:
                                Fp_points.append(P)
                else:
                    E = EllipticCurve_from_j(j_inv)
                    for _ in range(2 * p):
                        if len(Fp_points) >= max_display_points:
                            break
                        P = [np.random.randint(0, p), np.random.randint(0, p)]
                        if P not in Fp_points:
                            Fp_points.append(P)
                every_curves_fp_points[vertex_id] = Fp_points
                every_curves_real_data[vertex_id] = get_curves_real_data(
                    E,
                    num_points=len(Fp_points),
                    randomize_coeff_signs=(j_inv != initial_j_invariant),
                )

        E0_Fp_points = every_curves_fp_points[initial_j_invariant.to_integer()]
        real_points, real_implicit_function, x_range, y_range = every_curves_real_data[
            initial_j_invariant.to_integer()
        ]

        print("Creating graph mobjects...")
        # Draw R2 Graph
        real_grid = NumberPlane(
            x_range=[x_range[0], x_range[1], max(1, int(x_range[1] - x_range[0]) // 5)],
            y_range=[y_range[0], y_range[1], max(1, int(y_range[1] - y_range[0]) // 5)],
            x_length=6,
            y_length=6,
            tips=False,
            background_line_style={
                "stroke_width": 1.5,
                "stroke_color": DARK_BLUE,
                "stroke_opacity": 0.7,
            },
            axis_config={
                "numbers_to_include": range(
                    int(x_range[0]),
                    int(x_range[1] + 1),
                    max(1, int(x_range[1] - x_range[0]) // 5),
                ),
                "stroke_opacity": 0.8,
            },
        )
        ec_equation = MathTex(latex(E0)).next_to(real_grid, UP)
        real_numbers = MathTex(r"\mathbb{R}", color=RED).next_to(real_grid, RIGHT)
        elliptic_curve_real_image = real_grid.plot_implicit_curve(
            real_implicit_function, color=RED, stroke_width=7
        )
        point_dots = []
        for i, pt in enumerate(real_points):
            point_dots.append(Dot(real_grid.c2p(pt[0], pt[1]), radius=0.03, color=RED))

        # Introduce the real curve
        self.add(real_grid)
        self.play(
            Create(elliptic_curve_real_image), Write(ec_equation), Write(real_numbers)
        )
        self.wait(1)
        self.remove(elliptic_curve_real_image)
        self.add(*point_dots)

        # Transform grid and real points to Fp
        finite_grid = NumberPlane(
            x_range=[0, p, p // prime_resolution],
            y_range=[0, p, p // prime_resolution],
            x_length=6,
            y_length=6,
            tips=False,
            background_line_style={
                "stroke_width": 1,
                "stroke_color": DARK_BLUE,
                "stroke_opacity": 0.5,
            },
            axis_config={"include_numbers": False, "stroke_opacity": 0.8},
            y_axis_config={"label_direction": LEFT},
        )
        fp_coords = [finite_grid.c2p(int(pt[0]), int(pt[1])) for pt in E0_Fp_points]
        fp_numbers = MathTex(r"\mathbb{F}^2_p").next_to(real_grid, RIGHT)
        self.play(
            ReplacementTransform(real_grid, finite_grid),
            ReplacementTransform(real_numbers, fp_numbers),
            *[
                dot.animate().move_to(coord)
                for coord, dot in zip(fp_coords, point_dots)
            ],
            Write(overlay),
            run_time=3,
        )
        self.wait(1)

        # Create the entire graph
        ss_graph = SupersingularIsogenyGraph(
            Fq, levels, adjacency_list, unit_length=1.8
        )

        # Create the intermediate mobjects (those that need be created each level)
        new_mobjects_on_level = [
            ss_graph.get_including_level(i) for i in range(len(levels))
        ]

        # Smaller, hovering Fp grids with points displayed
        hovering_Fp_grids = {}
        for vertex_id, vertex in ss_graph.vertices.items():
            hover_grid = NumberPlane(
                x_range=[0, p, p // prime_resolution],
                y_range=[0, p, p // prime_resolution],
                x_length=1.5,
                y_length=1.5,
                tips=False,
                background_line_style={
                    "stroke_width": 0.35,
                    "stroke_color": DARK_BLUE,
                    "stroke_opacity": 0.4,
                },
                axis_config={"include_numbers": False, "stroke_opacity": 0.6},
            ).next_to(vertex, OUT, buff=1.5)
            hover_points = VGroup()
            for P in every_curves_fp_points[vertex_id]:
                hover_points.add(
                    Dot(hover_grid.c2p(P[0], P[1]), radius=0.015, color=PINK)
                )
            hovering_Fp_grids[vertex_id] = VGroup(hover_grid, hover_points)

        # Smallet analogous grids with real points displayed
        hovering_real_grids = {}
        for vertex_id, vertex in ss_graph.vertices.items():
            real_points, real_implicit_function, x_range, y_range = (
                every_curves_real_data[vertex_id]
            )

            hover_grid = NumberPlane(
                x_range=[
                    x_range[0],
                    x_range[1],
                    max(1, int(x_range[1] - x_range[0]) // 5),
                ],
                y_range=[
                    y_range[0],
                    y_range[1],
                    max(1, int(y_range[1] - y_range[0]) // 5),
                ],
                x_length=1.5,
                y_length=1.5,
                tips=False,
                background_line_style={
                    "stroke_width": 0.35,
                    "stroke_color": BLUE_D,
                    "stroke_opacity": 0.4,
                },
                axis_config={"include_numbers": False, "stroke_opacity": 0.6},
            ).next_to(vertex, OUT, buff=1.5)
            hover_points = VGroup()
            for P in real_points:
                hover_points.add(
                    Dot(hover_grid.c2p(P[0], P[1]), radius=0.015, color=RED)
                )
            hovering_real_grids[vertex_id] = VGroup(hover_grid, hover_points)

        # Create the equations that lay above the hovering grids
        hovering_equations = {}
        for vertex_id, small_grid in zip(ss_graph.vertices, hovering_Fp_grids.values()):
            ec_equation_str = latex(EllipticCurve_from_j(Fq.from_integer(vertex_id)))
            if len(ec_equation_str) > 22:
                ec_equation_str = ec_equation_str[:22] + r"\dots"
            hovering_equations[vertex_id] = (
                MathTex(
                    ec_equation_str,
                    font_size=30,
                    z_index=2,
                )
                .next_to(small_grid, UP, buff=0.25)
                .add_background_rectangle(
                    color=BLACK, opacity=0.7, buff=0.2, corner_radius=0.1
                )
            )

        self.move_camera(phi=25 * DEGREES, run_time=1.5)
        self.wait(0.5)
        first_vertex_id = levels[0][0]
        self.play(
            ReplacementTransform(
                finite_grid,
                hovering_Fp_grids[first_vertex_id][0],
            ),
            TransformMatchingShapes(
                VGroup(*point_dots),
                hovering_Fp_grids[first_vertex_id][1],
                replace_mobject_with_target_in_scene=True,
            ),
            ReplacementTransform(ec_equation, hovering_equations[first_vertex_id]),
            FadeOut(fp_numbers),
            run_time=1.5,
        )
        self.wait(0.5)
        self.play(Write(new_mobjects_on_level[0]))
        self.wait(0.5)
        # Play the scene
        graph_mobjects_introduced = [*new_mobjects_on_level[0]]
        grid_mobjects_introduced = [hovering_Fp_grids[first_vertex_id]]
        equation_mobjects_introduced = [hovering_equations[first_vertex_id]]
        for i in range(1, len(new_mobjects_on_level)):
            prev_vertices, prev_edges = new_mobjects_on_level[i - 1]
            vertices, edges = new_mobjects_on_level[i]

            # Quick creation animations
            self.play(
                LaggedStart(
                    VGroup(vertex[0] for vertex in prev_vertices)
                    .animate(rate_func=rate_functions.there_and_back_with_pause)
                    .set_color(ALICE_COLORS[1]),
                    AnimationGroup(*[Create(edge) for edge in edges]),
                    lag_ratio=0.5,
                    run_time=1,
                )
            )
            self.play(DrawBorderThenFill(vertices, run_time=0.5))
            graph_mobjects_introduced.extend([vertices, edges])

            # Add the overlaying hovering grids and labels one-by-one
            if i < 3:
                for vertex_id in levels[i]:
                    hover_grid = hovering_Fp_grids[vertex_id]
                    vertex_label = hovering_equations[vertex_id]
                    self.play(
                        Write(vertex_label),
                        GrowFromPoint(hover_grid[0], hover_grid[0].get_center() + IN),
                        Create(hover_grid[1]),
                    )
                    grid_mobjects_introduced.append(hover_grid)
                    equation_mobjects_introduced.append(vertex_label)

            # Add all the overlaying hovering grids and labels at once
            else:
                self.play(
                    *[
                        AnimationGroup(
                            Write(hovering_equations[vertex_id]),
                            GrowFromPoint(
                                hovering_Fp_grids[vertex_id][0],
                                hovering_Fp_grids[vertex_id][0].get_center() + IN,
                            ),
                            Create(hovering_Fp_grids[vertex_id][1]),
                        )
                        for vertex_id in levels[i]
                    ],
                )
                grid_mobjects_introduced.extend(
                    [hovering_Fp_grids[vertex_id] for vertex_id in levels[i]]
                )
                equation_mobjects_introduced.extend(
                    [hovering_equations[vertex_id] for vertex_id in levels[i]]
                )

            # Fancy Fp to R2 transformation
            for vertex_id in levels[i]:
                hovering_Fp_grids[vertex_id][0].save_state()
                hovering_Fp_grids[vertex_id][1].save_state()

            self.wait(0.5)
            self.play(
                *[FadeOut(hovering_equations[vertex_id]) for vertex_id in levels[i]],
                run_time=0.5,
            )
            self.play(
                *[
                    AnimationGroup(
                        Transform(
                            hovering_Fp_grids[vertex_id][0],
                            hovering_real_grids[vertex_id][0],
                        ),
                        Transform(
                            hovering_Fp_grids[vertex_id][1],
                            hovering_real_grids[vertex_id][1],
                        ),
                    )
                    for vertex_id in levels[i]
                ],
                run_time=1,
            )
            self.wait(1)
            self.play(
                *[
                    AnimationGroup(
                        hovering_Fp_grids[vertex_id][0].animate.restore(),
                        hovering_Fp_grids[vertex_id][1].animate.restore(),
                        FadeIn(hovering_equations[vertex_id]),
                    )
                    for vertex_id in levels[i]
                ],
                run_time=1,
            )

            # Group together mobjects that are in the scene compute translation and scalling that would center the graph
            graph_mobject_in_scene = VGroup(
                *graph_mobjects_introduced,
                *grid_mobjects_introduced,
                *equation_mobjects_introduced,
            )
            translation_vector = (
                -graph_mobject_in_scene.get_center()
            )  # Centers the graph
            aspect_ratio = graph_mobject_in_scene.length_over_dim(
                0
            ) / graph_mobject_in_scene.length_over_dim(1)
            desired_aspect_ratio = self.camera.frame_width / self.camera.frame_height
            # We inflate scaling_factor to add padding to graph. We do this by effectively having frame dimensions be 0.6x the actual frame dimensions
            if desired_aspect_ratio > aspect_ratio:
                scaling_factor = (
                    0.6 * self.camera.frame_height / graph_mobject_in_scene.height
                )
            else:
                scaling_factor = (
                    0.6 * self.camera.frame_width / graph_mobject_in_scene.width
                )

            if scaling_factor < 1:
                # Scale and transform every mob individually
                if i < len(levels) - 1:
                    self.play(
                        *[
                            mob.animate()
                            .shift(translation_vector)
                            .scale(scaling_factor, about_point=ORIGIN)
                            for group in graph_mobjects_introduced
                            for mob in group
                        ],
                        *[
                            mob.animate()
                            .shift(translation_vector)
                            .scale(scaling_factor, about_point=ORIGIN)
                            for grid_part in grid_mobjects_introduced
                            for mob in grid_part
                        ],
                        *[
                            mob.animate()
                            .shift(translation_vector)
                            .scale(scaling_factor, about_point=ORIGIN)
                            for mob in equation_mobjects_introduced
                        ],
                        run_time=0.5,
                    )
                    # Scale unscene graph levels to match
                    for j in range(i + 1, len(levels)):
                        unseen_vertices, unseen_edges = new_mobjects_on_level[j]
                        for mob in [*unseen_vertices, *unseen_edges]:
                            mob.shift(translation_vector).scale(
                                scaling_factor, about_point=ORIGIN, scale_stroke=True
                            )
                        unseen_hovering = []

                        for vertex_id in levels[j]:
                            unseen_hovering.extend(hovering_Fp_grids[vertex_id])
                            unseen_hovering.extend(hovering_real_grids[vertex_id])
                            unseen_hovering.append(hovering_equations[vertex_id])

                        for mob in unseen_hovering:
                            mob.shift(translation_vector).scale(
                                scaling_factor, about_point=ORIGIN
                            )
                if i == 2:
                    self.begin_ambient_camera_rotation(rate=0.15)
                    self.move_camera(phi=60 * DEGREES, theta=-150 * DEGREES, run_time=2)

        self.wait(2)


class EdgesExplained(Scene):
    """
    A Manim scene that visualizes the edges of a supersingular isogeny graph and explains how the modular polynomial is used to find the neighbors of a vertex.
    The scene begins by introducing the modular polynomial and then shows how it is used to find the neighbors of a vertex in the graph.
    The modular polynomial is reduced modulo p and then evaluated at the j-invariant of the vertex. The roots of the resulting polynomial are the j-invariants of the neighboring vertices.
    
    SIKE:
        manim_configuration["output_file"] = "EdgesExplainedl2"
        scene = EdgesExplained(ell=2)
        scene.render()
        manim_configuration["output_file"] = "EdgesExplainedl3"
        scene = EdgesExplained(ell=3)
        scene.render()
        manim_configuration["output_file"] = "EdgesExplainedl5"
        scene = EdgesExplained(ell=5)
        scene.render()
        manim_configuration["disable_caching"] = True
        manim_configuration["output_file"] = "EdgesExplainedl7"
        scene = EdgesExplained(ell=7)
        scene.render()
        manim_configuration["output_file"] = "EdgesExplainedl11"
        scene = EdgesExplained(ell=1)
        scene.render()
        manim_configuration["disable_caching"] = False
    """
    def __init__(self, p=2063, ell=2):
        self.ell = ell
        self.p = p
        super().__init__()

    def construct(self):
        ell = self.ell
        p = self.p
        p_label = prime_overlay_label(p).scale(1.3)
        l_label = MathTex(
            rf"\ell = ", ell, tex_to_color_map={r"2": ALICE_COLORS[0]}
        ).scale(1.3)
        overlay = (
            VGroup(p_label, l_label, z_index=2)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UL, buff=0.5)
        )

        top_background_rectangle = RoundedRectangle(
            color=BLACK, corner_radius=0.1, z_index=-1
        ).stretch_to_fit_height(overlay.height + 0.2).stretch_to_fit_width(self.camera.frame_width).move_to(overlay, aligned_edge=UP).set_opacity(0.8)
        self.add(top_background_rectangle)

        print("Performing sage calculations...")
        Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
        Fp = Fq.prime_subfield()
        levels, adjacency_list = get_entire_sage_data_by_level(p, ell, maximum_depth=4)

        poly_ring_integers = PolynomialRing(Integers(), 2, names=("X", "Y"))
        poly_ring_Fp = PolynomialRing(Fp, 2, names=("X", "Y"))
        X, Y = poly_ring_Fp.gens()

        mod_poly = poly_ring_integers(classical_modular_polynomial(ell))
        mod_poly_string = latex(mod_poly)
        mod_poly_reduced = poly_ring_Fp(classical_modular_polynomial(ell))
        mod_poly_reduced_string = latex(mod_poly_reduced)

        print("Creating graph mobjects...")
        font_sizes = {
            2: 48,
            3: 44,
            5: 22,
            7: 14,
            11: 12,
        }
        if ell not in font_sizes:
            raise NotImplementedError(f"ell = {ell} not implemented yet")
        font_size = font_sizes[ell]
        font_size_larger = font_size * 1.5 if ell > 3 else font_size
        mod_poly_label = (
            MathTex(
                r"\Phi_{" + str(ell) + r"}(X, Y) = ",
                mod_poly_string,
                substrings_to_isolate=["+", "-"],
                font_size=font_size,
            )
            .to_corner(UL, buff=1)
            .shift(DOWN * 1)
        )
        mod_poly_reduced_label = (
            MathTex(
                r"\Phi_{" + str(ell) + r"}(X, Y) \equiv ",
                mod_poly_reduced_string,
                r" \pmod p",
                substrings_to_isolate=["+", "-"],
                font_size=font_size,
            )
            .to_corner(UL, buff=1)
            .shift(DOWN * 1)
        )

        def arrange_equation_smartly(math_tex_equation, aligned_index = 1, skip_analysis_of = None):
            if skip_analysis_of is None:
                skip_analysis_of = [")", "(", " "]
            for i in range(len(math_tex_equation)):
                if i == 0 or math_tex_equation[i].get_tex_string() in skip_analysis_of:
                    continue
                if (
                    math_tex_equation[i].get_right()[0]
                    > self.camera.frame_width / 2 - 1
                ):
                    rows_first_index = (
                        i
                        if math_tex_equation[i].get_tex_string() in ["+", "-", r"\cdot", r" \pmod p", r"\pmod p"]
                        else i - 1
                    )
                    VGroup(*math_tex_equation[rows_first_index:]).next_to(
                        math_tex_equation[rows_first_index - 1], DOWN, buff=0.2
                    )
                    VGroup(*math_tex_equation[rows_first_index:]).align_to(
                        math_tex_equation[aligned_index], LEFT
                    )
            return math_tex_equation

        arrange_equation_smartly(mod_poly_label)
        arrange_equation_smartly(mod_poly_reduced_label)
        p_label_above_coefficients = VGroup(
            *[
                MathTex(p, font_size=(font_size * 13) // 10)
                .move_to(mod_poly_label[i])
                .set_color(ALICE_COLORS[1])
                .set_opacity(0)
                for i in range(2, len(mod_poly_label), 2)
            ]
        )

        self.add(overlay)
        self.add(mod_poly_label[0])
        self.play(Write(VGroup(*mod_poly_label[1:])))
        p_label.save_state()
        self.play(
            VGroup(*p_label[1:]).animate.scale(1.2),
            run_time=0.5,
        )
        self.wait(0.5)
        self.play(
            LaggedStart(
                *[
                    LaggedStart(
                        TransformFromCopy(
                            VGroup(*p_label[1:]),
                            extra_p_label,
                        ),
                        AnimationGroup(
                            Transform(
                                mod_poly_label[1 + 2 * i],
                                mod_poly_reduced_label[1 + 2 * i]
                                .copy()
                                .move_to(mod_poly_label[1 + 2 * i], aligned_edge=LEFT),
                            ),
                            Transform(
                                mod_poly_label[2 + 2 * i],
                                mod_poly_reduced_label[2 + 2 * i]
                                .copy()
                                .move_to(mod_poly_label[2 + 2 * i], aligned_edge=LEFT),
                            ),
                        ),
                        lag_ratio=0.5,
                    )
                    for i, extra_p_label in enumerate(p_label_above_coefficients)
                ],
                run_time=4,
                lag_ratio=0.3
            )
        )
        self.remove(*p_label_above_coefficients)
        self.play(
            ReplacementTransform(mod_poly_label, VGroup(*mod_poly_reduced_label[:-1]))
        )
        self.play(Restore(p_label), Write(mod_poly_reduced_label[-1]))
        self.wait(1)

        # Smalle bottom left line width label
        line_width_reduced_mod_poly = (
            MathTex(
                r"\Phi_{" + str(ell) + r"}(X, Y) \equiv ",
                mod_poly_reduced_string,
                r" \pmod p",
                substrings_to_isolate=["+", "-"],
                font_size=font_size,
            )
            .scale(0.7)
            .to_edge(LEFT, buff=0.25)
        )
        arrange_equation_smartly(line_width_reduced_mod_poly)
        line_width_reduced_mod_poly.to_edge(DOWN, buff=0.5)
        evaluated_labels = {}
        evaluated_labels_factored = {}
        for level in levels:
            for vertex_id in level:
                j_inv = Fq.from_integer(vertex_id)
                evaluated_mod_poly = mod_poly_reduced.subs({X: j_inv}).univariate_polynomial()
                evaluated_mod_poly_roots = evaluated_mod_poly.roots()

                tex_color_map = {latex(j_inv): ALICE_COLORS[1]}
                formatted_factors = []
                for (r, e) in evaluated_mod_poly_roots:
                    factor_string = "Y"
                    if r != 0:
                        factor_string = f"(Y - {latex(r)})"
                    if e != 1:
                        factor_string += f"^{e}"
                    formatted_factors.append(factor_string)
                    tex_color_map[latex(r)] = ALICE_COLORS[0]
                formatted_mod_poly_factored = r" \cdot ".join(formatted_factors)

                evaluated_labels[vertex_id] = arrange_equation_smartly(
                    MathTex(
                        r"\Phi_{" + str(ell) + r"}(",
                        latex(j_inv),
                        r", Y) \equiv ",
                        latex(evaluated_mod_poly),
                        r" \pmod p",
                        substrings_to_isolate=["+", "-", r"\cdot"],
                        tex_to_color_map={latex(j_inv): ALICE_COLORS[1]},
                        font_size=font_size_larger,
                    )
                    .scale(0.8)
                    .next_to(p_label, RIGHT, aligned_edge=UP, buff=0.5),
                    aligned_index=3,
                )
                evaluated_labels_factored[vertex_id] = arrange_equation_smartly(
                    MathTex(
                        r"\Phi_{" + str(ell) + r"}(",
                        latex(j_inv),
                        r", Y) \equiv ",
                        formatted_mod_poly_factored,
                        r" \pmod p",
                        substrings_to_isolate=[r"\cdot"],
                        tex_to_color_map=tex_color_map,
                        font_size=font_size_larger,
                    )
                    .scale(0.8)
                    .next_to(p_label, RIGHT, aligned_edge=UP, buff=0.5),
                    aligned_index=3
                )

        copied_long_equation = (
            MathTex(
                r"\Phi_{" + str(ell) + r"}(X, Y) \equiv ",
                mod_poly_reduced_string,
                r" \pmod p",
                substrings_to_isolate=["+", "-"],
                font_size=font_size_larger,
            )
            .scale(0.8)
        ).next_to(p_label, RIGHT, aligned_edge=UP, buff=0.5)
        arrange_equation_smartly(copied_long_equation)

        self.play(
            ReplacementTransform(mod_poly_reduced_label.copy(), copied_long_equation),
            ReplacementTransform(mod_poly_reduced_label, line_width_reduced_mod_poly)
        )
        bottom_background_rectangle = (
            RoundedRectangle(color=BLACK, corner_radius=0.1, z_index=-1)
            .set_opacity(0.8)
            .stretch_to_fit_height(line_width_reduced_mod_poly.height + 0.2)
            .stretch_to_fit_width(self.camera.frame_width)
            .move_to(line_width_reduced_mod_poly, aligned_edge=DOWN)
        )
        self.remove(line_width_reduced_mod_poly)
        self.add(line_width_reduced_mod_poly, bottom_background_rectangle)

        # Create the entire graph
        ss_graph = SupersingularIsogenyGraph(Fq, levels, adjacency_list).scale(
            0.65, about_point=ORIGIN
        ).shift(UP).set_z_index(-10)

        # Create the intermediate mobjects (those that need be created each level)
        new_mobjects_on_level = [
            ss_graph.get_including_level(i) for i in range(len(levels))
        ]

        # Play the scene
        graph_mobjects_introduced = [*new_mobjects_on_level[0]]
        self.play(DrawBorderThenFill(new_mobjects_on_level[0][0]))
        current_factored_label = None
        for i in range(1, len(levels)):
            prev_vertices, prev_edges = new_mobjects_on_level[i - 1]

            # Creation animations
            for vertex_id, vertex in zip(levels[i - 1], prev_vertices):
                neighboring_edges = ss_graph.get_neighboring_edges(
                    vertex_id, exclude_previous_level=True
                )
                neighboring_vertices = ss_graph.get_neighboring_vertices(
                    vertex_id, exclude_previous_level=True
                )

                neighboring_edges = VGroup(*neighboring_edges)
                neighboring_vertices = VGroup(*neighboring_vertices)
                # highlight the vertex
                self.play(VGroup(vertex[0]).animate().set_color(ALICE_COLORS[1]))

                # For the first integer, must introduce the evaluated_label
                if vertex_id == initial_j_invariant.to_integer():
                    self.play(
                        ReplacementTransform(vertex[-1].copy(), evaluated_labels[vertex_id][1]),
                        ReplacementTransform(
                            copied_long_equation[0], evaluated_labels[vertex_id][:3:2]
                        ),
                        copied_long_equation[1:]
                        .animate()
                        .next_to(
                            evaluated_labels[vertex_id][2],
                            RIGHT,
                            aligned_edge=UP,
                            buff=0.1,
                        ),
                    )
                    self.play(ReplacementTransform(copied_long_equation[1:], evaluated_labels[vertex_id][3:]))
                    self.remove(evaluated_labels[vertex_id])
                    self.add(evaluated_labels[vertex_id])
                    self.wait(0.5)
                    self.play(TransformMatchingTex(evaluated_labels[vertex_id], evaluated_labels_factored[vertex_id]))
                    current_factored_label = evaluated_labels_factored[vertex_id]
                else:
                    self.play(
                        ReplacementTransform(
                            vertex[-1].copy(), evaluated_labels[vertex_id][1]
                        ),
                        FadeOut(current_factored_label[1]),
                        ReplacementTransform(
                            current_factored_label[2:], evaluated_labels[vertex_id][2:]
                        ),
                        ReplacementTransform(
                            current_factored_label[0], evaluated_labels[vertex_id][0]
                        )
                    )
                    self.wait(0.25)
                    self.play(TransformMatchingTex(evaluated_labels[vertex_id], evaluated_labels_factored[vertex_id]))
                    current_factored_label = evaluated_labels_factored[vertex_id]

                self.play(Circumscribe(VGroup(*current_factored_label[3:-1])), run_time=0.5)
                self.play(
                    AnimationGroup(*[Create(edge) for edge in neighboring_edges]),
                    run_time=0.5
                )
                self.play(
                    DrawBorderThenFill(neighboring_vertices, run_time=0.5),
                )
                self.play(VGroup(vertex[0]).animate().set_color(ALICE_COLORS[0]), run_time=0.5)
                graph_mobjects_introduced.extend([neighboring_vertices, neighboring_edges])

            # Group together mobjects that are in the scene compute translation and scalling that would center the graph
            graph_mobject_in_scene = VGroup(*graph_mobjects_introduced)
            translation_vector = (
                -graph_mobject_in_scene.get_center()
            )  # Centers the graph
            aspect_ratio = graph_mobject_in_scene.length_over_dim(
                0
            ) / graph_mobject_in_scene.length_over_dim(1)
            desired_aspect_ratio = self.camera.frame_width / self.camera.frame_height
            # We inflate scaling_factor to add padding to graph. We do this by effectively having frame dimensions be 0.9x the actual frame dimensions
            if desired_aspect_ratio > aspect_ratio:
                scaling_factor = (
                    0.75 * self.camera.frame_height / graph_mobject_in_scene.height
                )
            else:
                scaling_factor = (
                    0.75 * self.camera.frame_width / graph_mobject_in_scene.width
                )

            if scaling_factor < 1:
                # Scale and transform every mob individually
                self.play(
                    *[
                        mob.animate()
                        .shift(translation_vector)
                        .scale(scaling_factor, about_point=ORIGIN, scale_stroke=True)
                        for group in graph_mobjects_introduced
                        for mob in group
                    ],
                )
                # Scale unscene graph levels to match
                for j in range(i + 1, len(new_mobjects_on_level)):
                    unsceene_vertices, unscreen_edges = new_mobjects_on_level[j]
                    for mob in [*unsceene_vertices, *unscreen_edges]:
                        mob.shift(translation_vector).scale(
                            scaling_factor, about_point=ORIGIN, scale_stroke=True
                        )


if __name__ == "__main__":
    manim_configuration = {
        "quality": "production_quality",
        "preview": False,
        "output_file": "PreviewVideo",
        "disable_caching": False,
        "max_files_cached": 1000,
        "write_to_movie": True,
        "show_file_in_browser": False,
    }
    np.random.seed(0)
    manim_configuration["output_file"] = "EdgesExplained"
    with tempconfig(manim_configuration):
        scene = EdgesExplained(ell=2)
        scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel2"
    with tempconfig(manim_configuration):
        scene = BreadthFirstPrime(ell=2)
        scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel3"
    with tempconfig(manim_configuration):
        scene = BreadthFirstPrime(ell=3)
        scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel5"
    with tempconfig(manim_configuration):
        scene = BreadthFirstPrime(ell=5)
        scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel7"
    with tempconfig(manim_configuration):
        scene = BreadthFirstPrime(ell=7)
        scene.render()
    manim_configuration["output_file"] = "BreadthFirstSmallPrimel11"
    with tempconfig(manim_configuration):
        scene = BreadthFirstPrime(ell=11)
        scene.render()

    manim_configuration["disable_caching"] = True
    manim_configuration["output_file"] = "VerticesExplained"
    with tempconfig(manim_configuration):
        scene = VerticesExplained(p=2063, ell=2)
        scene.render()
    manim_configuration["output_file"] = "BreadthFirstBigPrimel2"
    with tempconfig(manim_configuration):
        scene = BreadthFirstPrime(p = (2**216)*(3**137)-1, ell=2)
        scene.render()
    manim_configuration["output_file"] = "BreadthFirstBigPrimel3"
    with tempconfig(manim_configuration):
        scene = BreadthFirstPrime(p = (2**216)*(3**137)-1, ell=3)
        scene.render()
