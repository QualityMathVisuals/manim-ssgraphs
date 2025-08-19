from manim import *
from SSGraphIntroduction import prime_overlay_label, get_initial_sage_data, ALICE_COLORS, BOB_COLORS, PRIME_COLOR
from SupersingularIsogenyGraphs import IsogenyDiamond, GraphVertexMobject
from sage.all import *

from sage.schemes.elliptic_curves.hom_composite import EllipticCurveHom_composite

def path_from_isogeny(isogeny):
    predetermined_path = [phi.codomain().j_invariant().to_integer() for phi in isogeny.factors()]
    predetermined_path = list(dict.fromkeys(predetermined_path))
    domain = isogeny.domain().j_invariant().to_integer()
    if domain != predetermined_path[0]:
        predetermined_path.insert(0, domain)
    return predetermined_path

def get_SIKE_paths(E0, e2, e3):
    PA, QA = E0.torsion_basis(2 ** e2)
    PB, QB = E0.torsion_basis(3 ** e3)
    sk_A = np.random.randint(0, 2 ** e2)
    sk_B = np.random.randint(0, 3 ** e3)
    sk_PA = sk_A * PA +  QA
    sk_PB = sk_B * PB +  QB
    PhiA = EllipticCurveHom_composite(E0, sk_PA)
    PhiB = EllipticCurveHom_composite(E0, sk_PB)
    EA = PhiA.codomain()
    EB = PhiB.codomain()

    pushed_two_torsion = PhiB(PA), PhiB(QA)
    pushed_three_torsion = PhiA(PB), PhiA(QB)

    A_path_one = path_from_isogeny(PhiA)
    B_path_one = path_from_isogeny(PhiB)

    PhiPrimeA = EllipticCurveHom_composite(EB, PhiB(sk_PA))
    PhiPrimeB = EllipticCurveHom_composite(EA, PhiA(sk_PB))

    A_path_two = path_from_isogeny(PhiPrimeA)
    B_path_two = path_from_isogeny(PhiPrimeB)

    shared_sk_id = A_path_two[-1]

    return A_path_one, B_path_one, A_path_two, B_path_two, sk_A, sk_B, pushed_two_torsion, pushed_three_torsion, shared_sk_id

def choose_random_int(scene, a, b, predetermined_outcome=None, run_time=1, center=ORIGIN, width=None, **kwargs):
    rand_ints = [np.random.randint(a, b) for _ in range(run_time * 30 + 1)]
    if predetermined_outcome is not None:
        rand_ints[-1] = predetermined_outcome
    int_tex = [MathTex(*list(f'{i}'), **kwargs).move_to(center) for i in rand_ints]
    if width is not None:
        for tex in int_tex:
            tex.scale_to_fit_width(width)
    run_time_per = run_time / 30
    scene.add(int_tex[0])
    for i in range(len(int_tex) -1):
        scene.wait(run_time_per)
        scene.remove(int_tex[i])
        scene.add(int_tex[i + 1])
    return int_tex[-1], rand_ints[-1]

def to_base(number, base):
    """Converts a non-negative number to a list of digits in the given base.

    The base must be an integer greater than or equal to 2 and the first digit
    in the list of digits is the most significant one.
    """
    if not number:
        return [0]

    digits = []
    while number:
        digits.append(number % base)
        number //= base
    return list(reversed(digits))

class SIDHExchange(Scene):
    def __init__(self, e2=5, e3=4, perspective="Both"):
        self.e2 = e2
        self.e3 = e3
        self.perspective = perspective
        super().__init__()

    def construct(self):
        e2 = self.e2
        e3 = self.e3
        p = (2 ** e2) * (3 ** e3) - 1

        p_label = prime_overlay_label(p)
        l_string = str(2) + r" \, " + str(3)
        l_label = MathTex(
            rf"\ell = ", str(2), r" \, ", str(3)
        )
        l_label[1].set_color(ALICE_COLORS[0])
        l_label[3].set_color(BOB_COLORS[0])
        overlay = (
            VGroup(p_label, l_label, z_index=2)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UL, buff=0.5)
        )
        self.add(overlay[0], overlay[1])

        print("Performing sage calculations...")
        Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
        A_path_one, B_path_one, A_path_two, B_path_two, sk_A, sk_B, pushed_two_torsion, pushed_three_torsion, shared_sk_id = get_SIKE_paths(E0, e2, e3)

        print("Creating graph mobjects...")
        sike_graph = IsogenyDiamond(Fq, A_path_one, B_path_one, A_path_two, B_path_two)
        levels = sike_graph.levels

        # Color sides Appriporiately
        l2_subgraph = list(dict.fromkeys(A_path_one + A_path_two))
        l3_subgraph = list(dict.fromkeys(B_path_one + B_path_two))
        mixed_color = interpolate_color(ALICE_COLORS[1], BOB_COLORS[1], 0.5)
        for vertex_id in sike_graph.vertices:
            if vertex_id in l2_subgraph and vertex_id not in l3_subgraph:
                sike_graph.vertices[vertex_id][0].set_color(ALICE_COLORS[0])
            elif vertex_id in l3_subgraph and vertex_id not in l2_subgraph:
                sike_graph.vertices[vertex_id][0].set_color(BOB_COLORS[0])
            else:
                sike_graph.vertices[vertex_id][0].set_color(mixed_color)

        level_for_swap = max(len(A_path_one), len(B_path_one))

        new_mobjects_on_level = [
            sike_graph.get_including_level(i) for i in range(len(levels))
        ]

        # Create the text overlays
        if self.perspective == "Alice":
            perspective_l = 2
            perspective_exp = self.e2
            main_torsion_string = "E_0", R'[2^{' + str(self.e2) + R'}]'
            other_torsion_string = "E_0", R'[3^{' + str(self.e3) + r'}]'
            main_pushed_torsion = pushed_three_torsion
            other_pushed_torsion = pushed_two_torsion
            main_color = ALICE_COLORS[0]
            other_color = BOB_COLORS[0]
            abrev_main = r'A'
            abrev_other = r'B'
            Alice = VGroup(
                Text(r' Alice ', color=ALICE_COLORS[0], font_size=50),
                MathTex(fr'\ell = ', f'2')
            ).arrange(DOWN)
            Alice[1][1].set_color(ALICE_COLORS[0])
            Bob = VGroup(
                Text(r' Bob ', color=BOB_COLORS[0], font_size=50),
                MathTex(fr'\ell = ', f'3')
            ).arrange(DOWN)
            Bob[1][1].set_color(BOB_COLORS[0])
            main_label = Alice
            other_label = Bob
            sk_num = sk_A
            main_paths = A_path_one, A_path_two
            main_subgraph = l2_subgraph
        elif self.perspective == "Bob":
            perspective_l = 3
            perspective_exp = self.e3
            main_torsion_string = "E_0", R'[3^{' + str(self.e3) +r'}]'
            other_torsion_string = "E_0", R'[2^{' + str(self.e2) + r'}]'
            main_pushed_torsion = pushed_two_torsion
            other_pushed_torsion = pushed_three_torsion
            main_color = BOB_COLORS[0]
            other_color = ALICE_COLORS[0]
            abrev_main = r'B'
            abrev_other = r'A'
            Alice = VGroup(
                Text(r' Alice ', color=ALICE_COLORS[0], font_size=50),
                MathTex(fr'\ell = ', f'2')
            ).arrange(DOWN)
            Alice[1][1].set_color(ALICE_COLORS[0])
            Bob = VGroup(
                Text(r' Bob ', color=BOB_COLORS[0], font_size=50),
                MathTex(fr'\ell = ', f'3')
            ).arrange(DOWN)
            Bob[1][1].set_color(BOB_COLORS[0])
            main_label = Bob
            other_label = Alice
            sk_num = sk_B
            main_paths = B_path_one, B_path_two
            main_subgraph = l3_subgraph
        elif self.perspective == "Both":
            self.add(overlay)
            self.wait(5)
            # last_alice_vert, last_alice_edges = new_mobjects_on_level[-1]
            # last_alice_edge = sike_graph.get_edge(
            #     A_path_two[-1], A_path_two[-2]
            # )
            # second_to_last_alice_edge = sike_graph.get_edge(
            #     A_path_two[-2], A_path_two[-3]
            # )
            # new_mobjects_on_level[-1] = VGroup(last_alice_vert, VGroup(second_to_last_alice_edge))

        # Play out intro for Alice and Bob
        if self.perspective != "Both":
            self.play(Write(main_label),run_time=1)
            num_tex, chosen = choose_random_int(self, 1, perspective_l**self.e2, predetermined_outcome=sk_num, center=DOWN, font_size=50)
            sk_tex = MathTex(rf'sk_{abrev_main} =').next_to(num_tex, LEFT)
            self.play(Write(sk_tex),run_time=1)
            sk_long_str = []
            for bi in to_base(chosen, perspective_l):
                sk_long_str.append(str(bi))
            if len(sk_long_str) < perspective_exp:
                sk_long_str = ["0"] * (perspective_exp - len(sk_long_str)) + sk_long_str
            extra_sk = MathTex(rf'\mapsto ', *sk_long_str).scale_to_fit_height(num_tex.get_height()).next_to(num_tex, RIGHT)
            self.play(Write(extra_sk),run_time=1)
            VGroup(*extra_sk[1:]).set_color(main_color)
            self.wait(1)
            sk_gp = VGroup(sk_tex, num_tex, extra_sk)
            self.play(
                sk_gp.animate.arrange(RIGHT).to_corner(DL),
                main_label[0].animate.to_corner(DL).shift(UP * 0.6),
                TransformMatchingTex(
                    main_label[1], overlay[1], replace_mobject_with_target_in_scene=True
                ),
                run_time=1
            )
            perspective_label = VGroup(
                main_label[0],
                sk_gp
            )
            self.add(perspective_label)

            # Add edge labels.
            for i in range(1, len(main_paths[0])):
                edge = sike_graph.get_edge(
                    main_paths[0][i - 1], main_paths[0][i]
                )
                edge_label_str = sk_long_str[i - 1]
                edge_label = MathTex(edge_label_str, font_size=50).set_color(main_color)
                perp_direction = rotate_vector(edge[0].get_end() - edge[0].get_start(), PI / 2)
                edge_label.move_to(
                    edge[0].get_center() + perp_direction * 0.5
                )
                edge.add(edge_label)
            for i in range(1, len(main_paths[1])):
                edge = sike_graph.get_edge(
                    main_paths[1][i - 1], main_paths[1][i]
                )
                edge_label_str = sk_long_str[i - 1]
                edge_label = MathTex(edge_label_str, font_size=50).set_color(main_color)
                perp_direction = rotate_vector(edge[0].get_end() - edge[0].get_start(), PI / 2)
                edge_label.move_to(
                    edge[0].get_center() + perp_direction * 0.5
                )
                edge.add(edge_label)

        # Play the scene
        graph_mobjects_introduced = [*new_mobjects_on_level[0]]
        self.play(DrawBorderThenFill(new_mobjects_on_level[0][0]))
        for i in range(1, len(new_mobjects_on_level)):
            prev_vertices, prev_edges = new_mobjects_on_level[i - 1]
            vertices, edges = new_mobjects_on_level[i]
            run_time = 0.5
            if self.perspective == "Both":
                if level_for_swap == i:
                    # needs to be adjusted to match the animation time for swapping labels in Alice and Bob Perspective Scenes.
                    self.wait(20)

                if i < len(new_mobjects_on_level) - 1:
                    # Creation animations
                    self.play(
                        LaggedStart(
                            *[vertex[0].animate(rate_func=rate_functions.there_and_back_with_pause)
                            .set_color(ALICE_COLORS[1] if vertex[0].get_color() == ALICE_COLORS[0] else BOB_COLORS[1]) for vertex in prev_vertices],
                            AnimationGroup(*[Create(edge) for edge in edges]),
                            lag_ratio=0.5,
                        ),
                        run_time=run_time,
                    )
                    self.play(
                        DrawBorderThenFill(vertices, run_time=run_time)
                    )
                else:
                    # Creation animations
                    self.play(
                        LaggedStart(
                            *[vertex[0].animate(rate_func=rate_functions.there_and_back_with_pause)
                            .set_color(ALICE_COLORS[1] if vertex[0].get_color() == ALICE_COLORS[0] else BOB_COLORS[1]) for vertex in prev_vertices],
                            AnimationGroup(*[Create(edges[1])]),
                            lag_ratio=0.5,
                        ),
                        run_time=run_time,
                    )
                    self.play(
                        DrawBorderThenFill(vertices, run_time=run_time)
                    )
            else:
                if level_for_swap == i:
                    pk_main_vertex = sike_graph.get_vertex(main_paths[0][-1])
                    pk_main = sike_graph.get_vertex_id_by_label_str(pk_main_vertex[2].get_tex_string())
                    pk_main_fq = Fq.from_integer(pk_main)
                    # pk_main_mathtex_simplified = MathTex(fr"j_{abrev_main}", ",", fr"\varphi_{abrev_main}( {main_torsion_string} )", font_size=50)
                    pk_main_mathtex_expanded = MathTex(
                        latex(pk_main_fq),
                        ", ",
                        r"\varphi",
                        "(",
                        *main_torsion_string,
                        ")",
                        font_size=40,
                        tex_to_color_map={
                            rf"\varphi": main_color,
                        },
                    )
                    pk_main_mathtex_torsion_points = MathTex(
                        r", \langle",
                        latex(main_pushed_torsion[0]),
                        ", ",
                        latex(main_pushed_torsion[1]),
                        r"\rangle",
                        font_size=30
                    )
                    pk_main_j_mathtex = pk_main_mathtex_expanded[0].set_color(main_color).copy()
                    pk_main_P_torsion_mathtex = pk_main_mathtex_torsion_points[1].set_color(main_color).copy()
                    pk_main_Q_torsion_mathtex = pk_main_mathtex_torsion_points[3].set_color(main_color).copy()
                    pk_main_j_mathtex.next_to(pk_main_vertex, DOWN)
                    pk_scaling_factor = sk_gp.get_height() / pk_main_vertex[2].get_height()
                    pk_main_label = (
                        VGroup(
                            MathTex(
                                rf"pk_{abrev_main} = j(",
                                "E",
                                ")",
                                r" = ",
                                font_size=40,
                                tex_to_color_map={
                                    "E": main_color,
                                },
                            ),
                            pk_main_mathtex_expanded,
                        )
                        .arrange(RIGHT, aligned_edge=DOWN)
                        .next_to(perspective_label, UP, buff=0.6)
                        .align_to(sk_gp, LEFT)
                    )

                    other_label.to_corner(DR)
                    pk_other_vertex = sike_graph.get_vertex(main_paths[1][0])
                    pk_other = sike_graph.get_vertex_id_by_label_str(pk_other_vertex[2].get_tex_string())
                    pk_other_fq = Fq.from_integer(pk_other)
                    # pk_other_mathtex_simplified = MathTex(fr"j_{abrev_other}", ",", fr"\varphi_{abrev_other}( {other_torsion_string} )", font_size=50)
                    pk_other_mathtex_expanded = MathTex(
                        latex(pk_other_fq),
                        ", ",
                        r"\varphi",
                        "(",
                        *other_torsion_string,
                        ")",
                        font_size=40,
                        tex_to_color_map={
                            rf"\varphi": other_color,
                        },
                    )
                    pk_other_mathtex_torsion_points = MathTex(
                        r", \langle",
                        latex(other_pushed_torsion[0]),
                        ", ",
                        latex(other_pushed_torsion[1]),
                        r"\rangle",
                        font_size=30
                        )
                    pk_other_j_mathtex = pk_other_mathtex_expanded[0].set_color(other_color).copy()
                    pk_other_P_torsion_mathtex = pk_other_mathtex_torsion_points[1].set_color(other_color).copy()
                    pk_other_Q_torsion_mathtex = pk_other_mathtex_torsion_points[3].set_color(other_color).copy()
                    pk_other_label = (
                        VGroup(
                            MathTex(
                                rf"pk_{abrev_other} =j(",
                                "E",
                                ")",
                                r" = ",
                                font_size=40,
                                tex_to_color_map={
                                    "E": other_color,
                                },
                            ),
                            pk_other_mathtex_expanded,
                        )
                        .arrange(RIGHT, aligned_edge=DOWN)
                        .next_to(other_label, UP)
                        .align_to(other_label, RIGHT)
                    )

                    self.play(
                        pk_main_vertex.animate().scale(
                            pk_scaling_factor,
                            about_point=pk_main_vertex.get_center(),
                            scale_stroke=True,
                        )
                    )
                    self.play(Circumscribe(pk_main_vertex))
                    self.wait()
                    self.play(Write(pk_main_label[0]))
                    self.play(GrowFromPoint(pk_main_j_mathtex, point=pk_main_vertex.get_center()))
                    self.wait()
                    self.play(ReplacementTransform(pk_main_j_mathtex, pk_main_label[1][0]), Write(pk_main_label[1][1:]))
                    self.play(
                        pk_main_vertex.animate().scale(
                            1/pk_scaling_factor,
                            about_point=pk_main_vertex.get_center(),
                            scale_stroke=True,
                        )
                    )
                    self.wait()
                    self.play(Write(other_label))
                    self.play(GrowFromPoint(pk_other_label[1], point=other_label.get_center()))
                    self.play(Write(pk_other_label[0]))
                    self.wait()
                    self.play(
                        FadeOut(pk_main_label[0], pk_other_label[0]), Swap(
                            pk_main_label[1], pk_other_label[1]
                        )
                    )
                    self.wait()
                    self.play(FadeOut(other_label, pk_main_label[1]), pk_other_label[1].animate().move_to(ORIGIN))
                    self.wait()
                    self.play(ReplacementTransform(pk_other_label[1], pk_other_vertex))
                    self.wait()

                seg_idx = i if i < level_for_swap else i - level_for_swap + 1
                path_indx = 0 if i < level_for_swap else 1
                if seg_idx >= len(main_paths[path_indx]):
                    perspective_vertex = None
                else:
                    perspective_vertex = sike_graph.get_vertex(main_paths[path_indx][seg_idx])
                    perspective_edge = sike_graph.get_edge(
                        main_paths[path_indx][seg_idx - 1], main_paths[path_indx][seg_idx]
                    )
                    previous_vertex = sike_graph.get_vertex(main_paths[path_indx][seg_idx - 1])

                if perspective_vertex is None:
                    self.wait(2 * run_time)
                else:
                    self.play(
                        LaggedStart(
                            previous_vertex[0].animate(rate_func=rate_functions.there_and_back_with_pause)
                            .set_color(main_color),
                            Create(perspective_edge),
                            lag_ratio=0.5,
                        ),
                        run_time=run_time,
                    )
                    self.play(
                        DrawBorderThenFill(perspective_vertex, run_time=run_time),
                    )

            graph_mobjects_introduced.extend([vertices, edges])
            # Group together mobjects that are in the scene compute translation and scalling that would center the graph
            graph_mobjects_introduced_vgp = VGroup(*graph_mobjects_introduced)
            translation_vector = (
                -graph_mobjects_introduced_vgp.get_center()
            )  # Centers the graph
            aspect_ratio = graph_mobjects_introduced_vgp.length_over_dim(
                0
            ) / graph_mobjects_introduced_vgp.length_over_dim(1)
            desired_aspect_ratio = self.camera.frame_width / self.camera.frame_height
            # We inflate scaling_factor to add padding to graph. We do this by effectively having frame dimensions be 0.9x the actual frame dimensions
            if desired_aspect_ratio > aspect_ratio:
                scaling_factor = (
                    0.8 * self.camera.frame_height / graph_mobjects_introduced_vgp.height
                )
            else:
                scaling_factor = (
                    0.8 * self.camera.frame_width / graph_mobjects_introduced_vgp.width
                )

            if scaling_factor < 1:
                if self.perspective != "Both":
                    def should_animate_mob_translation(mob):
                        return mob in self.mobjects or (
                            isinstance(mob, GraphVertexMobject)
                            and (
                                mob[0].get_color() == main_color
                                or (
                                    mob[0].get_color() == mixed_color
                                    and (
                                        sike_graph.get_vertex_id_by_label_str(
                                            mob[2].get_tex_string()
                                        )
                                        in [*main_paths[0], shared_sk_id] + ([main_paths[1][0]] if path_indx > 0 else [])
                                    )
                                )
                            )
                        )
                    self.play(
                        *[
                            mob.animate()
                            .shift(translation_vector)
                            .scale(
                                scaling_factor, about_point=ORIGIN, scale_stroke=True
                            )
                            for group in graph_mobjects_introduced
                            for mob in group
                            if should_animate_mob_translation(mob)
                        ],
                        run_time=run_time,
                    )
                    for group in graph_mobjects_introduced:
                        for mob in group:
                            if not should_animate_mob_translation(mob):
                                mob.shift(translation_vector).scale(
                                    scaling_factor, about_point=ORIGIN, scale_stroke=True
                                )
                else:
                    self.play(
                        *[
                            mob.animate()
                            .shift(translation_vector)
                            .scale(scaling_factor, about_point=ORIGIN, scale_stroke=True)
                            for group in graph_mobjects_introduced
                            for mob in group
                        ],
                        run_time=run_time,
                    )

                for j in range(i + 1, len(new_mobjects_on_level)):
                    unsceene_vertices, unscreen_edges = new_mobjects_on_level[j]
                    for mob in [*unsceene_vertices, *unscreen_edges]:
                        mob.shift(translation_vector).scale(
                            scaling_factor, about_point=ORIGIN, scale_stroke=True
                        )

        # End of scene. Must draw to last vertex for Alice since technically the last mob_object_introduced is her last unique vertex before the shared secret key.
        if self.perspective == "Alice":
            last_vertex = sike_graph.get_vertex(main_paths[1][-1])
            second_last_vertex = sike_graph.get_vertex(main_paths[1][-2])
            edge_connecting_last = sike_graph.get_edge(
                main_paths[1][-2], main_paths[1][-1]
            )
            self.play(
                LaggedStart(
                    second_last_vertex[0].animate(rate_func=rate_functions.there_and_back_with_pause)
                    .set_color(ALICE_COLORS[1]),
                    Create(edge_connecting_last),
                    lag_ratio=0.5,
                    run_time=1
                )
            )
            self.play(DrawBorderThenFill(last_vertex, run_time=0.5))
        elif self.perspective == "Bob":
            self.wait(2)
        elif self.perspective == "Both":
            second_last_vertex = sike_graph.get_vertex(A_path_two[-2])
            vertices, edges = new_mobjects_on_level[-1]      
            self.play(
                LaggedStart(
                    second_last_vertex[0].animate(rate_func=rate_functions.there_and_back_with_pause)
                    .set_color(ALICE_COLORS[1]),
                    Create(edges[0]),
                    lag_ratio=0.5,
                ),
                run_time=2
            )

        shared_vert = sike_graph.get_vertex(shared_sk_id)
        shared_vert_scaling_factor = p_label.get_height() / shared_vert[2].get_height()
        self.play(shared_vert.animate.scale(shared_vert_scaling_factor).set_color(YELLOW))
        ssk_tex = MathTex(latex(Fq.from_integer(shared_sk_id)), color=YELLOW, z_index = 3, font_size=50).center()
        self.play(GrowFromPoint(ssk_tex, point=shared_vert.get_center()))
        self.wait(3)
            


manim_configuration = {
    "quality": "production_quality",
    "preview": False,
    "output_file": "BobPerspective",
    "disable_caching": False,
    "max_files_cached": 1000,
    "write_to_movie": True,
    "show_file_in_browser": False,
}
if __name__ == '__main__':
    # e2, e3 = 5, 4
    # e2, e3 = 8, 5
    # e2, e3 = 18, 13
    # e2, e3 = 23, 16
    with tempconfig(manim_configuration):
        e2, e3 = 13, 7
        np.random.seed(27)
        scene = SIDHExchange(e2, e3, perspective="Bob")
        scene.render()
