from manim import *
from sage.all import *
from sage.schemes.elliptic_curves.hom_composite import EllipticCurveHom_composite
from SSGraphIntroduction import ALICE_COLORS, BOB_COLORS, PRIME_COLOR


class SikeParameters(Scene):
    def construct(self):
        bit12_params = [
            5,
            4,
            [(2552, 2357), (2586, 1247)],
            [(1920, 2033), (1033, 770)],
            [(137, 533), (333, 254)],
            [(763, 655), (2286, 1623)],
        ]

        e2 = bit12_params[0]
        e3 = bit12_params[1]
        p = (2**e2) * (3**e3) - 1
        Fq = FiniteField(p**2, "w")
        E0 = EllipticCurve(Fq, [0, 6, 0, 1, 0])
        torsion_subgroup_sizes = [2**e2, 3**e3]
        PA, QA = E0.torsion_basis(torsion_subgroup_sizes[0])
        PB, QB = E0.torsion_basis(torsion_subgroup_sizes[1])
        nA, mA = randrange(0, torsion_subgroup_sizes[0]), randrange(
            0, torsion_subgroup_sizes[0]
        )  # Secret keys
        nB, mB = randrange(0, torsion_subgroup_sizes[1]), randrange(
            0, torsion_subgroup_sizes[1]
        )
        A = nA * PA + QA
        B = nB * PB + QB
        PhiA = EllipticCurveHom_composite(E0, A)
        PhiB = EllipticCurveHom_composite(E0, B)
        EA = PhiA.codomain().montgomery_model()
        EB = PhiB.codomain().montgomery_model()

        def add_list_marker(tex_mob):
            list_marker = MathTex(r"\cdot")
            return VGroup(list_marker, tex_mob).arrange(RIGHT)

        def point_to_string(point):
            return f"({latex(point[0])}, {latex(point[1])})"

        content_font_size = 36
        public_head_label = Tex(r"Public Parameters:").to_corner(UL, buff=0.2)
        prime_label = MathTex(
            r"p = 2^{" + str(e2) + r"}3^{" + str(e3) + r"} - 1, \,",
            r"\ell_A = 2, \,",
            r"\ell_B = 3",
            font_size=content_font_size,
        )
        w_i_label = MathTex(
            r"\mathbb F_{p^2} = \mathbb F_p(w) ",
            font_size=content_font_size,
        )
        E0_label = MathTex(r"E_0", r": ", latex(E0), font_size=content_font_size)
        torsion_group_labels = MathTex(
            R"\langle P_A, Q_A \rangle = E_0[2] \, \, \,", R"\langle P_B, Q_B \rangle = E_0[3]", font_size=content_font_size
        )
        torsion_point_PA = MathTex(
            r"P_A =", f"{point_to_string(PA)}", font_size=content_font_size
        )
        torsion_point_QA = MathTex(
            r"Q_A = ", f"{point_to_string(QA)}", font_size=content_font_size
        )
        torsion_point_PB = MathTex(
            r"P_B =", f"{point_to_string(PB)}", font_size=content_font_size
        )
        torsion_point_QB = MathTex(
            r"Q_B = ", f"{point_to_string(QB)}", font_size=content_font_size
        )
        public_mobs = [
            add_list_marker(prime_label),
            add_list_marker(w_i_label),
            add_list_marker(E0_label),
            add_list_marker(torsion_group_labels),
            add_list_marker(torsion_point_PA),
            add_list_marker(torsion_point_QA),
            add_list_marker(torsion_point_PB),
            add_list_marker(torsion_point_QB),
        ]
        public_param_content_gp = (
            VGroup(*public_mobs)
            .scale(1)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.175)
            .next_to(public_head_label, DOWN, aligned_edge=LEFT)
            .shift(RIGHT * 0.5)
        )
        public_param_bounding_box = SurroundingRectangle(
            public_param_content_gp, color=PRIME_COLOR
        )

        secret_key_head_label = (
            Tex(r"Secret Key Example:").to_edge(LEFT, buff=0.2).shift(DOWN * 2)
        )
        alice_secret_key_int = MathTex(
            rf" n_A = ", f"{nA}", font_size=content_font_size
        )
        bob_secret_key_int = MathTex(rf" n_B = ", f"{nB}", font_size=content_font_size)
        alice_secret_key_isogeny = MathTex(
            r"\text{ or equivalently }",
            r"\varphi: E_0 \to E_A",
            r"\text{ with }",
            r"\ker \varphi = \langle n_AP_A + Q_A\rangle",
            font_size=content_font_size,
        )
        bob_secret_key_isogeny = MathTex(
            r"\text{ or equivalently }",
            r"\phi: E_0 \to E_B",
            r"\text{ with }",
            r"\ker \phi = \langle n_BP_B + Q_B\rangle",
            font_size=content_font_size,
        )
        secret_key_mobs = [
            add_list_marker(
                VGroup(alice_secret_key_int, alice_secret_key_isogeny).arrange(RIGHT)
            ),
            add_list_marker(
                VGroup(bob_secret_key_int, bob_secret_key_isogeny).arrange(RIGHT)
            ),
        ]
        secret_key_content_gp = (
            VGroup(*secret_key_mobs)
            .scale(1)
            .arrange(DOWN, aligned_edge=LEFT)
            .next_to(secret_key_head_label, DOWN, aligned_edge=LEFT)
            .shift(RIGHT * 0.5)
        )
        alice_secret_bounding_box = SurroundingRectangle(
            secret_key_content_gp[0], color=ALICE_COLORS[1]
        )
        bob_secret_bounding_box = SurroundingRectangle(
            secret_key_content_gp[1], color=BOB_COLORS[1]
        )

        public_key_head_label = (
            Tex(r"Public Key Example:").to_edge(UP, buff=0.2).shift(RIGHT * 2)
        )
        alice_public_curve = MathTex(
            r"E_A", r": ", latex(EA), font_size=content_font_size
        )
        alice_public_j_inv = MathTex(
            r"j(E_A) = ", latex(EA.j_invariant()), font_size=content_font_size
        )
        alice_public_torsion_PB = MathTex(
            r"\varphi(P_B) = ",
            f"{point_to_string(PhiA(PB))}",
            font_size=content_font_size,
        )
        alice_public_torsion_QB = MathTex(
            r"\varphi(Q_B) = ",
            f"{point_to_string(PhiA(QB))}",
            font_size=content_font_size,
        )
        bob_public_curve = MathTex(
            r"E_B", r": ", latex(EB), font_size=content_font_size
        )
        bob_public_j_inv = MathTex(
            r"j(E_B) = ", latex(EB.j_invariant()), font_size=content_font_size
        )
        bob_public_torsion_PA = MathTex(
            r"\varphi(P_A) = ",
            f"{point_to_string(PhiB(PA))}",
            font_size=content_font_size,
        )
        bob_public_torsion_QA = MathTex(
            r"\varphi(Q_A) = ",
            f"{point_to_string(PhiB(QA))}",
            font_size=content_font_size,
        )
        public_key_mobs = [
            add_list_marker(alice_public_curve),
            add_list_marker(alice_public_j_inv),
            add_list_marker(alice_public_torsion_PB),
            add_list_marker(alice_public_torsion_QB),
            add_list_marker(bob_public_curve),
            add_list_marker(bob_public_j_inv),
            add_list_marker(bob_public_torsion_PA),
            add_list_marker(bob_public_torsion_QA),
        ]
        public_key_content_gp = (
            VGroup(*public_key_mobs)
            .scale(1)
            .arrange(DOWN, aligned_edge=LEFT)
            .next_to(public_key_head_label, DOWN, aligned_edge=LEFT)
            .shift(RIGHT * 0.5)
        )
        alice_public_bounding_box = SurroundingRectangle(
            VGroup(*public_key_mobs[:4]), color=ALICE_COLORS[0]
        )
        bob_public_bounding_box = SurroundingRectangle(
            VGroup(*public_key_mobs[4:]), color=BOB_COLORS[0]
        )

        self.add(public_head_label, secret_key_head_label, public_key_head_label)
        self.play(Write(public_param_content_gp), run_time=3)
        self.play(Write(public_param_bounding_box), run_time=1)
        self.wait()
        self.play(Write(public_key_content_gp), run_time=3)
        self.play(Write(alice_public_bounding_box), run_time=1)
        self.play(Write(bob_public_bounding_box), run_time=1)
        self.wait()
        self.play(Write(secret_key_content_gp), run_time=3)
        self.play(Write(alice_secret_bounding_box), run_time=1)
        self.play(Write(bob_secret_bounding_box), run_time=1)
        self.wait(20)



manim_configuration = {
    "quality": "production_quality",
    "preview": False,
    "output_file": "SIKE12bitParameters",
    "disable_caching": False,
    "max_files_cached": 1000,
    "write_to_movie": True,
    "show_file_in_browser": False,
}
if __name__ == "__main__":
    with tempconfig(manim_configuration):
        np.random.seed(0)
        scene = SikeParameters()
        scene.render()
