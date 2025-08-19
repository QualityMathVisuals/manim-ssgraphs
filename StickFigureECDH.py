from manim import *
from sage.all import *
from QMVLib.StickFigures import *
from QMVLib.UsefulMobjects import *

ALICE_COLORS = (BLUE_B, GREEN_D)
BOB_COLORS = (ORANGE, RED_D)


class StickFigureECDH(Scene):
    def construct(self):
        p = 2063
        big_prime = p
        Fq = FiniteField(p)
        EC = EllipticCurve(Fq, [404, 1301])
        bob_log = 1176
        alice_log = 1339
        prim_root = EC.gens()[0]
        targets = bob_log * prim_root, alice_log * prim_root
        shared_secret = (alice_log * bob_log) * prim_root

        def point_to_string(point):
            return f"({point[0]}, {point[1]})"

        random_positions = []
        for i in range(10):
            random_positions.append(
                [np.random.uniform() * -8 + 4, np.random.uniform() * 1.5 + 3.5, 0]
            )
        for i in range(10):
            random_positions.append(
                [np.random.uniform() * 4 + 3, np.random.uniform() * 6, 0]
            )
        for i in range(10):
            random_positions.append(
                [np.random.uniform() * -4 - 3, np.random.uniform() * 6, 0]
            )

        alice_color = ALICE_COLORS[0]
        bob_color = BOB_COLORS[0]
        shared_color = interpolate_color(alice_color, bob_color, 0.5)
        bob_secret = MathTex(r"b = " + str(bob_log), color=bob_color).to_corner(
            UR, buff=0.1
        )
        alice_secret = MathTex(r"a = " + str(alice_log), color=alice_color).to_corner(
            UL, buff=0.1
        )
        alice = SlantedStickFigure(height=2, color=alice_color, facing=RIGHT).shift(
            LEFT * 1 + DOWN * 2
        )
        bob = SlantedStickFigure(height=2, color=bob_color, facing=LEFT).shift(
            RIGHT * 1 + DOWN * 2
        )
        eves = VGroup(
            *[
                StickFigure(height=1.0, color=RED_D).move_to(pos).set_opacity(0.6)
                for pos in random_positions
            ]
        )

        speech_bubble_alice = SpeechBubble().set_color(WHITE)
        speech_bubble_bob = SpeechBubble().set_color(WHITE)
        speech_bubble_alice.flip(RIGHT).flip().shift(LEFT * 2 + DOWN * 2)
        speech_bubble_bob.flip(RIGHT).shift(RIGHT * 2 + DOWN * 2)
        d_log_randomization_tex_alice = []
        for _ in range(60 * 8):
            rand = np.random.randint(big_prime - 1)
            if rand == (alice_log * bob_log) % big_prime:
                continue
            d_log_randomization_tex_alice.append(
                VGroup(
                    MathTex(r"{" + str(rand) + r"}", point_to_string(prim_root)),
                    MathTex(r"\stackrel{?}{=}"),
                    MathTex(point_to_string(targets[1])),
                )
                .align_to(ORIGIN, RIGHT)
                .arrange(DOWN)
                .set_color(RED)
            )
        d_log_randomization_tex_bob = []
        for _ in range(60 * 8):
            rand = np.random.randint(big_prime - 1)
            if rand == (alice_log * bob_log) % big_prime:
                continue
            d_log_randomization_tex_bob.append(
                VGroup(
                    MathTex(r"{" + str(rand) + r"}", point_to_string(prim_root)),
                    MathTex(r"\stackrel{?}{=}"),
                    MathTex(point_to_string(targets[0])),
                )
                .align_to(ORIGIN, RIGHT)
                .arrange(DOWN)
                .set_color(RED)
            )

        alice_thoughts_center = UP * 0.7 + LEFT * 5
        alice_thoughts = [
            MathTex(r"a", point_to_string(prim_root))
            .move_to(alice_thoughts_center)
            .set_color(alice_color),
            MathTex(str(alice_log), point_to_string(prim_root))
            .move_to(alice_thoughts_center)
            .set_color(alice_color),
            MathTex(point_to_string(targets[1]))
            .move_to(alice_thoughts_center)
            .set_color(alice_color),
            MathTex(point_to_string(targets[0]))
            .move_to(alice_thoughts_center)
            .set_color(bob_color),
            MathTex(r"a", point_to_string(targets[0]))
            .move_to(alice_thoughts_center)
            .set_color(bob_color),
            MathTex(str(alice_log), point_to_string(targets[0]))
            .move_to(alice_thoughts_center)
            .set_color(bob_color),
            MathTex(point_to_string(shared_secret))
            .move_to(alice_thoughts_center)
            .set_color(shared_color),
        ]
        alice_speech = [
            MathTex(point_to_string(targets[1]))
            .move_to(speech_bubble_alice.get_center())
            .set_color(alice_color),
        ]
        alice_thoughts[4][0].set_color(alice_color)
        alice_thoughts[5][0].set_color(alice_color)
        all_alice_speech = VGroup(*alice_speech)
        d_log_randomization_tex_alice = (
            VGroup(*d_log_randomization_tex_alice)
            .scale(3 / 4)
            .move_to(all_alice_speech.get_center())
            .shift(UP)
        )

        bob_thoughts_center = UP * 0.7 + RIGHT * 5
        bob_thoughts = [
            MathTex(r"b", point_to_string(prim_root))
            .move_to(bob_thoughts_center)
            .set_color(bob_color),
            MathTex(str(bob_log), point_to_string(prim_root))
            .move_to(bob_thoughts_center)
            .set_color(bob_color),
            MathTex(point_to_string(targets[0]))
            .move_to(bob_thoughts_center)
            .set_color(bob_color),
            MathTex(point_to_string(targets[1]))
            .move_to(bob_thoughts_center)
            .set_color(alice_color),
            MathTex(r"b", point_to_string(targets[1]))
            .move_to(bob_thoughts_center)
            .set_color(alice_color),
            MathTex(str(bob_log), point_to_string(targets[1]))
            .move_to(bob_thoughts_center)
            .set_color(alice_color),
            MathTex(point_to_string(shared_secret))
            .move_to(bob_thoughts_center)
            .set_color(shared_color),
        ]
        bob_speech = [
            MathTex(point_to_string(targets[0]))
            .move_to(speech_bubble_bob.get_center())
            .set_color(bob_color),
        ]
        bob_thoughts[4][0].set_color(bob_color)
        bob_thoughts[5][0].set_color(bob_color)
        all_bob_speech = VGroup(*bob_speech)
        VGroup(*d_log_randomization_tex_bob).scale(3 / 4).move_to(
            all_bob_speech.get_center()
        ).shift(UP)

        public_info_board = BillBoard(
            width=3, height=2, fill_color=DARK_GRAY, fill_opacity=0.6, stroke_width=4
        ).shift(UP * 2)
        public_info_board_text = (
            VGroup(
                Text(r"Public:", font_size=38),
                MathTex(r"p = " + str(big_prime), font_size=32),
                MathTex(latex(EC), font_size=32),
                MathTex(r"P = ", point_to_string(prim_root), font_size=32),
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .scale(0.75)
            .move_to(public_info_board.get_center() + UP * 0.3)
        )
        public_info = VGroup(public_info_board, public_info_board_text)

        bottom_text_alice = VGroup(
            MathTex(r"b", point_to_string(prim_root))
            .to_edge(DOWN)
            .shift(LEFT * 2)
            .set_color(bob_color),
            MathTex(r"a", r"(", r"b", point_to_string(prim_root), r")")
            .to_edge(DOWN)
            .shift(LEFT * 2)
            .set_color(bob_color),
        )
        bottom_text_shared = MathTex(r" = ").to_edge(DOWN).shift(UP * 0.2)
        bottom_text_bob = VGroup(
            MathTex(r"a", point_to_string(prim_root))
            .to_edge(DOWN)
            .shift(RIGHT * 2)
            .set_color(alice_color),
            MathTex(r"b", r"(", r"a", point_to_string(prim_root), r")")
            .to_edge(DOWN)
            .shift(RIGHT * 2)
            .set_color(alice_color),
        )
        bottom_text_alice[1][0].set_color(alice_color)
        bottom_text_alice[1][1].set_color(alice_color)
        bottom_text_alice[1][4].set_color(alice_color)
        bottom_text_bob[1][0].set_color(bob_color)
        bottom_text_bob[1][1].set_color(bob_color)
        bottom_text_bob[1][4].set_color(bob_color)
        middle_eve = (
            StickFigure(height=1.0, color=RED_D)
            .set_opacity(0.6)
            .scale(1.5)
            .shift(RIGHT + DOWN * 1.2)
        )

        self.add(alice, bob)
        self.play(FadeIn(public_info_board), Write(public_info_board_text))
        self.play(Write(eves))
        self.play(
            VGroup(alice, bob).animate.scale(5).shift(DOWN * 3),
            eves.animate.scale(3),
            public_info.animate.scale(1.2).shift(UP * 0.5),
        )
        self.play(Write(alice_secret))
        self.play(Write(bob_secret))
        self.wait()
        self.play(TransformFromCopy(public_info_board_text[3][1], alice_thoughts[0][1]))
        self.play(Write(alice_thoughts[0][0]))
        self.wait(0.5)
        self.play(TransformMatchingTex(alice_thoughts[0], alice_thoughts[1]))
        self.wait(0.5)
        self.play(FadeTransform(alice_thoughts[1], alice_thoughts[2]))
        self.wait()
        # At this point, Alice has her public key.

        self.play(Write(speech_bubble_alice))
        self.play(FadeTransform(alice_thoughts[2], alice_speech[0]))
        self.wait(0.5)
        self.play(FadeIn(middle_eve))
        self.play(
            alice_speech[0].animate.move_to(bottom_text_alice[0].get_center()),
            FadeOut(speech_bubble_alice),
        )
        self.play(Write(d_log_randomization_tex_alice[0]))
        self.wait(0.5)

        # Now do the randomization phase.
        opaque_duration = 3
        fade_out_duration = 1
        for i in range(int(60 * opaque_duration)):
            self.wait(1 / 60)
            self.add(d_log_randomization_tex_alice[i + 1])
            self.remove(d_log_randomization_tex_alice[i])

        for i in range(int(60 * fade_out_duration)):
            self.wait(1 / 60)
            index_curr = i + int(60 * opaque_duration)
            self.add(d_log_randomization_tex_alice[index_curr + 1])
            self.remove(d_log_randomization_tex_alice[index_curr])
            diff = 60 * fade_out_duration - i
            d_log_randomization_tex_alice[index_curr + 1].set_opacity(
                diff / (60 * fade_out_duration)
            )

        self.play(FadeOut(middle_eve))
        # Now repeat the same for Bob.
        middle_eve.shift(LEFT * 2)

        self.play(TransformFromCopy(public_info_board_text[3][1], bob_thoughts[0][1]))
        self.play(Write(bob_thoughts[0][0]))
        self.wait(0.5)
        self.play(TransformMatchingTex(bob_thoughts[0], bob_thoughts[1]))
        self.wait(0.5)
        self.play(FadeTransform(bob_thoughts[1], bob_thoughts[2]))
        self.wait()
        # At this point, Bob has his public key.

        self.play(Write(speech_bubble_bob))
        self.play(FadeTransform(bob_thoughts[2], bob_speech[0]))
        self.wait(0.5)
        self.play(FadeIn(middle_eve))
        self.play(
            bob_speech[0].animate.move_to(bottom_text_bob[0].get_center()),
            FadeOut(speech_bubble_bob),
        )
        self.play(Write(d_log_randomization_tex_bob[0]))
        self.wait(0.5)

        # Now do the randomization phase.
        for i in range(int(60 * opaque_duration)):
            self.wait(1 / 60)
            self.add(d_log_randomization_tex_bob[i + 1])
            self.remove(d_log_randomization_tex_bob[i])

        for i in range(int(60 * fade_out_duration)):
            self.wait(1 / 60)
            index_curr = i + int(60 * opaque_duration)
            self.add(d_log_randomization_tex_bob[index_curr + 1])
            self.remove(d_log_randomization_tex_bob[index_curr])
            diff = 60 * fade_out_duration - i
            d_log_randomization_tex_bob[index_curr + 1].set_opacity(
                diff / (60 * fade_out_duration)
            )

        self.play(FadeOut(middle_eve))

        # Next, we swap
        ephemeral_alice = alice_speech[0].copy()
        ephemeral_bob = bob_speech[0].copy()
        self.play(
            TransformMatchingTex(ephemeral_alice, bob_thoughts[3]),
            TransformMatchingTex(ephemeral_bob, alice_thoughts[3]),
            alice_speech[0].animate.shift(RIGHT * 4),
            bob_speech[0].animate.shift(LEFT * 4),
        )
        self.play(
            TransformMatchingTex(alice_speech[0], bottom_text_bob[0]),
            TransformMatchingTex(bob_speech[0], bottom_text_alice[0]),
        )
        self.wait()

        self.play(TransformMatchingTex(alice_thoughts[3], alice_thoughts[4]))
        self.play(TransformMatchingTex(bottom_text_alice[0], bottom_text_alice[1]))
        self.wait()
        self.play(TransformMatchingTex(bob_thoughts[3], bob_thoughts[4]))
        self.play(TransformMatchingTex(bottom_text_bob[0], bottom_text_bob[1]))
        self.wait()
        self.play(TransformMatchingTex(alice_thoughts[4], alice_thoughts[5]))
        self.play(TransformMatchingTex(bob_thoughts[4], bob_thoughts[5]))
        self.wait()
        self.play(Transform(alice_thoughts[5], alice_thoughts[6]))
        self.play(Transform(bob_thoughts[5], bob_thoughts[6]))
        self.wait()
        self.play(Write(bottom_text_shared))
        self.wait(3)


manim_configuration = {
    "quality": "production_quality",
    "preview": False,
    "output_file": "StickFigureECDH",
    "disable_caching": False,
    "max_files_cached": 1000,
    "write_to_movie": True,
    "show_file_in_browser": False,
}
if __name__ == "__main__":
    with tempconfig(manim_configuration):
        np.random.seed(0)
        scene = StickFigureECDH()
        scene.render()
