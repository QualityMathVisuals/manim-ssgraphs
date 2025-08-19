from manim import *
from SIKEProtocol import prime_overlay_label, ALICE_COLORS, BOB_COLORS, get_initial_sage_data, get_SIKE_paths
from SupersingularIsogenyGraphs import IsogenyDiamond, get_random_seperated_vectors

class KaniFolding(Scene):
    def __init__(self, e2=5, e3=4):
        self.e2 = e2
        self.e3 = e3
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

        sike_graph.center()
        sike_graph.scale(1/5)

        self.add(sike_graph)

        folded_graph = sike_graph.copy()
        sike_graph.set_opacity(0.2)

        E0_vert = folded_graph.get_vertex(A_path_one[0])
        EA_vert = folded_graph.get_vertex(A_path_one[-1])
        EB_vert = folded_graph.get_vertex(B_path_one[-1])
        Es_vert = folded_graph.get_vertex(shared_sk_id)

        E0_to_EA = [folded_graph.get_vertex(a_path_id) for a_path_id in A_path_one]
        E0_to_EB = [folded_graph.get_vertex(b_path_id) for b_path_id in B_path_one]
        EA_to_Es = [folded_graph.get_vertex(a_path_id) for a_path_id in A_path_two]
        EB_to_Es = [folded_graph.get_vertex(b_path_id) for b_path_id in B_path_two]

        E0_vert.scale(4).move_to(LEFT * 5 + UP * 2)
        EA_vert.scale(4).move_to(RIGHT * 5 + UP * 2)
        EB_vert.scale(4).move_to(RIGHT * 5 + DOWN * 2)
        Es_vert.scale(4).move_to(LEFT * 5 + DOWN * 2)

        e2_unit = 10 / self.e2
        e3_unit = 10 / self.e3
        # def rightward_vec(e2 = True):
        #     if e2:
        #         return get_random_seperated_vectors(RIGHT, 3, e2_unit)[np.random.randint(3)]
        #     return get_random_seperated_vectors(RIGHT, 4, e3_unit)[np.random.randint(4)]

        # for i, vertex in enumerate(E0_to_EA):
        #     rightward

        self.add(folded_graph)






manim_configuration = {
    "quality": "high_quality",
    "preview": False,
    "output_file": "PreviewVideo",
    "disable_caching": False,
    "max_files_cached": 1000,
    "write_to_movie": True,
    "show_file_in_browser": False,
}
if __name__ == '__main__':
    with tempconfig(manim_configuration):
        e2, e3 = 13, 7
        np.random.seed(0)
        scene = KaniFolding(e2, e3)
        scene.render()