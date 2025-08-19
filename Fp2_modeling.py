from manim import *
from sage.all import *
from sage.rings.factorint import factor_trial_division
from sage.schemes.elliptic_curves.mod_poly import classical_modular_polynomial
from SupersingularIsogenyGraphs import SupersingularIsogenyGraph, EmbeddedSupersingularIsogenyGraph

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


def get_initial_sage_data(p, modulus=None):
    if modulus is not None:
        Fq = FiniteField(p**2, "w", modulus=modulus)
    elif p % 4 == 3:
        Fq = FiniteField(p**2, "w", modulus=[1, 0, 1])
    elif p % 3 == 2:
        Fq = FiniteField(p**2, "w")
    else:
        raise Exception("p must be congruent to 3 mod 4 or 2 mod 3")

    w = Fq.gen()
    E0 = EllipticCurve(Fq, [1, 0] if p % 4 == 3 else [0, 1])
    initial_j_invariant = E0.j_invariant()

    return Fq, w, E0, initial_j_invariant


def get_entire_sage_data_by_level(p, ell, maximum_depth=100, modulus=None):
    Fq, w, E0, initial_j_invariant = get_initial_sage_data(p, modulus=modulus)
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


class Fp2Embedding(Scene):
    def __init__(
        self,
        p=2063,
        ell=2,
        embedding_func=None,
        virtual_y_offset=0,
        modulus=None,
    ):
        self.ell = ell
        self.p = p
        self.virtual_y_offset = int(virtual_y_offset)  # Can range from [0, p - 1]

        # Default embedding: (a, b) -> (a/p, b/p) normalized to [0,1]^2
        def y_offset_corrected_embedding(field_element):
            """Embedding that accounts for the vertical offset"""
            f = embedding_func or self._default_embedding
            x, y = f(field_element)
            adjusted_zero_y_pos = self.virtual_y_offset / (self.p - 1)
            y += adjusted_zero_y_pos
            return x, y % 1

        self.embedding_func = y_offset_corrected_embedding
        if modulus is not None:
            self.modulus = modulus
        else:
            self.modulus = None
        super().__init__()

    def _default_embedding(self, field_element):
        """Default embedding with virtual coordinate wrapping"""
        p = self.p
        a, b = float(field_element[0]), float(field_element[1])
        return (a / p, b / p)

    def _get_generator_info(self, gen):
        """Get information about the field generator for display"""
        gen_symbol = "i" if gen**2 == -1 else r"w"
        if gen**2 == -1:
            gen_relation = rf"{gen_symbol}^2 = -1"
            field_desc = r"\mathbb{F}_{p^2} = \mathbb{F}_{p}(i)"
        else:
            if self.virtual_y_offset is not None:
                gensqrd = gen**2
                gen_relation = rf"{gen_symbol}^2 = {gensqrd[0]}{gen_symbol} + {gensqrd[1]}"
            else:
                gen_relation = rf"{gen_symbol}^2 = {latex(gen**2)}"
            gen_relation = rf"{gen_symbol}^2 = {latex(gen**2)}"
            field_desc = r"\mathbb{F}_{p^2} = \mathbb{F}_{p}(w)"

        gen_symbol = "i" if gen**2 == -1 else r"w"
        embedding_description = Rf"a + b{gen_symbol} \mapsto (a, b)"
        return gen_symbol, gen_relation, field_desc, embedding_description
    
    def _get_frobenius_action_info(self, Fq):
        """Get information about the Frobenius action for display"""
        frobenius = Fq.frobenius_endomorphism()
        matrix_cols = [[1, 0]]
        gen_acted = frobenius(Fq.gen())
        matrix_cols += [[gen_acted[0], gen_acted[1]]]
        frobenius_matrix = Matrix(Fq, matrix_cols)
        matrix_str = f"1 & {gen_acted[0]} \\\\ 0 & -1"
        frobenius_matrix_rep = r"\begin{bmatrix}" + matrix_str + r"\end{bmatrix}"
        return frobenius_matrix, frobenius_matrix_rep

    def construct(self):
        ell = self.ell
        p = self.p

        # Get field data for generator info
        Fq, w, E0, initial_j_invariant = get_initial_sage_data(p, self.modulus)
        gen_symbol, gen_relation, field_desc, embedding_description = (
            self._get_generator_info(w)
        )
        frobenious_matrix, frobenius_matrix_str = self._get_frobenius_action_info(Fq)

        # Create overlay labels
        p_label = prime_overlay_label(p)
        l_label = MathTex(
            rf"\ell = ", ell, tex_to_color_map={f"{ell}": ALICE_COLORS[0]}
        )
        info_label = VGroup(
            MathTex(gen_relation),
            MathTex(field_desc),
            MathTex(embedding_description),
            MathTex(r"\pi: ", frobenius_matrix_str),
        ).arrange(DOWN, aligned_edge=LEFT)

        overlay = (
            VGroup(p_label, l_label, info_label)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UL, buff=0.5)
            .add_background_rectangle(
                color=BLACK, opacity=0.6, buff=0.2, corner_radius=0.1
            )
        )
        self.add(overlay)

        print("Performing sage calculations...")
        # Fq, w, E0, initial_j_invariant already computed above
        maximum_depth = 20 if p < 10**7 else 10
        levels, adjacency_list = get_entire_sage_data_by_level(
            p, ell, maximum_depth=maximum_depth, modulus=self.modulus
        )

        print("Creating graph mobjects...")
        graph_shift = 2 * RIGHT
        frame_size = 7

        ss_graph = EmbeddedSupersingularIsogenyGraph(
            Fq,
            levels,
            adjacency_list,
            embedding_func=self.embedding_func,
            vertex_scaling=0.25,
            coordinate_plane_size=frame_size,
        ).shift(graph_shift)

        VGroup(*ss_graph.edges.values()).set_stroke(width=0.5)

        # Create bounding box and labels
        bounding_box = Rectangle(
            color=PINK, height=frame_size, width=frame_size, stroke_width=2, z_index=-2
        ).shift(graph_shift)

        # Create coordinate labels based on virtual coordinate system
        frame_offset = frame_size / 2
        virtual_y_top = (p - 1) - self.virtual_y_offset
        virtual_y_bottom = -self.virtual_y_offset

        bottom_label = MathTex(f"{virtual_y_bottom}").next_to(
            graph_shift + np.array([-frame_offset, -frame_offset, 0]),
            LEFT,
            aligned_edge=RIGHT,
        )
        top_label = MathTex(f"{virtual_y_top}").next_to(
            graph_shift + np.array([-frame_offset, frame_offset, 0]),
            LEFT,
            aligned_edge=RIGHT,
        )

        # Calculate where y=0 appears in the display
        zero_ratio = (-virtual_y_bottom) / (virtual_y_top - virtual_y_bottom)
        zero_y_pos = -frame_offset + zero_ratio * frame_size
        y_zero_line = Line(
            graph_shift + np.array([-frame_offset, zero_y_pos, 0]),
            graph_shift + np.array([frame_offset, zero_y_pos, 0]),
            color=PINK,
            stroke_width=1,
            z_index=-2,
        )
        zero_line_label = (
            MathTex("0")
            .next_to(
                graph_shift + np.array([-frame_offset, zero_y_pos, 0]),
                LEFT,
                aligned_edge=RIGHT,
            )
            .scale(0.8)
        )

        graph_labels = VGroup(bottom_label, top_label, zero_line_label)

        self.add(bounding_box, ss_graph, graph_labels, y_zero_line)

        

        def frobenius_action(x):
            return x**p

        frobenius_graph_copy = EmbeddedSupersingularIsogenyGraph(
            Fq,
            levels,
            adjacency_list,
            embedding_func=lambda x: self.embedding_func(frobenius_action(x)),
            coordinate_plane_size=frame_size,
        ).shift(graph_shift)
        # for vertex in frobenius_graph_copy.vertices.values():
        #     vertex[0].set_color(ORANGE)

        def remove_spine(graph):
            self.remove(graph)
            for vertex_id in list(graph.vertices.keys()):
                if vertex_id < p:
                    graph.remove(graph.vertices[vertex_id])

            for edge_id in list(graph.edges.keys()):
                if edge_id[0] < p or edge_id[1] < p:
                    graph.remove(graph.edges[edge_id])

            self.add(graph)

        def isolate_spine(graph, add_color=True):
            self.remove(graph)
            for vertex_id in list(graph.vertices.keys()):
                if vertex_id >= p:
                    graph.remove(graph.vertices[vertex_id])
                elif add_color:
                    graph.vertices[vertex_id][0].set_color(PRIME_COLOR)

            for edge_id in list(graph.edges.keys()):
                if edge_id[0] >= p or edge_id[1] >= p:
                    graph.remove(graph.edges[edge_id])
            self.add(graph)

        def top_half(graph):
            self.remove(graph)
            halfway_int = (p - 1) // 2
            for vertex_id in list(graph.vertices.keys()):
                elt = Fq.from_integer(vertex_id)
                if elt[1] > halfway_int:
                    graph.remove(graph.vertices[vertex_id])

            for edge_id in list(graph.edges.keys()):
                elt1 = Fq.from_integer(edge_id[0])
                elt2 = Fq.from_integer(edge_id[1])
                if elt1[1] > halfway_int or elt2[1] > halfway_int:
                    graph.remove(graph.edges[edge_id])
            self.add(graph)

        def remove_conjugate_edges(graph):
            self.remove(graph)
            conjugate_edges = []
            for edge_id in list(graph.edges.keys()):
                elt1 = Fq.from_integer(edge_id[0])
                elt2 = Fq.from_integer(edge_id[1])
                if elt1[1]  == -elt2[1] + p and elt1[1] != elt2[1]:
                    conjugate_edges.append(graph.edges[edge_id])
                    graph.remove(graph.edges[edge_id])

            conjugate_edges = VGroup(*conjugate_edges).set_color(ORANGE).set_stroke(width=2)
            self.add(graph, conjugate_edges)

        def strictly_crossing(graph):
            remove_spine(graph)
            halfway_int = (p - 1) // 2
            self.remove(graph)
            for edge_id in list(graph.edges.keys()):
                elt1 = Fq.from_integer(edge_id[0])
                elt2 = Fq.from_integer(edge_id[1])
                elt1_top = elt1[1] > halfway_int
                elt2_top = elt2[1] > halfway_int
                if elt1_top == elt2_top:
                    graph.remove(graph.edges[edge_id])

            self.add(graph)
        # self.play((Transform(ss_graph[i], frobenius_graph_copy[i]) for i in range(len(ss_graph))), run_time=8)
        # remove_spine(ss_graph)
        # isolate_spine(ss_graph)
        # remove_spine(frobenius_graph_copy)
        # top_half(ss_graph)
        # strictly_crossing(ss_graph)

        def top_half_no_isolated_vertices(graph):
            self.remove(graph)
            halfway_int = (p - 1) // 2
            incidence_dict = {v_id: [] for v_id in graph.vertices.keys()}
            for edge_id in graph.edges.keys():
                incidence_dict[edge_id[0]].append(edge_id[1])
                incidence_dict[edge_id[1]].append(edge_id[0])

            for vertex_id in list(graph.vertices.keys()):
                elt = Fq.from_integer(vertex_id)
                if elt[1] > halfway_int:
                    graph.remove(graph.vertices[vertex_id])
                    for neighbor in incidence_dict[vertex_id]:
                        incidence_dict[neighbor].remove(vertex_id)
                        graph.remove(graph.edges[tuple(sorted((vertex_id, neighbor)))])
                    del incidence_dict[vertex_id]

            def remove_terminating_vertices(incidence_dict, graph):
                for vertex_id in list(incidence_dict.keys()):
                    incidence_count = len(incidence_dict[vertex_id])
                    if incidence_count <= 1:
                        if incidence_count == 1 and vertex_id < p:
                            # Keep the prime vertices even if they are terminal
                            continue
                        graph.remove(graph.vertices[vertex_id])
                        for neighbor in incidence_dict[vertex_id]:
                            incidence_dict[neighbor].remove(vertex_id)
                            graph.remove(graph.edges[tuple(sorted((vertex_id, neighbor)))])
                        del incidence_dict[vertex_id]
                return incidence_dict
            
            def remove_spine_only_vertices(incidence_dict, graph):
                for vertex_id in list(incidence_dict.keys()):
                    if vertex_id < p:
                        only_spine_adjacent = all(neighbor_id < p for neighbor_id in incidence_dict[vertex_id])
                        if only_spine_adjacent:
                            graph.remove(graph.vertices[vertex_id])
                            for neighbor in incidence_dict[vertex_id]:
                                incidence_dict[neighbor].remove(vertex_id)
                                graph.remove(graph.edges[tuple(sorted((vertex_id, neighbor)))])
                            del incidence_dict[vertex_id]
                return incidence_dict
            
            prev_vertex_count = -1
            while len(incidence_dict) != prev_vertex_count:
                prev_vertex_count = len(incidence_dict)
                print(f"Remaining vertices: {len(incidence_dict)}")
                incidence_dict = remove_terminating_vertices(incidence_dict, graph)

            incidence_dict = remove_spine_only_vertices(incidence_dict, graph)


            sage_graph = Graph(incidence_dict)
            components = sage_graph.connected_components()
            largest_component = max(components, key=len)
            print(f"Largest component size: {len(largest_component)}")
            print(f"Total components: {len(components)}")
            print(components)
            self.add(graph)

        top_half_no_isolated_vertices(ss_graph)

manim_configuration = {
    "quality": "medium_quality",
    "preview": False,
    "output_file": "PreviewVideo",
    "disable_caching": False,
    "max_files_cached": 1000,
    "write_to_movie": True,
    "show_file_in_browser": False,
}
if __name__ == "__main__":
    with tempconfig(manim_configuration):
        np.random.seed(0)
        # p = 1019
        # Fp2Embedding(p=p).render()
        # Fp2Embedding(p=p, virtual_y_offset=(p - 1) / 2).render()
        # Fp2Embedding(p=p, virtual_y_offset=200).render()

        # # Important: w^p = w * (w^2)^((p-1)/2) which is always \legendre symbol (w^2, p) * w, so its always = -1.
        # # Basically, all moduli of the form x^2 - c where c is a quadratic residue mod p will work and give the same frobenius action
        # # If For w^2 = aw + b, then The actions is much more iteresting:
        # p = 1759
        # GpX = GF(p)["x"]
        # x = GpX.gen()
        # Fp2Embedding(p=p, virtual_y_offset=(p - 1) / 2).render()
        # Fp2Embedding(p=p, virtual_y_offset=(p - 1) / 2, modulus=x**2 - GpX(3)).render()
        # Fp2Embedding(
        #     p=p, virtual_y_offset=(p - 1) / 2, modulus=x**2 - 5 * x - GpX(108)
        # ).render()
        p = 8623
        p = 9767
        Fp2Embedding(p=p, virtual_y_offset=(p - 1) / 2).render()
        
