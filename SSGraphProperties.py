from manim import *
from SSGraphIntroduction import prime_overlay_label, get_initial_sage_data, ALICE_COLORS, BOB_COLORS, PRIME_COLOR
from SupersingularIsogenyGraphs import SupersingularIsogenyGraph
from sage.schemes.elliptic_curves.mod_poly import classical_modular_polynomial

def get_random_walk(p, ell, walk_length, non_backtrack=True):
    Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
    levels = [[initial_j_invariant.to_integer()]]
    all_vertex_ids = [initial_j_invariant.to_integer()]
    adjacency_list = (
        []
    )  # pairs of vertex ids in the graph ordered so that the first < second w.r.t jinv.to_integer()
    final_walk = selected_vertices = [initial_j_invariant.to_integer()]

    for i in range(1, walk_length + 1):
        current_selected_id = selected_vertices[i - 1]
        next_level = []
        # Get the neighbors of the selected vertex in the current level
        # and add them to the next level
        # Also, add the edges to the adjacency list and select a new vertex depending on the non_backtrack parameter
        selected_j_inv = Fq.from_integer(current_selected_id)
        modular_polynomial_evaluated = classical_modular_polynomial(ell, selected_j_inv)
        modular_roots = modular_polynomial_evaluated.roots(multiplicities=False)
        all_neighbor_ids = [
            neighbor_j.to_integer()
            for neighbor_j in modular_roots
            if neighbor_j != selected_j_inv
        ]
        new_neighbor_ids = [
            neighbor_id
            for neighbor_id in all_neighbor_ids
            if neighbor_id not in all_vertex_ids
        ]

        for neighbor_id in all_neighbor_ids:
            # Create the unique edge ID
            edge_id = tuple(sorted((current_selected_id, neighbor_id)))
            if edge_id not in adjacency_list:
                adjacency_list.append(edge_id)

        next_level.extend(new_neighbor_ids)
        all_vertex_ids.extend(new_neighbor_ids)
        all_vertex_ids = list(set(all_vertex_ids))  # Remove duplicates

        # Double check- Remove duplicates from the next level
        next_level = list(set(next_level))
        if len(next_level) == 0:
            break
        levels.append(next_level)

        # Select a new vertex in the next level
        if non_backtrack:
            selectable_vertices = [
                vertex_id
                for vertex_id in all_neighbor_ids
                if vertex_id not in selected_vertices
            ]
        else:
            selectable_vertices = all_neighbor_ids

        if len(selectable_vertices) == 0:
            print("No more vertices to select from so that no backtracking occurs")
            break
        selected_id = np.random.choice(selectable_vertices, 1)[0]
        selected_vertices.append(int(selected_id))

    return levels, adjacency_list, final_walk

def get_many_random_walks(p, walk_length, number_of_walks,  ell = None, non_backtrack=True):
    Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
    levels = [[initial_j_invariant.to_integer()]]
    all_vertex_ids = [initial_j_invariant.to_integer()]
    adjacency_list = (
        []
    )  # pairs of vertex ids in the graph ordered so that the first < second w.r.t jinv.to_integer()
    final_walks = [[initial_j_invariant.to_integer()] for _ in range(number_of_walks)]

    for i in range(1, walk_length + 1):
        next_level = []
        for j in range(number_of_walks):
            current_walk = final_walks[j]
            current_selected_id = current_walk[i - 1]
            # Get the neighbors of the selected vertex in the current level
            # and add them to the next level
            # Also, add the edges to the adjacency list and select a new vertex depending on the non_backtrack parameter
            selected_j_inv = Fq.from_integer(current_selected_id)
            if ell is None:
                ell = 2 if j == 0 else 3  # Default to 2 for the first walk, 3 for the rest
            modular_polynomial_evaluated = classical_modular_polynomial(ell, selected_j_inv)
            modular_roots = modular_polynomial_evaluated.roots(multiplicities=False)
            all_neighbor_ids = [
                neighbor_j.to_integer()
                for neighbor_j in modular_roots
                if neighbor_j != selected_j_inv
            ]
            new_neighbor_ids = [
                neighbor_id
                for neighbor_id in all_neighbor_ids
                if neighbor_id not in all_vertex_ids
            ]

            for neighbor_id in all_neighbor_ids:
                # Create the unique edge ID
                edge_id = tuple(sorted((current_selected_id, neighbor_id)))
                if edge_id not in adjacency_list:
                    adjacency_list.append(edge_id)

            next_level.extend(new_neighbor_ids)
            all_vertex_ids.extend(new_neighbor_ids)
            all_vertex_ids = list(set(all_vertex_ids))  # Remove duplicates

            # Double check- Remove duplicates from the next level
            next_level = list(set(next_level))

            # Select a new vertex in the next level
            if non_backtrack:
                selectable_vertices = [
                    vertex_id
                    for vertex_id in all_neighbor_ids
                    if vertex_id not in current_walk
                ]
            else:
                selectable_vertices = all_neighbor_ids

            if len(selectable_vertices) == 0:
                print("No more vertices to select from so that no backtracking occurs")
                break
            selected_id = np.random.choice(selectable_vertices, 1)[0]
            current_walk.append(int(selected_id))

        if len(next_level) == 0:
            break
        levels.append(next_level)

    return levels, adjacency_list, final_walks


class WalkableGraph(Scene):
    """
    A Scene that visualizes a random walk on a supersingular isogeny graph.

    The scene constructs a supersingular isogeny graph based on a given prime `p` and isogeny degree `ell`.
    It then performs a random walk on the graph and highlights the vertices and edges traversed during the walk.
    The scene progressively introduces the graph level by level, scaling and centering the graph to fit the screen.
    Finally, it displays the random walk and fades out the rest of the graph.

    SIKE:
        np.random.seed(0)
        manim_configuration[""output_file"] = "WalkableGraphplargel3d100"
        scene = WalkableGraph(p=2**216 * 3**137 - 1, ell=3, walk_length=100)
        scene.render()
    """
    def __init__(self, p=2063, ell=2, walk_length=20):
        self.walk_length = walk_length
        self.ell = ell
        self.p = p
        super().__init__()

    def construct(self):
        ell = self.ell
        p = self.p
        walk_length = self.walk_length

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
        levels, adjacency_list, final_walk = get_random_walk(
            p, ell, walk_length
        )

        print("Creating graph mobjects...")
        ss_graph = SupersingularIsogenyGraph(Fq, levels, adjacency_list)
        walk_vertices = [ss_graph.get_vertex(vertex_id) for vertex_id in final_walk]
        for vertex in walk_vertices:
            vertex[0].set_color(ALICE_COLORS[1])
        walk_edges = [ ss_graph.get_edge(final_walk[i], final_walk[i + 1]) for i in range(len(final_walk) - 1)]
        walk_mobs = VGroup(*walk_vertices, *walk_edges)
        non_walk_vertices = [ss_graph.get_vertex(vertex_id) for vertex_id in ss_graph.vertices if vertex_id not in final_walk]
        non_walk_edges = [ss_graph.get_edge(edge[0], edge[1]) for edge in ss_graph.edges if edge[0] not in final_walk or edge[1] not in final_walk]
        for vertex in non_walk_vertices:
            vertex.set_opacity(0.35)
        for edge in non_walk_edges:
            edge.set_opacity(0.35)
        non_walk_mobs = VGroup(*non_walk_vertices, *non_walk_edges)

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
            run_time = 0.3 if i < 10 else 0.075
            self.play(
                *[Create(edge) for edge in edges],
                run_time=run_time
            )
            self.play(
                DrawBorderThenFill(vertices, run_time=run_time),
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
                    0.8 * self.camera.frame_height / graph_mobject_in_scene.height
                )
            else:
                scaling_factor = (
                    0.8 * self.camera.frame_width / graph_mobject_in_scene.width
                )

            if scaling_factor < 1:
                # Scale and transform every mob individually
                self.play(
                    *[
                        mob.animate(run_time=run_time)
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

        self.remove(*graph_mobjects_introduced)
        self.add(non_walk_mobs)
        self.add(walk_mobs)
        self.wait()
        self.play(FadeOut(non_walk_mobs))
        self.wait(2)


class WellMixedGraph(Scene):
    """
    A Scene that visualizes multiple random walks on a supersingular isogeny graph to demonstrate the "well-mixed" property.

    This scene constructs a supersingular isogeny graph based on a given prime `p` and isogeny degree `ell`.
    It then performs multiple random walks on the graph, each with a different color, and highlights the vertices and edges traversed during the walks.
    The scene progressively introduces the graph level by level, scaling and centering the graph to fit the screen.
    Finally, it displays the random walks and fades out the rest of the graph, zooming in on the vertices to emphasize the mixing.

    SIKE:
    manim_configuration["disable_caching"] = True
    with tempconfig(manim_configuration):
        np.random.seed(0)
        scene = WellMixedGraph(p=2**216 * 3**137 - 1, ell=2, walk_length=101, num_walks=10)
        scene.render()
    """
    def __init__(self, p=2**216 * 3**137 - 1, ell=2, walk_length=15, num_walks=5):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.ell = ell
        self.p = p
        super().__init__()

    def construct(self):
        ell = self.ell
        p = self.p
        walk_length = self.walk_length
        num_walks = self.num_walks

        walk_background_gradient = color_gradient([ALICE_COLORS[0], BOB_COLORS[0]], num_walks)
        walk_select_gradient = color_gradient([ALICE_COLORS[1], BOB_COLORS[1]], num_walks)

        p_label = prime_overlay_label(p).scale(1.3)
        l_label = MathTex(
            rf"\ell = ", *[str(ell) + r" \, " for _ in range(num_walks)],
        ).scale(1.3)
        overlay = (
            VGroup(p_label, l_label, z_index=2)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UL, buff=0.5)
        ).add_background_rectangle(
            color=BLACK, opacity=0.6, buff=0.2, corner_radius=0.1
        )
        for i in range(num_walks):
            l_label[i + 1].set_color(walk_select_gradient[i])
        self.add(overlay)

        # Note to self: The flow of manim animation should always go somewhat like what is below:
        # 1. Perform any sage calculations neccesary to geenrate all mobjects that will be used in the scene
        # 2. Create all intermediate mobjects that will be used in the scene .animate should be avoided when possible. Always opt for Transform in these situations.
        # 3. Animate the mobjects in the scene
        print("Performing sage calculations...")
        Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
        levels, adjacency_list, final_walks = get_many_random_walks(
            p, walk_length, num_walks, ell
        )

        print("Creating graph mobjects...")
        ss_graph = SupersingularIsogenyGraph(Fq, levels, adjacency_list).scale(0.5, about_point=ORIGIN)
        all_walk_ids = []
        for final_walk in final_walks:
            all_walk_ids.extend(final_walk)
        all_walk_ids = list(dict.fromkeys(all_walk_ids))
        all_non_walk_ids = [vertex_id for vertex_id in ss_graph.vertices if vertex_id not in all_walk_ids]

        all_walk_vertices = [ss_graph.get_vertex(vertex_id) for vertex_id in all_walk_ids]
        all_walk_edges = [ss_graph.get_edge(final_walks[i][j], final_walks[i][j + 1]) for i in range(num_walks) for j in range(len(final_walks[i]) - 1)]
        all_non_walk_vertices = [ss_graph.get_vertex(vertex_id) for vertex_id in all_non_walk_ids]
        all_non_walk_edges = [ss_graph.get_edge(edge[0], edge[1]) for edge in ss_graph.edges if edge[0] not in all_walk_ids or edge[1] not in all_walk_ids]

        for vertex_id, vertex in zip(all_walk_ids, all_walk_vertices):
            paths_indices_including_vertex = [
                i for i, final_walk in enumerate(final_walks) if vertex_id in final_walk
            ]
            if len(paths_indices_including_vertex) == 1:
                vertex[0].set_color(walk_select_gradient[paths_indices_including_vertex[0]])
            else:
                # We use interpolate_color(color1, color2, alpha) to get a mix of two colors
                colors_containing_vertex = [
                    walk_select_gradient[i] for i in paths_indices_including_vertex
                ]
                while len(colors_containing_vertex) > 1:
                    new_colors = []
                    for i in range(0, len(colors_containing_vertex) - 1, 2):
                        color1 = colors_containing_vertex[i]
                        color2 = colors_containing_vertex[i + 1]
                        alpha = 0.5
                        new_colors.append(interpolate_color(color1, color2, alpha))
                    colors_containing_vertex = new_colors
                vertex[0].set_color(colors_containing_vertex[0])

        all_walk_mobs = VGroup(*all_walk_vertices, *all_walk_edges)
        all_non_walk_mobs = VGroup(*all_non_walk_vertices, *all_non_walk_edges)
        for mob in all_non_walk_mobs:
            mob.set_opacity(0.35)

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
            run_time = 0.4 if i < 10 else 0.05
            self.play(
                *[Create(edge) for edge in edges],
                run_time=run_time
            )
            self.play(
                DrawBorderThenFill(vertices, run_time=run_time),
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

            if scaling_factor < 1:
                # Scale and transform every mob individually
                self.play(
                    *[
                        mob.animate(run_time=run_time)
                        .shift(translation_vector)
                        .scale(scaling_factor, about_point=ORIGIN, scale_stroke=False)
                        for group in graph_mobjects_introduced
                        for mob in group
                    ],
                )
                # Scale unscene graph levels to match
                for j in range(i + 1, len(new_mobjects_on_level)):
                    unsceene_vertices, unscreen_edges = new_mobjects_on_level[j]
                    for mob in [*unsceene_vertices, *unscreen_edges]:
                        mob.shift(translation_vector).scale(
                            scaling_factor, about_point=ORIGIN, scale_stroke=False
                        )

        self.remove(*graph_mobjects_introduced)
        self.add(all_non_walk_mobs)
        self.add(all_walk_mobs)
        self.wait()
        self.play(FadeOut(all_non_walk_mobs))
        zoom_factor = 20
        self.play(all_walk_mobs.animate.scale(zoom_factor, about_point=ORIGIN), run_time=2)
        self.wait(1)
        self.play(*[mob.animate.scale(1 / zoom_factor, about_point=ORIGIN) for mob in all_walk_mobs], run_time=5)
        self.play(
            LaggedStart(
                *[
                    vertex.animate.scale(zoom_factor / 2).set_z_index(2) for vertex in all_walk_vertices
                ]
            ),
            run_time=10,
        )
        self.play(
            Succession(
                *[
                    ss_graph.get_vertex(walk[-1])
                    .animate.scale(zoom_factor / 2)
                    .set_z_index(3)
                    for walk in final_walks
                ]
            ),
            run_time=3,
        )
        self.wait()


class ShowcaseSIKE(Scene):
    def __init__(self, p=2**216 * 3**137 - 1):
        self.walk_length = 101
        self.num_walks = 2
        self.p = p
        super().__init__()

    def construct(self):
        p = self.p
        walk_length = self.walk_length
        num_walks = self.num_walks

        walk_background_gradient = [ALICE_COLORS[0], BOB_COLORS[0]]
        walk_select_gradient =  [ALICE_COLORS[1], BOB_COLORS[1]]

        p_label = prime_overlay_label(p).scale(1.3)
        l_label = MathTex(
            rf"\ell = ", r"2\,", r"3"
        ).scale(1.3)
        overlay = (
            VGroup(p_label, l_label, z_index=2)
            .arrange(DOWN, aligned_edge=LEFT)
            .to_edge(UL, buff=0.5)
        ).add_background_rectangle(
            color=BLACK, opacity=0.6, buff=0.2, corner_radius=0.1
        )
        for i in range(num_walks):
            l_label[i + 1].set_color(walk_select_gradient[i])
        self.add(overlay)

        # Note to self: The flow of manim animation should always go somewhat like what is below:
        # 1. Perform any sage calculations neccesary to geenrate all mobjects that will be used in the scene
        # 2. Create all intermediate mobjects that will be used in the scene .animate should be avoided when possible. Always opt for Transform in these situations.
        # 3. Animate the mobjects in the scene
        print("Performing sage calculations...")
        Fq, w, E0, initial_j_invariant = get_initial_sage_data(p)
        levels, adjacency_list, final_walks = get_many_random_walks(
            p, walk_length, num_walks
        )

        print("Creating graph mobjects...")
        ss_graph = SupersingularIsogenyGraph(Fq, levels, adjacency_list).scale(0.5, about_point=ORIGIN)
        all_walk_ids = []
        for final_walk in final_walks:
            all_walk_ids.extend(final_walk)
        all_walk_ids = list(dict.fromkeys(all_walk_ids))
        all_non_walk_ids = [vertex_id for vertex_id in ss_graph.vertices if vertex_id not in all_walk_ids]

        all_walk_vertices = [ss_graph.get_vertex(vertex_id) for vertex_id in all_walk_ids]
        all_walk_edges = [ss_graph.get_edge(final_walks[i][j], final_walks[i][j + 1]) for i in range(num_walks) for j in range(len(final_walks[i]) - 1)]
        all_non_walk_vertices = [ss_graph.get_vertex(vertex_id) for vertex_id in all_non_walk_ids]
        all_non_walk_edges = [ss_graph.get_edge(edge[0], edge[1]) for edge in ss_graph.edges if edge[0] not in all_walk_ids or edge[1] not in all_walk_ids]

        for vertex_id, vertex in zip(all_walk_ids, all_walk_vertices):
            paths_indices_including_vertex = [
                i for i, final_walk in enumerate(final_walks) if vertex_id in final_walk
            ]
            if len(paths_indices_including_vertex) == 1:
                vertex[0].set_color(walk_select_gradient[paths_indices_including_vertex[0]])
            else:
                # We use interpolate_color(color1, color2, alpha) to get a mix of two colors
                colors_containing_vertex = [
                    walk_select_gradient[i] for i in paths_indices_including_vertex
                ]
                while len(colors_containing_vertex) > 1:
                    new_colors = []
                    for i in range(0, len(colors_containing_vertex) - 1, 2):
                        color1 = colors_containing_vertex[i]
                        color2 = colors_containing_vertex[i + 1]
                        alpha = 0.5
                        new_colors.append(interpolate_color(color1, color2, alpha))
                    colors_containing_vertex = new_colors
                vertex[0].set_color(colors_containing_vertex[0])

        all_walk_mobs = VGroup(*all_walk_vertices, *all_walk_edges)
        all_non_walk_mobs = VGroup(*all_non_walk_vertices, *all_non_walk_edges)
        for mob in all_non_walk_mobs:
            mob.set_opacity(0.35)

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
            # Linearly interpolate run_time from 0.5 (i=0) to 0.05 (i=10), clamp for i > 10
            if i <= 10:
                run_time = 0.5 - (0.45 * i / 10)
            else:
                run_time = 0.05
            self.play(
                *[Create(edge) for edge in edges],
                run_time=run_time
            )
            self.play(
                DrawBorderThenFill(vertices, run_time=run_time),
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

            if scaling_factor < 1:
                # Scale and transform every mob individually
                self.play(
                    *[
                        mob.animate(run_time=run_time)
                        .shift(translation_vector)
                        .scale(scaling_factor, about_point=ORIGIN, scale_stroke=False)
                        for group in graph_mobjects_introduced
                        for mob in group
                    ],
                )
                # Scale unscene graph levels to match
                for j in range(i + 1, len(new_mobjects_on_level)):
                    unsceene_vertices, unscreen_edges = new_mobjects_on_level[j]
                    for mob in [*unsceene_vertices, *unscreen_edges]:
                        mob.shift(translation_vector).scale(
                            scaling_factor, about_point=ORIGIN, scale_stroke=False
                        )

        self.remove(*graph_mobjects_introduced)
        self.add(all_non_walk_mobs)
        self.add(all_walk_mobs)
        self.wait()
        # self.play(FadeOut(all_non_walk_mobs))
        zoom_factor = 10
        origin_center = graph_mobjects_introduced[0].get_center().copy()
        self.play(VGroup(all_walk_mobs, all_non_walk_mobs).animate.scale(zoom_factor, about_point=origin_center), run_time=2)
        self.wait(1)
        self.play(*[mob.animate.scale(1 / zoom_factor, about_point=origin_center) for mob in all_walk_mobs + all_non_walk_mobs], run_time=5)
        self.play(
            LaggedStart(
                *[
                    vertex.animate.scale(zoom_factor / 2).set_z_index(2) for vertex in all_walk_vertices
                ]
            ),
            run_time=10,
        )
        self.play(
            Succession(
                *[
                    ss_graph.get_vertex(walk[-1])
                    .animate.scale(zoom_factor / 4)
                    .set_z_index(3)
                    for walk in final_walks
                ]
            ),
            run_time=3,
        )
        self.wait()



if __name__ == '__main__':
    manim_configuration = {
        "quality": "production_quality",
        "preview": False,
        "output_file": "PreviewVideo",
        "disable_caching": False,
        "max_files_cached": 1000,
        "write_to_movie": True,
        "show_file_in_browser": False,
    }
    np.random.seed(11)
    manim_configuration["output_file"] = "ShowcaseSIKE"
    manim_configuration["disable_caching"] = False
    with tempconfig(manim_configuration):
        scene = ShowcaseSIKE(p=2**216 * 3**137 - 1)
        scene.render()


    manim_configuration["output_file"] = "WalkableGraph"
    manim_configuration["disable_caching"] = True
    with tempconfig(manim_configuration):
        scene = WalkableGraph(p=2**216 * 3**137 - 1, ell=3, walk_length=100)
        scene.render()

    manim_configuration["output_file"] = "WellMixedGraph"
    manim_configuration["disable_caching"] = True
    with tempconfig(manim_configuration):
        scene = WellMixedGraph(p=2**216 * 3**137 - 1, ell=2, walk_length=101, num_walks=10)
        scene.render()

