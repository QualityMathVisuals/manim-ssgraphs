from manim import *
from sage.all import *


class GraphEdgeMobject(VGroup):
    def __init__(
        self,
        vertex1,
        vertex2,
        edge_color: ManimColor = WHITE,
        edge_thickness: float = DEFAULT_STROKE_WIDTH,
    ):
        super().__init__()  # Initialize the VGroup
        vertex_center1 = vertex1.get_center()
        vertex_center2 = vertex2.get_center()
        direction_vector = vertex_center2 - vertex_center1
        angle = np.arctan2(direction_vector[1], direction_vector[0])

        edge = Line(
            start=vertex1[0].point_at_angle(angle),
            end=vertex2[0].point_at_angle(angle + PI),
            stroke_width=edge_thickness,
            color=edge_color,
            z_index=-1,
        )
        self.add(edge)


class GraphVertexMobject(VGroup):
    def __init__(
        self,
        vertex_label,
        background_color: ManimColor = "#3A7DB4",
        background_opacity: float = 0.85,
        radius: float = 0.5,
        border_color: ManimColor = WHITE,
        border_thickness: float = 2,
        label_font_size: float = 50,
    ):
        super().__init__()  # Initialize the VGroup
        self.background_opacity = background_opacity

        background_circle = Circle(
            radius=radius,
            color=background_color,
            fill_opacity=background_opacity,
            stroke_opacity=0,
            z_index=0,
        )
        border_circle = Circle(
            radius=radius,
            color=border_color,
            stroke_width=border_thickness,
            z_index=1,
        )
        label = MathTex(
            vertex_label,
            font_size=label_font_size,
            z_index=1,
        )

        if label.width > background_circle.width * 0.85:
            label.scale_to_fit_width(background_circle.width * 0.85)

        self.add(background_circle, border_circle, label)

    def scale(self, scale_factor: float, scale_stroke=True, **kwargs):
        # Current patching of bug where VMobject.scale(scale_stroke=True) creates bad visual with nested SVG mobjects.
        background_circle, border_circle, label = self
        background_circle.scale(scale_factor, scale_stroke, **kwargs)
        border_circle.scale(scale_factor, scale_stroke, **kwargs)
        label.scale(scale_factor, scale_stroke=False, **kwargs)

        return self

    def set_opacity(self, opacity):
        # Set the opacity of the background circle and border circle
        background_circle, border_circle, label = self
        background_circle.set_fill(opacity=opacity * self.background_opacity)
        border_circle.set_stroke(opacity=opacity)

        # Set the opacity of the label
        label.set_opacity(opacity)

        return self


def get_random_separated_angles(
    n: int,
    min_separation: float = None,
) -> np.ndarray:
    angles = []
    attempts = 0
    max_attempts = 100 * n
    min_separation = min_separation or PI / (6 * n)

    while len(angles) < n and attempts < max_attempts:
        new_angle = np.random.normal(0, 0.6)
        is_valid = True
        for existing_angle in angles:
            if abs(new_angle - existing_angle) < min_separation:
                is_valid = False
                break

        if is_valid:
            angles.append(new_angle)

        attempts += 1

    if len(angles) < n:
        base_separation = 1.3 / n
        angles = [(i - n / 2) * base_separation for i in range(n)]

    return np.array(angles)


def get_random_seperated_vectors(
    direction_vec: np.ndarray, num_vectors: int, unit_length=1
):
    if num_vectors == 0:
        return []

    norm = np.linalg.norm(direction_vec)
    direction_vec = direction_vec / norm
    angles = get_random_separated_angles(num_vectors)

    unit_vectors = [rotate_vector(direction_vec, angle) for angle in angles]
    random_multiples_of_diameter = np.random.normal(3, 0.8, num_vectors)
    final_vectors = [
        unit_length * mult * vec
        for mult, vec in zip(random_multiples_of_diameter, unit_vectors)
    ]

    return final_vectors


class SupersingularIsogenyGraph(VGroup):
    def __init__(self, Fq, levels, adjacency_list, unit_length=None):
        super().__init__()
        self.Fq = Fq  # Finite field
        self.levels = levels  # Save the levels
        self.vertices = {}  # Mapping from vertex ID to vertex Mobject
        self.edges = {}  # Mapping from edge ID to edge Mobject

        # Mapping from vertex ID to list of connected vertex IDs
        max_degree = 0
        self.adjacency_dict = {}
        for v1, v2 in adjacency_list:
            if v1 not in self.adjacency_dict:
                self.adjacency_dict[v1] = []
            if v2 not in self.adjacency_dict:
                self.adjacency_dict[v2] = []
            self.adjacency_dict[v1].append(v2)
            self.adjacency_dict[v2].append(v1)
            if len(self.adjacency_dict[v1]) > max_degree:
                max_degree = len(self.adjacency_dict[v1])

        # A list of vertex ids that have labels assigned.
        self._labeled_vertices = {}
        # Subscript index for j_1, j_2, etc. (j invariants that are too long)
        self._running_j_index = 1

        # This fills out the self.vertices with mobjects.
        for level in levels:
            for vertex_id in level:
                vertex_label = self._vertex_id_label(vertex_id)
                vertex_mobject = GraphVertexMobject(vertex_label)
                self.vertices[vertex_id] = vertex_mobject

        # This places the vertices in the correct locations.
        if not unit_length:
            unit_length = 1 + (max_degree / 8)
        self.place_vertices_by_levels(levels, unit_length=unit_length)

        # This fills out the self.edges with mobjects.
        for level in levels:
            for vertex_id in level:
                # Get the neighbors of the vertex
                neighbor_ids = self.adjacency_dict[vertex_id]
                for neighbor_id in neighbor_ids:
                    # Create the unique edge ID
                    edge_id = tuple(sorted((vertex_id, neighbor_id)))
                    if edge_id not in self.edges:
                        # Create the edge mobject
                        edge_mobject = GraphEdgeMobject(
                            self.vertices[vertex_id], self.vertices[neighbor_id]
                        )
                        self.edges[edge_id] = edge_mobject

        self.add(*self.vertices.values(), *self.edges.values())

    def place_vertices_by_levels(self, levels, unit_length=1):

        # Manually place the first 3 levels:
        # Center the first vertex. Already done.
        if len(levels[0]) > 1:
            print(f"Level 0: {levels[0]}")
            print("why are there more than 1 vertices in level 0?")

        placed_vertex_ids = levels[0].copy()

        for i in range(1, len(levels)):
            vertices_in_previous_level = levels[i - 1]

            for vertex_id in vertices_in_previous_level:
                # Get the neighbors that are not already placed
                neighbor_ids = [
                    neighbor_id
                    for neighbor_id in self.adjacency_dict[vertex_id]
                    if neighbor_id not in placed_vertex_ids
                ]
                neighboring_mobs = [
                    self.vertices[neighbor_id] for neighbor_id in neighbor_ids
                ]
                num_neighbors = len(neighboring_mobs)

                if i < 3:
                    # Create a fixed length, determined vector (looks nicer at start when graph is less random/dense)
                    if i == 1:
                        direction_vec = DOWN
                    elif i == 2:
                        direction_vec = RIGHT
                    norm = np.linalg.norm(direction_vec)
                    direction_vec = direction_vec / norm
                    fixed_length = unit_length * 2
                    angles = np.linspace(0, TAU, num_neighbors, endpoint=False)
                    unit_vectors = [
                        rotate_vector(direction_vec, angle) for angle in angles
                    ]
                    final_vectors = [fixed_length * vec for vec in unit_vectors]
                else:
                    # Get the direction vector between the current vertex and the previous vertex
                    current_center = self.vertices[vertex_id].get_center()
                    # Find a vertex that is a neighbor of vertex_id and is in the (i-2)-th level
                    possible_two_step_back_ids = [
                        v for v in self.adjacency_dict[vertex_id] if v in levels[i - 2]
                    ]
                    if len(possible_two_step_back_ids) == 0:
                        previous_center = ORIGIN
                    else:
                        two_step_back_id = possible_two_step_back_ids[0]
                        previous_center = self.vertices[two_step_back_id].get_center()

                    direction_vec = current_center - previous_center

                    # get a random fan of vectors
                    final_vectors = get_random_seperated_vectors(
                        direction_vec, num_neighbors, unit_length=unit_length
                    )

                # Move the neighboring vertices to their new positions
                for neighbor_id, neighbor_mob, delta_vector in zip(
                    neighbor_ids, neighboring_mobs, final_vectors
                ):
                    neighbor_mob.move_to(
                        self.vertices[vertex_id].get_center() + delta_vector
                    )
                    placed_vertex_ids.append(neighbor_id)

    # Returns the submobject of the graph (vertices and edges) that are up to the given level (inclusive).
    def get_up_to_level(self, level: int) -> VGroup:
        vertex_ids_up_to_level = [
            vertex_id for i in range(level + 1) for vertex_id in self.levels[i]
        ]
        vertex_mobs_up_to_level = [
            self.vertices[vertex_id] for vertex_id in vertex_ids_up_to_level
        ]
        edge_ids_up_to_level = [
            tuple(sorted((vertex_id, neighbor_id)))
            for vertex_id in vertex_ids_up_to_level
            for neighbor_id in self.adjacency_dict[vertex_id]
            if neighbor_id in vertex_ids_up_to_level
        ]
        edge_ids_up_to_level = list(set(edge_ids_up_to_level))
        edge_mobs_up_to_level = [
            self.edges[edge_id] for edge_id in edge_ids_up_to_level
        ]

        return VGroup(VGroup(*vertex_mobs_up_to_level), VGroup(*edge_mobs_up_to_level))

    # Returns the submobject of the graph (vertices and edges) that are at the given level.
    def get_including_level(
        self, level: int, include_further_levels: bool = False
    ) -> VGroup:
        # Get the vertex IDs at the given level
        vertex_ids_at_level = self.levels[level]
        vertex_mobs_at_level = [
            self.vertices[vertex_id] for vertex_id in vertex_ids_at_level
        ]
        # Determine the vertex IDs up to the desired level
        neighbor_search_space = [
            vertex_id for i in range(level + 1) for vertex_id in self.levels[i]
        ]

        if include_further_levels:
            # Include all levels after the current level
            neighbor_search_space.extend(
                [
                    vertex_id
                    for i in range(level + 1, len(self.levels))
                    for vertex_id in self.levels[i]
                ]
            )
        edge_ids_including_level = [
            tuple(sorted((vertex_id, neighbor_id)))
            for vertex_id in vertex_ids_at_level
            for neighbor_id in self.adjacency_dict[vertex_id]
            if neighbor_id in neighbor_search_space
        ]
        edge_ids_including_level = list(set(edge_ids_including_level))
        edge_mobs_at_level = [
            self.edges[edge_id] for edge_id in edge_ids_including_level
        ]

        return VGroup(VGroup(*vertex_mobs_at_level), VGroup(*edge_mobs_at_level))

    def _vertex_id_label(self, vertex_id: int) -> str:
        if vertex_id in self._labeled_vertices:
            return self._labeled_vertices[vertex_id]

        vertex_label = latex(self.Fq.from_integer(vertex_id))
        if len(vertex_label) > 8:
            vertex_label = r"j_{" + str(self._running_j_index) + "}"
            self._running_j_index += 1

        self._labeled_vertices[vertex_id] = vertex_label
        return vertex_label
    
    def get_vertex_id_by_label_str(self, vertex_label):
        for vertex_id, label in self._labeled_vertices.items():
            if label == vertex_label:
                return vertex_id
        return None

    def get_vertex(self, vertex_id):
        return self.vertices[vertex_id]

    def get_edge(self, vertex_id1, vertex_id2):
        edge_id = tuple(sorted((vertex_id1, vertex_id2)))
        return self.edges[edge_id]

    def get_neighbor_ids(self, vertex_id, exclude_previous_level=False):
        if vertex_id not in self.adjacency_dict:
            raise ValueError(f"Vertex ID {vertex_id} not found in the graph.")

        prev_level_of_vertex_id = []
        for i, level in enumerate(self.levels):
            if vertex_id in level and i > 0:
                prev_level_of_vertex_id = self.levels[i - 1]
                break

        if exclude_previous_level:
            return [
                neighbor_id
                for neighbor_id in self.adjacency_dict[vertex_id]
                if neighbor_id not in prev_level_of_vertex_id
            ]
        else:
            return self.adjacency_dict[vertex_id]

    def get_neighboring_edges(self, vertex_id, exclude_previous_level=False):
        neighbor_ids = self.get_neighbor_ids(
            vertex_id, exclude_previous_level=exclude_previous_level
        )
        return [
            self.edges[tuple(sorted((vertex_id, neighbor_id)))]
            for neighbor_id in neighbor_ids
        ]

    def get_neighboring_vertices(self, vertex_id, exclude_previous_level=False):
        neighbor_ids = self.get_neighbor_ids(
            vertex_id, exclude_previous_level=exclude_previous_level
        )
        return [self.vertices[neighbor_id] for neighbor_id in neighbor_ids]

    def get_vertices(self):
        return self.vertices.values()

    def get_edges(self):
        return self.edges.values()

    def scale(self, scale_factor, scale_stroke=True, **kwargs):
        about_point = kwargs.pop("about_point", ORIGIN)

        for vertex in self.vertices.values():
            vertex.scale(scale_factor, scale_stroke, about_point=about_point, **kwargs)
        for edge in self.edges.values():
            edge.scale(scale_factor, scale_stroke, about_point=about_point, **kwargs)
        return self


class IsogenyDiamond(SupersingularIsogenyGraph):
    def __init__(
        self,
        Fq,
        vertices_E0_to_E1,
        vertices_E0_to_E2,
        vertices_E2_to_E3,
        vertices_E1_to_E3,
        unit_length=None,
    ):
        adjacency_list = []
        levels = []
        # Create the levels and adjacency list from the paths
        verts_in_level = []

        self.diamond_sides = [
            vertices_E0_to_E1,
            vertices_E0_to_E2,
            vertices_E2_to_E3,
            vertices_E1_to_E3,
        ]

        max_path_length = max(len(vertices_E0_to_E1), len(vertices_E0_to_E2))
        for i in range(max_path_length):
            vertices_to_add = [
                path[i]
                for path in [vertices_E0_to_E1, vertices_E0_to_E2]
                if i < len(path)
            ]
            vertices_to_add = list(dict.fromkeys(vertices_to_add))
            level = [
                vertex_id
                for vertex_id in vertices_to_add
                if vertex_id not in verts_in_level
            ]
            verts_in_level.extend(level)
            if len(level) > 0:
                levels.append(level)

        max_path_length = max(len(vertices_E2_to_E3), len(vertices_E1_to_E3))
        for i in range(max_path_length):
            vertices_to_add = [
                path[i]
                for path in [vertices_E2_to_E3, vertices_E1_to_E3]
                if i < len(path)
            ]
            vertices_to_add = list(dict.fromkeys(vertices_to_add))
            level = [
                vertex_id
                for vertex_id in vertices_to_add
                if vertex_id not in verts_in_level
            ]
            verts_in_level.extend(level)
            if len(level) > 0:
                levels.append(level)

        for path in self.diamond_sides:
            for i in range(len(path) - 1):
                adjacency_list.append((path[i], path[i + 1]))


        super().__init__(Fq, levels, adjacency_list, unit_length=unit_length)

    def get_unit_length(self):
        return self.unit_length

    def set_unit_length(self, unit_length):
        self.unit_length = unit_length
        self.place_vertices_by_levels(self.levels, unit_length=unit_length)

    def place_vertices_by_levels(self, levels, unit_length=1):
        self.place_first_two_sides(levels, unit_length=unit_length)
        self.place_last_two_sides(levels, unit_length=unit_length)

    def place_first_two_sides(self, levels, unit_length):
        vertices_E0_to_E1, vertices_E0_to_E2, _, _ = self.diamond_sides

        # Much like the original place_vertices_by_levels, however, we start with the placement of the sides of the diamond and E3.
        # Manually place the first 3 levels:
        # Center the first vertex. Already done.
        if len(levels[0]) > 1:
            print(f"Level 0: {levels[0]}")
            print("why are there more than 1 vertices in level 0?")

        placed_vertex_ids = levels[0].copy()

        # for i in range(1, len(levels)):
        for i in range(1, max(len(vertices_E0_to_E1), len(vertices_E0_to_E2))):
            # vertices_in_previous_level = levels[i - 1]
            side_1_prev_vertex = (
                vertices_E0_to_E1[i - 1] if i < len(vertices_E0_to_E1) else None
            )
            side_2_prev_vertex = (
                vertices_E0_to_E2[i - 1] if i < len(vertices_E0_to_E2) else None
            )
            side_1_curr_vertex = (
                vertices_E0_to_E1[i] if i < len(vertices_E0_to_E1) else None
            )
            side_2_curr_vertex = (
                vertices_E0_to_E2[i] if i < len(vertices_E0_to_E2) else None
            )
            ells = []
            vertices_to_extend = []
            vertices_to_place = []
            if (
                side_1_prev_vertex is not None
                and side_1_curr_vertex is not None
                and side_1_curr_vertex not in placed_vertex_ids
            ):
                ells.append(2)
                vertices_to_extend.append(side_1_prev_vertex)
                vertices_to_place.append(side_1_curr_vertex)

            if (
                side_2_prev_vertex is not None
                and side_2_curr_vertex is not None
                and side_2_curr_vertex not in placed_vertex_ids
                and side_2_curr_vertex != side_1_curr_vertex
            ):
                ells.append(3)
                vertices_to_extend.append(side_2_prev_vertex)
                vertices_to_place.append(side_2_curr_vertex)

            for vertex_id, ell, neighbor_id in zip(
                vertices_to_extend, ells, vertices_to_place
            ):
                if i < 3:
                    # Create a fixed length, determined vector (looks nicer at start when graph is less random/dense)
                    if i == 1:
                        direction_vec = DOWN
                    elif i == 2:
                        direction_vec = RIGHT
                    norm = np.linalg.norm(direction_vec)
                    direction_vec = direction_vec / norm
                    fixed_length = unit_length * 2
                    angles = np.linspace(0, TAU, ell, endpoint=False)
                    unit_vectors = [
                        rotate_vector(direction_vec, angle) for angle in angles
                    ]
                    final_vectors = [fixed_length * vec for vec in unit_vectors]
                else:
                    # Get the direction vector between the current vertex and the previous vertex
                    current_center = self.vertices[vertex_id].get_center()
                    # Find a vertex that is a neighbor of vertex_id and is in the (i-2)-th level
                    possible_two_step_back_ids = [
                        v for v in self.adjacency_dict[vertex_id] if v in levels[i - 2]
                    ]
                    if len(possible_two_step_back_ids) == 0:
                        previous_center = ORIGIN
                    else:
                        two_step_back_id = possible_two_step_back_ids[0]
                        previous_center = self.vertices[two_step_back_id].get_center()

                    direction_vec = current_center - previous_center

                    # get a random fan of vectors
                    final_vectors = get_random_seperated_vectors(
                        direction_vec, ell, unit_length=unit_length
                    )

                # Move the neighboring vertices to their new positions
                neighbor_mob = self.vertices[neighbor_id]
                delta_vector = final_vectors[np.random.randint(len(final_vectors))]
                neighbor_mob.move_to(
                    self.vertices[vertex_id].get_center() + delta_vector
                )
                placed_vertex_ids.append(neighbor_id)

    def place_last_two_sides(self, levels, unit_length):
        vertices_E0_to_E1, vertices_E0_to_E2, vertices_E2_to_E3, vertices_E1_to_E3 = (
            self.diamond_sides
        )

        # Placing E3...
        initial_vertex_id = levels[0][0]
        E1_id = vertices_E1_to_E3[0]
        E2_id = vertices_E2_to_E3[0]
        origin_to_E1 = (
            self.vertices[E1_id].get_center()
            - self.vertices[initial_vertex_id].get_center()
        )
        origin_to_E2 = (
            self.vertices[E2_id].get_center()
            - self.vertices[initial_vertex_id].get_center()
        )

        E3_center = origin_to_E1 + origin_to_E2

        placed_vertex_ids = list(dict.fromkeys(vertices_E0_to_E1 + vertices_E0_to_E2))

        # for i in range(1, len(levels)):
        for i in range(1, max(len(vertices_E2_to_E3), len(vertices_E1_to_E3))):
            # vertices_in_previous_level = levels[i - 1]
            side_3_prev_vertex = (
                vertices_E2_to_E3[i - 1] if i < len(vertices_E2_to_E3) else None
            )
            side_4_prev_vertex = (
                vertices_E1_to_E3[i - 1] if i < len(vertices_E1_to_E3) else None
            )
            side_3_curr_vertex = (
                vertices_E2_to_E3[i] if i < len(vertices_E2_to_E3) else None
            )
            side_4_curr_vertex = (
                vertices_E1_to_E3[i] if i < len(vertices_E1_to_E3) else None
            )
            ells = []
            vertices_to_extend = []
            vertices_to_place = []
            if (
                side_3_prev_vertex is not None
                and side_3_curr_vertex is not None
                and side_3_curr_vertex not in placed_vertex_ids
            ):
                ells.append(2)
                vertices_to_extend.append(side_3_prev_vertex)
                vertices_to_place.append(side_3_curr_vertex)

            if (
                side_4_prev_vertex is not None
                and side_4_curr_vertex is not None
                and side_4_curr_vertex not in placed_vertex_ids
                and side_4_curr_vertex != side_3_curr_vertex
            ):
                ells.append(3)
                vertices_to_extend.append(side_4_prev_vertex)
                vertices_to_place.append(side_4_curr_vertex)

            for vertex_id, ell, neighbor_id in zip(
                vertices_to_extend, ells, vertices_to_place
            ):
                # Get the direction vector between the current vertex and the previous vertex
                current_center = self.vertices[vertex_id].get_center()
                # the difference. we want the direction_vector to be pointing towards E3.
                direction_vec = E3_center - current_center

                # get a random fan of vectors
                final_vectors = get_random_seperated_vectors(
                    direction_vec, ell, unit_length=unit_length
                )

                # Move the neighboring vertices to their new positions
                neighbor_mob = self.vertices[neighbor_id]
                delta_vector = final_vectors[np.random.randint(len(final_vectors))]
                neighbor_mob.move_to(
                    self.vertices[vertex_id].get_center() + delta_vector
                )
                placed_vertex_ids.append(neighbor_id)



class EmbeddedSupersingularIsogenyGraph(SupersingularIsogenyGraph):
    def __init__(
        self,
        Fq,
        levels,
        adjacency_list,
        embedding_func,
        vertex_scaling=1,
        coordinate_plane_size=7,
        vertical_y_offset=0,
    ):
        self.embedding_func = embedding_func
        self.coordinate_plane_size = coordinate_plane_size
        self.vertex_y_offset = vertical_y_offset
        self.vertex_scaling = vertex_scaling
        super().__init__(Fq, levels, adjacency_list)

        VGroup(*self.edges.values()).set_stroke(width=list(self.edges.values())[0].get_stroke_width())

    def place_vertices_by_levels(self, levels, unit_length=1):
        Fq = self.Fq
        for vertex_id in self.vertices.keys():
            self.vertices[vertex_id].scale(self.vertex_scaling)
            j_inv = Fq.from_integer(vertex_id)

            # Apply the embedding function to get normalized coordinates
            norm_x, norm_y = self.embedding_func(j_inv)

            # Scale and center to coordinate plane
            x = (norm_x - 0.5) * self.coordinate_plane_size
            y = (norm_y - 0.5) * self.coordinate_plane_size

            self.vertices[vertex_id].move_to(np.array([x, y, 0]))

