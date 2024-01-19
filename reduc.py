import numpy as np

def load_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex = list(map(float, line[2:].strip().split()))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = line[2:].strip().split()
                face_indices = [int(idx.split('/')[0]) - 1 for idx in face]
                faces.append(face_indices)

    return np.array(vertices), faces

def save_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f'v {" ".join(map(str, vertex))}\n')

        for face in faces:
            file.write(f'f {" ".join(map(lambda x: str(x + 1), face))}\n')

def simplify_mesh(vertices, faces, target_triangles):
    while len(faces) > target_triangles:
        edge_costs = {}
        for face in faces:
            for i in range(len(face)):
                edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                if edge in edge_costs:
                    edge_costs[edge] += 1
                else:
                    edge_costs[edge] = 1

        min_edge = min(edge_costs, key=edge_costs.get)

        new_faces = []
        for face in faces:
            if not any(v in min_edge for v in face):
                new_faces.append(face)

        faces = new_faces

    return vertices, faces

if __name__ == "__main__":
    input_file = "amforeas_hole.obj"
    output_file = "output33.obj"
    target_triangles = 33000  

    vertices, faces = load_obj(input_file)
    vertices, faces = simplify_mesh(vertices, faces, target_triangles)
    save_obj(output_file, vertices, faces)
