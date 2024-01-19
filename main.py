from flask import Flask, render_template, request
import trimesh
import numpy as np
import os
import pickle

app = Flask(__name__)


def compute_moment_of_inertia(mesh_vertices, mesh_faces):
    # Compute the center of mass
    center_of_mass = np.mean(mesh_vertices, axis=0)
    # Initialize the moment of inertia matrix
    moment_of_inertia = np.zeros((3, 3))
    # Iterate over each face in the mesh
    for face in mesh_faces:
        # Get the vertices of the face
        v1, v2, v3 = mesh_vertices[face]
        # Compute the normal vector of the face
        normal = np.cross(v2 - v1, v3 - v1)
        # Compute the area of the face
        area = np.linalg.norm(normal) / 2
        # Compute the centroid of the face
        centroid = (v1 + v2 + v3) / 3
        # Compute the moment of inertia of the face and add it to the overall moment of inertia matrix
        moment_of_inertia += area * (np.dot(centroid - center_of_mass, centroid - center_of_mass) * np.eye(3) - np.outer(centroid - center_of_mass, centroid - center_of_mass))
    return moment_of_inertia


def compute_average_distance_from_axis(mesh_vertices, axis_vector):
    # Compute the distance of each vertex from the axis
    distances = np.abs(np.dot(mesh_vertices, axis_vector))
    # Compute the average distance
    average_distance = np.mean(distances)
    return average_distance


def compute_variance_of_distance_from_axis(mesh_vertices, axis_vector):
    # Compute the distance of each vertex from the axis
    distances = np.abs(np.dot(mesh_vertices, axis_vector))
    # Compute the variance of the distances
    variance_distance = np.var(distances)
    return variance_distance


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        user_query_file = request.files['file']
        if user_query_file:
            # Save the user's uploaded file
            user_query_file.save(os.path.join('uploads', user_query_file.filename))

            # Load descriptors from the file
            descriptors_file_path = 'descriptors_database.pkl'
            axis_vector = np.array([1, 0, 0])

            if os.path.exists(descriptors_file_path):
                with open(descriptors_file_path, 'rb') as file:
                    descriptors_database = pickle.load(file)
            else:
                return "Descriptors file not found. Please calculate descriptors first."

            obj_folder_path = '3D Models/All Models'
            obj_files = [file for file in os.listdir(obj_folder_path) if file.endswith('.obj')]

            user_mesh = trimesh.load(os.path.join('uploads', user_query_file.filename))
            user_vertices = user_mesh.vertices
            user_faces = user_mesh.faces
            user_moment_of_inertia = compute_moment_of_inertia(user_vertices, user_faces)
            user_average_distance = compute_average_distance_from_axis(user_vertices, axis_vector)
            user_variance_distance = compute_variance_of_distance_from_axis(user_vertices, axis_vector)

            # Calculate similarities
            similarities = []
            for descriptor in descriptors_database:
                moment_of_inertia_sim = np.abs(descriptor[0] - user_moment_of_inertia).sum()
                average_distance_sim = np.abs(descriptor[1] - user_average_distance)
                variance_distance_sim = np.abs(descriptor[2] - user_variance_distance)
                similarity = moment_of_inertia_sim + average_distance_sim + variance_distance_sim
                similarities.append(similarity)

            # Sort the results by similarity
            results = [x for _, x in sorted(zip(similarities, obj_files), key=lambda pair: pair[0])]

            return render_template('results.html', results=results[:10])

    return "Invalid request"

if __name__ == '__main__':
    app.run(debug=True)
