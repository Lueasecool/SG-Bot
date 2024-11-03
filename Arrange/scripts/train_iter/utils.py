import yaml
from yaml import Loader
import cv2
import numpy as np
import os
import torch
import trimesh
import seaborn as sns

import pyrender
def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def render_box(scene_id, cats, predBoxes, predAngles, datasize='small', classes=None, render_type='txt2shape',
               render_shapes=True, store_img=False, render_boxes=False, demo=False, visual=False, without_lamp=False,
               str_append="", mani=0, missing_nodes=None, manipulated_nodes=None, objs_before=None, store_path=None):
    os.makedirs(store_path,exist_ok=True)
    if render_type not in ['txt2shape', 'retrieval', 'onlybox']:
        raise ValueError('Render type needs to be either set to txt2shape or retrieval or onlybox.')
    color_palette = np.array(sns.color_palette('hls', len(classes)))
    box_and_angle = torch.cat([predBoxes.float(), predAngles.float()], dim=-1)

    obj_n = len(box_and_angle)
    if mani == 2:
        if len(missing_nodes) > 0:
            box_and_angle = box_and_angle[missing_nodes]
        elif len(manipulated_nodes) > 0:
            box_and_angle = box_and_angle[sorted(manipulated_nodes)]

    mesh_dir = os.path.join(store_path, render_type, 'object_meshes', scene_id[0])
    os.makedirs(mesh_dir, exist_ok=True)
    
    if render_type == 'onlybox':
        lamp_mesh_list, trimesh_meshes = get_bbox(box_and_angle, cats, classes, colors=color_palette[cats], without_lamp=without_lamp)
    else:
        raise NotImplementedError

    if mani == 2:
        print("manipulated nodes: ", len(manipulated_nodes), len(trimesh_meshes))
        if len(missing_nodes) > 0:
            trimesh_meshes += objs_before
            query_label = classes[cats[0]].strip('\n')
            str_append += "_" + query_label
        elif len(manipulated_nodes) > 0:
            i, j, k = 0, 0, 0
            for i in range(obj_n):
                query_label = classes[cats[i]].strip('\n')
                i += 1
                if query_label == '_scene_' or query_label == 'floor' or (query_label == 'lamp' and without_lamp):
                    continue
                if i in manipulated_nodes:
                    objs_before[j] = trimesh_meshes[k]
                    str_append += "_" + query_label
                    #all_meshes.append(trimesh_meshes[j])
                    k += 1
                j += 1
            trimesh_meshes = objs_before

    if demo:
        mesh_dir_shifted = mesh_dir.replace('object_meshes', 'object_meshes_shifted')
        os.makedirs(mesh_dir_shifted, exist_ok=True)
        trimesh_meshes += lamp_mesh_list
        for i, mesh in enumerate(trimesh_meshes):
            mesh.export(os.path.join(mesh_dir_shifted,  f"{i}.obj"))
    scene = trimesh.Scene(trimesh_meshes)
    scene_path = os.path.join(store_path, render_type)
    if len(str_append) > 0:
        render_type += str_append
    os.makedirs(scene_path, exist_ok=True)
    scene.export(os.path.join(scene_path, "{0}_{1}.glb".format(scene_id[0], render_type)))

    if visual:
        scene.show()

    if store_img and not demo:
        img_path = os.path.join(store_path, render_type, "render_imgs")
        os.makedirs(img_path, exist_ok=True)
        color_img = render_img(trimesh_meshes)
        color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
        file_name = scene_id[0]
        if len(str_append) > 0:
            file_name += str_append
        cv2.imwrite(os.path.join(img_path, f'{file_name}.png'), color_bgr)

    if mani==1:
        return trimesh_meshes




def get_bbox(boxes, cat_ids, classes, colors, without_lamp=False):
    trimesh_meshes = []
    colors = iter(colors)
    lamp_mesh_list=[]
    for j in range(0, boxes.shape[0]):
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        box_points = params_to_8points_3dfront(boxes[j], degrees=True)
        trimesh_meshes.append(create_bbox_marker(box_points, tube_radius=0.02, color=next(colors)))
        # if query_label == 'nightstand':
        #     trimesh_meshes.pop()
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(trimesh_meshes.pop())


    return lamp_mesh_list, trimesh_meshes

def create_bbox_marker(corner_points, color=[0, 0, 255], tube_radius=0.002, sections=4):
    """Create a 3D mesh visualizing a bbox. It consists of 12 cylinders.

    Args:
        corner_points
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 4.

    Returns:
        trimesh.Trimesh: A mesh.
    """
    edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    bbox_edge_list = []
    for edge in edges:
        bbox_edge = trimesh.creation.cylinder(radius=tube_radius,sections=sections,segment=[corner_points[edge[0]],corner_points[edge[1]]])
        bbox_edge_list.append(bbox_edge)

    tmp = trimesh.util.concatenate(bbox_edge_list)
    tmp.visual.face_colors = color

    # z axis to x axis
    # R = np.array([[0,0,1],[1,0,0],[0,1,0]]).reshape(3,3)
    # t =  np.array([0, 0, -1.12169998e-01]).reshape(3,1)
    #
    # T = np.r_[np.c_[np.eye(3), t], [[0, 0, 0, 1]]]
    # tmp.apply_transform(T)

    return tmp


def params_to_8points_3dfront(box, degrees=False):
    """ Given bounding box as 7 parameters: l, h, w, cx, cy, cz, z, compute the 8 corners of the box
    """
    l, h, w, px, py, pz, angle = box
    points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                points.append([l.item()/2 * i, h.item() * j, w.item()/2 * k])
    points = np.asarray(points)
    points = points.dot(get_rotation_3dfront(angle.item(), degree=degrees))
    points += np.expand_dims(np.array([px.item(), py.item(), pz.item()]), 0)
    return points

def get_rotation_3dfront(y, degree=True):
    if degree:
        y = np.deg2rad(y)
    rot = np.array([[np.cos(y),     0,  -np.sin(y)],
                    [       0 ,     1,           0],
                    [np.sin(y),     0,   np.cos(y)]])
    return rot



def render_img(trimesh_meshes):
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
    for tri_mesh in trimesh_meshes:
        pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        scene.add(pyrender_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2)

    # set up positions and the origin
    camera_location = np.array([0.0, 8.0, 0.0])  # y axis
    look_at_point = np.array([0.0, 0.0, 0.0])
    up_vector = np.array([0.0, 0.0, -1.0])  # -z axis

    camera_direction = (look_at_point - camera_location) / np.linalg.norm(look_at_point - camera_location)
    right_vector = np.cross(camera_direction, up_vector)
    up_vector = np.cross(right_vector, camera_direction)

    camera_pose = np.identity(4)
    camera_pose[:3, 0] = right_vector
    camera_pose[:3, 1] = up_vector
    camera_pose[:3, 2] = -camera_direction
    camera_pose[:3, 3] = camera_location
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    point_light = pyrender.PointLight(color=np.ones(3), intensity=20.0)
    scene.add(point_light, pose=camera_pose)
    color, depth = renderer.render(scene)
    return color