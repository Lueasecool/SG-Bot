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

def render_box(obj_ids, predBoxes, predAngles, render_type='onlybox',
               store_img=False, render_boxes=False, demo=False, visual=False, without_lamp=False,
               str_append="", mani=0, missing_nodes=None, manipulated_nodes=None, objs_before=None, store_path=None):
    os.makedirs(store_path,exist_ok=True)
    if render_type not in ['txt2shape', 'retrieval', 'onlybox']:
        raise ValueError('Render type needs to be either set to txt2shape or retrieval or onlybox.')
    color_palette = np.array(sns.color_palette('hls', 100))


    box_and_angle = torch.cat([predBoxes.float(), predAngles.float()], dim=-1)

    obj_n = len(box_and_angle)

    # mesh_dir = os.path.join(store_path, render_type, 'object_meshes', "hello")
    # os.makedirs(mesh_dir, exist_ok=True)
    
    if render_type == 'onlybox':
        lamp_mesh_list, trimesh_meshes = get_bbox(box_and_angle,obj_ids, colors=color_palette[obj_ids])
    else:
        raise NotImplementedError



    scene = trimesh.Scene(trimesh_meshes)
    scene_path = os.path.join(store_path, render_type)
    if len(str_append) > 0:
        render_type += str_append
    os.makedirs(scene_path, exist_ok=True)
    scene.export(os.path.join(scene_path, "{0}_{1}.glb".format('hello', render_type)))

    
    # scene.show()


    img_path = os.path.join(store_path, render_type, "render_imgs")
    os.makedirs(img_path, exist_ok=True)
    color_img = render_img(trimesh_meshes)
    color_bgr = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGR)
    file_name = "hello"
    if len(str_append) > 0:
            file_name += str_append
    cv2.imwrite(os.path.join('/remote-home/2332082/ArrangeBot/Arrange/scripts/train_iter', f'{file_name}.png'), color_bgr)
    print("已生成图片")
    if mani==1:
        return trimesh_meshes




def get_bbox(boxes,obj_ids, colors):
    trimesh_meshes = []
    colors = iter(colors)
    lamp_mesh_list=[]
    print(boxes)
    print(boxes.shape)
    classes=['_scene_',
    'bowl',
    'box',
    'can',
    'cup',
    'fork',
    'knife',
    'pitcher',
    'plate',
    'support_table',
    'tablespoon',
    'teapot',
    'teaspoon',
    'obstacle']
    for j in range(0, boxes.shape[0]):
        query_label = classes[obj_ids[j]-1].strip('\n')
        print("rendering", classes[obj_ids[j]].strip('\n') )
        if query_label == '_scene_' or query_label == 'support_table':
            continue
        box_points = params_to_8points_3dfront(boxes[j], degrees=True)
        print('box_points',box_points)
        trimesh_meshes.append(create_bbox_marker(box_points, tube_radius=0.02, color=next(colors)))
        # if query_label == 'nightstand':
        #     trimesh_meshes.pop()
        
    

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