import numpy as np
import torch


def torch_unfold_camera_param(camera, device=None):
    R = torch.as_tensor(camera['R'], dtype=torch.float, device=device)
    T = torch.as_tensor(camera['T'], dtype=torch.float, device=device)
    fx = torch.as_tensor(camera['fx'], dtype=torch.float, device=device)
    fy = torch.as_tensor(camera['fy'], dtype=torch.float, device=device)
    f = torch.stack([fx, fy], dim=-1).reshape(-1, 2, 1)
    cx = torch.as_tensor(camera['cx'], dtype=torch.float, device=device)
    cy = torch.as_tensor(camera['cy'], dtype=torch.float, device=device)
    c = torch.stack([cx, cy], dim=-1).reshape(-1, 2, 1)
    k = torch.as_tensor(camera['k'], dtype=torch.float, device=device)
    p = torch.as_tensor(camera['p'], dtype=torch.float, device=device)

    return R, T, f, c, k, p


def torch_project_pose(x, camera):
    R, T, f, c, k, p = torch_unfold_camera_param(camera, device=x.device)
    return torch_project_point(x, R, T, f, c, k, p)


def torch_project_point(x, R, T, f, c, k, p):
    """
    Args
        x: BxPxJx3 points in world coordinates
        R: Bx3x3 Camera rotation matrix
        T: Bx3x1 Camera translation parameters
        f: Bx2x1 Camera focal length
        c: Bx2x1 Camera center
        k: Bx3x1
        p: Bx2x1
    Returns
        y: Bx...x2 points in pixel space
    """
    batch_size, num_persons, num_joints, num_dimension = x.shape
    assert num_dimension == 3

    x = x.reshape(batch_size, -1, 3).transpose(1, 2)  # [B, 3, PJ]

    xcam = torch.bmm(R, x - T)  # [B, 3, PJ]
    y = xcam[:, :2, :] / (xcam[:, 2:, :] + 1e-5)  # [B, 2, PJ]

    # === add camera distortion
    r = torch.sum(y ** 2, dim=1)  # [B, PJ]
    d = 1 + k[:, 0] * r + k[:, 1] * r * r + k[:, 2] * r * r * r  # [B, PJ]
    u = y[:, 0, :] * d + 2 * p[:, 0] * y[:, 0, :] * y[:, 1, :] + p[:, 1] * (r + 2 * y[:, 0, :] * y[:, 0, :])  # [B, PJ]
    v = y[:, 1, :] * d + 2 * p[:, 1] * y[:, 0, :] * y[:, 1, :] + p[:, 0] * (r + 2 * y[:, 1, :] * y[:, 1, :])  # [B, PJ]
    y = torch.stack([u, v], dim=1)  # [B, 2, PJ]

    ypixel = f * y + c  # [B, 2, PJ]
    ypixel = ypixel.transpose(1, 2)  # [B, PJ, 2]
    ypixel = ypixel.reshape(batch_size, num_persons, num_joints, 2)  # [B, P, J, 2]

    return ypixel


def torch_back_project_pose(y, depth, camera):
    R, T, f, c, k, p = torch_unfold_camera_param(camera, device=y.device)
    return torch_back_project_point(y, depth, R, T, f, c, k, p)


def torch_back_project_point(y, depth, R, T, f, c, k, p):
    """
    Args
        y: BxPxJx2 points in image frame
        depth: B or BxP
        R: Bx3x3 Camera rotation matrix
        T: Bx3x1 Camera translation parameters
        f: Bx2x1 Camera focal length
        c: Bx2x1 Camera center
        k: Bx3x1
        p: Bx2x1
    Returns
        x: Bx...x3 points in world coordinate
    """
    batch_size, num_persons, num_joints, num_dimension = y.shape
    assert num_dimension == 2

    y = y.reshape(batch_size, -1, 2).transpose(1, 2)  # [B, 2, PJ]

    xcam = (y - c) / f  # [B, 2, PJ]

    # === remove camera distortion (approx)
    r = torch.sum(xcam ** 2, dim=1)  # [B, PJ]
    d = 1 - k[:, 0] * r - k[:, 1] * r * r - k[:, 2] * r * r * r  # [B, PJ]
    u = xcam[:, 0, :] * d - 2 * p[:, 0] * xcam[:, 0, :] * xcam[:, 1, :] - p[:, 1] * (r + 2 * xcam[:, 0, :] * xcam[:, 0, :])  # [B, PJ]
    v = xcam[:, 1, :] * d - 2 * p[:, 1] * xcam[:, 0, :] * xcam[:, 1, :] - p[:, 0] * (r + 2 * xcam[:, 1, :] * xcam[:, 1, :])  # [B, PJ]
    xcam = torch.stack([u, v], dim=1)  # [B, 2, PJ]

    xcam = torch.cat([xcam, torch.ones(batch_size, 1, xcam.size(-1)).to(xcam.device)], dim=1)  # [B, 3, PJ]
    xcam = xcam.reshape(batch_size, 3, num_persons, num_joints)  # [B, 3, P, J]
    d = depth.reshape(batch_size, 1, -1, 1)  # [B, 1, 1 or P, 1]
    xcam = xcam * d
    xcam = xcam.reshape(batch_size, 3, -1)  # [B, 3, PJ]

    x = torch.bmm(torch.inverse(R), xcam) + T  # [B, 3, PJ]
    x = x.transpose(1, 2)  # [B, PJ, 3]
    x = x.reshape(batch_size, num_persons, num_joints, 3)  # [B, P, J, 3]

    return x


def unfold_camera_param(camera):
    R = camera['R']
    T = camera['T']
    f = np.array([[camera['fx']], [camera['fy']]]).reshape(-1, 1)
    c = np.array([[camera['cx']], [camera['cy']]]).reshape(-1, 1)
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def project_pose(x, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return project_point(x, R, T, f, c, k, p)


def project_point(x, R, T, f, c, k, p):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: 2x1 Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
        depth: N points
    """
    xcam = R.dot(x.T - T)
    y = xcam[:2] / (xcam[2] + 1e-5)

    # === add camera distortion
    r = np.sum(y ** 2, axis=0)
    d = 1 + k[0] * r + k[1] * r * r + k[2] * r * r * r
    u = y[0, :] * d + 2 * p[0] * y[0, :] * y[1, :] + p[1] * (r + 2 * y[0, :] * y[0, :])
    v = y[1, :] * d + 2 * p[1] * y[0, :] * y[1, :] + p[0] * (r + 2 * y[1, :] * y[1, :])
    y[0, :] = u
    y[1, :] = v

    ypixel = np.multiply(f, y) + c

    depth = xcam[2]

    return ypixel.T, depth


def back_project_pose(y, depth, camera):
    R, T, f, c, k, p = unfold_camera_param(camera)
    return back_project_point(y, depth, R, T, f, c, k, p)


def back_project_point(y, depth, R, T, f, c, k, p):
    """
    Args
        y: Nx2 points in image frame
        depth: N points
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: 2x1 Camera focal length
        c: 2x1 Camera center
        k: 3x1
        p: 2x1
    Returns
        x: Nx3 points in world coordinate
    """
    n = y.shape[0]

    xcam = (y.T - c) / f  # [2, N]

    # === remove camera distortion (approx)
    r = xcam[0, :] * xcam[0, :] + xcam[1, :] * xcam[1, :]
    d = 1 - k[0] * r - k[1] * r * r - k[2] * r * r * r
    u = xcam[0, :] * d - 2 * p[0] * xcam[0, :] * xcam[1, :] - p[1] * (r + 2 * xcam[0, :] * xcam[0, :])
    v = xcam[1, :] * d - 2 * p[1] * xcam[0, :] * xcam[1, :] - p[0] * (r + 2 * xcam[1, :] * xcam[1, :])
    xcam[0, :] = u
    xcam[1, :] = v

    xcam = np.concatenate([xcam, np.ones([1, n])], axis=0)  # [3, N]
    xcam = xcam * depth

    invR = np.linalg.inv(R)
    x = invR @ xcam + T

    return x.T


def rotate_points(points, center, rot_rad):
    """
    :param points:  N*2
    :param center:  2
    :param rot_rad: scalar
    :return: N*2
    """
    rot_rad = rot_rad * np.pi / 180.0
    rotate_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                          [np.sin(rot_rad), np.cos(rot_rad)]])
    center = center.reshape(2, 1)
    points = points.T
    points = rotate_mat.dot(points - center) + center

    return points.T
