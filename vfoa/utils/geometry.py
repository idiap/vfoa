import numpy as np


def getEulerAngles(R):
    """
    Obtain the Euler angles corresponding to the rotation matrix R
    IMPORTANT: This function assumes all the angles are within the range -90 to 90 degrees!!
    """
    rx = -np.arcsin(R[1, 2])
    ry = np.arctan2(R[0, 2], R[2, 2])
    rz = np.arctan2(R[1, 0], R[1, 1])

    return rz, rx, ry


def vectorToYawElevation(vector, unit='deg'):
    """ Convert a vector to two angle representation by building the rotation matrix where the input vector is z axis.
        The input vector become the z axis, x axis is kept in the XZ plan of the coordinate system, on the left of the
        z axis. The y axis is deduced from the two firsts.
        The output angles are:
            - the first angle is the euler's yaw, called "yaw" or "phi" (rotation around y);
            - the second is the inverse euler's pitch, called "elevation" or "theta" (rotation around x). """
    vector = np.resize(vector, (1, 3))[0]

    R = [0.0, 0.0, 0.0]
    R[2] = vector / np.linalg.norm(vector)
    R[0] = np.cross([0.0, 1.0, 0.0], R[2])
    R[1] = np.cross(R[2], R[0])

    roll, pitch, yaw = getEulerAngles(np.array(R).T)
    if unit == 'deg':
        yaw = yaw / np.pi*180
        elevation = -pitch / np.pi*180
    elif unit == 'rad':
        elevation = -pitch
    else:
        raise Exception('Unknow angle unit ' + unit)

    return [yaw, elevation]


def yawElevationToVector(yawElevation, unit='deg'):
    """ Transforms an angular representation of gaze (phi, theta) into a vectorial one (x,y,z).
        WARNING: angles outer than [-90, 90] will give wrong vectors """
    # TODO: generalize with matrix rotation
    yawElevation = map(float, yawElevation)
    if unit == 'deg':
        yaw, elevation = yawElevation[0]/180*np.pi, yawElevation[1]/180*np.pi
        if np.abs(yaw) >= 90 or np.abs(elevation) >= 90:
            raise Exception('Can not handle angles outer [-90, 90]')
    elif unit == 'rad':
        yaw, elevation = yawElevation
        if np.abs(yaw) >= np.pi/2 or np.abs(elevation) >= np.pi/2:
            raise Exception('Can not handle angles outer [-90, 90]')
    else:
        raise Exception('Unknow angle unit ' + unit)

    y = np.sin(elevation)

    x = np.sqrt((np.cos(elevation)**2 * np.tan(yaw)**2) / (1 + np.tan(yaw)**2))
    if yaw < 0:  # The signed of x is taken from the sign of the phi
        x = -x

    if x**2 + y**2 >= 1.0:
        z = 0.0
    else:
        z = np.sqrt(1 - x**2 - y**2)
    if np.isnan(z):
        print '[yawElevationToVector]', np.array(yawElevation).flatten()*180/np.pi, [x, y, z]

    return np.resize([x, y, z], (3, 1))


def angleBetweenVectors_deg(x, y):
    """ Angle in degree between x and y vectors """
    x = np.resize(x.copy(), (3, 1))
    y = np.resize(y.copy(), (3, 1))
    angle = np.arccos(np.dot(x.T, y) / (np.linalg.norm(x) * np.linalg.norm(y))) / np.pi * 180
    return np.linalg.norm(angle)
