import numpy as np
def create_spirals(n_points, n_spirals=3, noise=0.1, seed=100,shuffle = True):
    """
    Returns array of n_spiral spirals with increasing noise and radius
    """
    np.random.seed(seed)

    angle_separation = 2 * np.pi / n_spirals  # The angle separation between each spiral

    X, Y = [], []
    for i in range(n_spirals):
        X.append(create_single_spiral(n_points, angle_offset=angle_separation * i, noise=noise))
        Y.append(np.ones(n_points) * i)

    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    #if not shuffle:
    #    return X, Y
    #XY = np.concatenate((X,Y),axis = 1)
    #np.random.shuffle(XY)
    #X = XY[:,:-1]
    #Y = XY[:,-1]
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.long)
def create_single_spiral(n_points, angle_offset, noise=0.1):
    # Create numbers in the range [0., 6 pi], where the initial square root maps the uniformly
    # distributed points to lie mainly towards the upper limit of the range
    n = np.sqrt(np.random.rand(n_points, 1)) * 3 * (2 * np.pi)

    # Calculate the x and y coordinates of the spiral and add random noise to each coordinate
    x = -np.cos(n + angle_offset) * n ** 2 + np.random.randn(n_points, 1) * noise * n * np.sqrt(n)
    y = np.sin(n + angle_offset) * n ** 2 + np.random.randn(n_points, 1) * noise * n * np.sqrt(n)

    return np.hstack((x, y))

def create_circle(n_points, radius=410, noise=0.01, seed=100):
    np.random.seed(seed)

    samples = np.random.randn(n_points, 2)
    # Sample random points on a circle
    circle = radius * samples / (np.sqrt(np.sum(samples ** 2, axis=1, keepdims=True)))
    X = circle + noise * np.random.randn(n_points, 2)
    Y = -1*np.ones((n_points,1))
    #return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.long)
    #if not shuffle:
    #    return X, Y
    #XY = np.concatenate((X,Y),axis = 1)
    #np.random.shuffle(XY)
    #X = XY[:,:-1]
    #Y = XY[:,-1]
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.long)

def create_mixed_data(ID_points,OOD_points,n_spirals,radius=410,ID_noise=0.1,OOD_noise=0.1,seed = 100, shuffle = True):
    ID_X, ID_Y = create_spirals(ID_points,n_spirals,ID_noise,seed, shuffle)
    OOD_X, OOD_Y = create_circle(OOD_points, radius,OOD_noise,seed)
    ID_X = np.squeeze(ID_X)
    ID_Y = np.squeeze(ID_Y)
    OOD_Y = np.squeeze(OOD_Y)
    X = np.concatenate((ID_X,OOD_X))
    Y = np.concatenate((ID_Y,OOD_Y))
    Y = np.expand_dims(Y,axis = 1)
    if not shuffle:
        return X, Y
    XY = np.concatenate((X,Y),axis = 1)
    np.random.shuffle(XY)
    X = XY[:,:-1]
    Y = XY[:,-1]
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.long)
