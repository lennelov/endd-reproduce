from utils.DirichletEnDDLoss import DirichletEnDDLoss

def get_model(base_model):
    """Take an uncompiled model and return model compiled for ENDD.

    Warning: This function works in place. Model is returned only for
    conveniance.
    """
    base_model.compile(optimizer='adam', loss=DirichletEnDDLoss())
    return base_model
