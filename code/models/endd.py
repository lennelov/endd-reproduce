from utils import losses


def get_model(base_model):
    """Take an uncompiled model and return model compiled for ENDD.

    Warning: This function works in place. Model is returned only for
    conveniance.
    """
    base_model.compile(optimizer='adam', loss=losses.DirichletEnDDLoss())
    return base_model
