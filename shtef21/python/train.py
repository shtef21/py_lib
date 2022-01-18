
def train(model, **kwargs):

    assert all(p in kwargs for p in 'optimizer', 'loss_fn'), 'Missing required parameters!'
