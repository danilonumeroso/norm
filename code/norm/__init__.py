def set_seed(seed):
    import numpy as np
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_date():
    from datetime import datetime
    return datetime.utcnow().isoformat()[:-7]
