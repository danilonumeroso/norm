def random_search(space, num_samples):
    configs = []
    for _ in range(num_samples):
        c_ = {}
        for param_name, sample in space.items():
            c_[param_name] = sample()

        configs.append(c_)

    return configs
