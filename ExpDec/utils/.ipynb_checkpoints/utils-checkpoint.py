import yaml

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def save_config_to_yaml(config, path):
    assert path.endswith('.yaml')
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
        f.close()

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr