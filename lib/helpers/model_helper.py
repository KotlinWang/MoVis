from lib.models.movis import build_monodetr


def build_model(cfg):
    return build_monodetr(cfg)
