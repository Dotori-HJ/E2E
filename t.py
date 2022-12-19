from timm.data import create_transform

transform = create_transform(
    # input_size=384,
    input_size=224,
    is_training=True,
    # color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
    # auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
    # re_prob=config.AUG.REPROB,
    # re_mode=config.AUG.REMODE,
    # re_count=config.AUG.RECOUNT,
    # interpolation=config.DATA.INTERPOLATION,
)

print(transform)