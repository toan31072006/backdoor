def build_model(name, num_keypoints=14, num_coor=3, num_person=1,
                subcarrier_num=180, dataset='person-in-wifi-3d', pretrained=False):
    name = name.lower()
    if name == 'hpeli':
        from models.hpeli import HPELiNet, hpeli_init
        m = HPELiNet(num_keypoints, num_coor, subcarrier_num, num_person, dataset)
        m.apply(hpeli_init)
        return m
    if name == 'metafi':
        from models.metafi import MetaFiNet, metafi_init   # requires torchvision
        m = MetaFiNet(num_keypoints, num_coor, num_person, dataset, pretrained)
        m.apply(metafi_init)
        return m
    raise ValueError(f'unknown model {name}')
