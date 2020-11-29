LABELS = ['BUILDING', 'CLUTTER', 'VEGETATION', 'TREE', 'GROUND', 'CAR']

# Class to color (BGR)
LABELMAP = {
    0 : (255,   0, 255),
    1 : (255,   0, 0),
    2 : (0,  0, 255),
    3 : (255,  255,  0),
    4 : (0,  255, 0),
    5 : (255, 255, 255),
    6 : (0, 255,   255),
}

# Color (BGR) to class
INV_LABELMAP = {
    (255,   0, 255) : 0,    # IGNORE
    (255,   0, 0) : 1,    # BUILDING
    (0,  0, 255) : 2,    # CLUTTER
    (255,  255,  0) : 3,    # VEGETATION
    (0,  255, 0) : 4,    # TREE
    (255, 255, 255) : 5,    # GROUND
    (0, 255,   255) : 6,    # CAR
}

LABELMAP_RGB = { k: (v[2], v[1], v[0]) for k, v in LABELMAP.items() }

INV_LABELMAP_RGB = { v: k for k, v in LABELMAP_RGB.items() }