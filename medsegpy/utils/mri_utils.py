import argparse

# TODO (arjundd): discuss if we should assume that the user has properly preprocessed the .seg file (i.e. summed the masks beforehand)
# Mask Channels in .seg files
# Indices where masks occur in .seg files
# Currently assuming that the user has preprocessed the .seg file
MASK_FEMORAL_CARTILAGE = 0  # [0]
MASK_TIBIAL_CARTILAGE = [1, 2]  # [1,2]
MASK_PATELLAR_CARTILAGE = 3  # [3]
MASK_MENISCUS = [4, 5]  # [4,5]

fc = FC = MASK_FEMORAL_CARTILAGE
tc = TC = MASK_TIBIAL_CARTILAGE
pc = PC = MASK_PATELLAR_CARTILAGE
men = MEN = MASK_MENISCUS

SUPPORTED_TISSUES = ['fc', 'tc', 'pc', 'men']


# TODO: REMOVE
def get_tissue_name(inds: list):
    names_to_val = {'fc': MASK_FEMORAL_CARTILAGE,
                    'tc': MASK_TIBIAL_CARTILAGE,
                    'pc': MASK_PATELLAR_CARTILAGE,
                    'men': MASK_MENISCUS}
    names = []
    for ind in inds:
        tissue = ind
        for t in names_to_val.keys():
            if ind == names_to_val[t]:
                tissue = t
                break
        names.append(tissue)

    return names
