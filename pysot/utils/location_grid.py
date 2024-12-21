import torch
def compute_locations(features,stride):
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.device
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = torch.stack((shift_x, shift_y), dim=1) + stride
    locations = torch.stack((shift_x, shift_y), dim=1) + 32  #alex:48 // 32
    return locations

def get_gt_mask(featmap,  gt_bboxes):
    featmap_sizes = featmap[0].size()[-2:]
    a = featmap_sizes
    featmap_strides = 8
    imit_range = [0, 0, 0, 0, 0]
    with torch.no_grad():
        mask_batch = []
        for batch in range(len(gt_bboxes)):
            mask_level = []
            gt_level = gt_bboxes[batch]  # gt_bboxes: BatchsizexNpointx4coordinate
            h, w = featmap_sizes[0], featmap_sizes[1]
            mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
            a = gt_level.shape[0]
            gt_level_map = gt_level / featmap_strides
            lx = max(int(gt_level_map[0]) - imit_range[0], 0)
            rx = min(int(gt_level_map[2]) + imit_range[0], w)
            ly = max(int(gt_level_map[1]) - imit_range[0], 0)
            ry = min(int(gt_level_map[3]) + imit_range[0], h)
            if (lx == rx) or (ly == ry):
                mask_per_img[ly, lx] += 1
            else:
                mask_per_img[ly:ry, lx:rx] += 1
            mask_per_img = (mask_per_img > 0).double()
            mask_level.append(mask_per_img)
            mask_batch.append(mask_level)

            mask_batch_level = []
            for level in range(len(mask_batch[0])):
                tmp = []
                for batch in range(len(mask_batch)):
                    tmp.append(mask_batch[batch][level])
                mask_batch_level.append(torch.stack(tmp, dim=0))

        return mask_batch_level

def get_text_mask(featmap):
    featmap_sizes = featmap.size()[-2:]
    with torch.no_grad():
        mask_batch = []
        mask_level = []# gt_bboxes: BatchsizexNpointx4coordinate
        h, w = featmap_sizes[0], featmap_sizes[1]
        mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
        lx = 3
        rx = 28
        ly = 3
        ry = 28
        if (lx == rx) or (ly == ry):
            mask_per_img[ly, lx] += 1
        else:
            mask_per_img[ly:ry, lx:rx] += 1
        mask_per_img = (mask_per_img > 0).double()
        mask_level.append(mask_per_img)
        mask_batch.append(mask_level)

        mask_batch_level = []
        for level in range(len(mask_batch[0])):
            tmp = []
            for batch in range(len(mask_batch)):
                tmp.append(mask_batch[batch][level])
            mask_batch_level.append(torch.stack(tmp, dim=0))

        return mask_batch_level