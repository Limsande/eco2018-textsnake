"""
Partition the Ecotron-EInsect-2018 dataset into training, validation, and testing sets.

Usage:
    partition.py <image-source> <mask-source> <train-split> <test-split> <val-split> [--dry-run]

Synopsis:
    partition.py images masks 70 20 10

Parameter:
    <image-source>  directory with images
    <mask-source>  directory with corresponding masks
    <train-split>  percentage of data going into training set
    <test-split>  percentage of data going into test set
    <val-split>  percentage of data goig into validation set
    --dry-run  Only simulate the process, but don't actually touch anything

Images can overlap in both x and y direction. We make sure that no data goes into more than one split. Because of the
overlaps, there is no other possibility than simply ignoring some of the crops, i.e. do not put them in any split. The
splits are moved into subdirectories "training", "test", and "validation".

We want parts of every image in all splits. The idea is to select <test-split> % of each image for the test split and
<val-split> % for the validation split. We do this by computing how many not-overlapping crops fit into the origin image
width. Since this computation is done for each image, we can perfectly deal with images of different widths.

With this number we can separate an image into a number of columns. We than conceptually extend these columns to the
full image height. For the test split we select #columns/100*<test-split> (rounded to next int) of them. All
not-overlapping crops being *completely* in *one* of the selected columns go into the test split. We alternate starting
to select from top to bottom. This is because otherwise, if there is only one row of not-overlapping crops, we would
bias the upper edge.

Finally, to guarantee that no part of the selected crops goes into another split because of overlaps, all crops
overlapping the selected ones are excluded from any further selection.

Additional #columns/100*<val-split> columns are selected for the validation split and processed as just described.

While running this algorithm once per image, we start selecting columns from the left, moving further to the right with
every image and go back to the left side once the right edge is reached.

NOTE that this only works with origin images in landscape format which were cropped seamlessly, i.e. the crop size is a
multiple of the offset. It is also not possible to select less than 100/#columns % of data for test and validation
split, respectively.

This script is designed to work with the output of dl_cropImages.sh. However, it should work with every input files as
long as the file names match this format:
    <arbitrary string>-<crop size x>x<crop size y>+<position x>+<position y>.<arbitrary file type>
Precisely, the names must match this regular expression:
    ^.*-[0-9]+x[0-9]+\\+[0-9]+\\+[0-9]+\\.[a-zA-Z0-9_]+$
An example:
    EInsect_T017_CTS_09.05.18_000000_1_HMC-00510x00510+00000+00000.tif
"""

import argparse
import pathlib

import numpy as np
import os
import re
import shutil
import sys

test_mode = True

n_removed = 0


def get_args_parser() -> argparse.ArgumentParser:
    """
    Defines and returns a parser for given command line arguments.
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Partition the Ecotron-EInsect-2018 dataset into training, validation, and testing sets.'
    )
    data_group = parser.add_argument_group('Data input')
    data_group.add_argument(
        '--images',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with images'
    )
    data_group.add_argument(
        '--root-masks',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with root masks for given images'
    )
    data_group.add_argument(
        '--centerline-masks',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with center line masks for given images'
    )
    data_group.add_argument(
        '--radii-maps',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with radii maps for given images'
    )
    data_group.add_argument(
        '--sin-maps',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with sine maps for given images'
    )
    data_group.add_argument(
        '--cos-maps',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with cosine maps for given images'
    )

    split_group = parser.add_argument_group('Split control')
    split_group.add_argument(
        '--train-split',
        type=int,
        required=True,
        metavar='INT',
        help='percentage of data going into training set'
    )
    split_group.add_argument(
        '--val-split',
        type=int,
        required=True,
        metavar='INT',
        help='percentage of data going into validation set'
    )
    split_group.add_argument(
        '--test-split',
        type=int,
        required=True,
        metavar='INT',
        help='percentage of data going into test set'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only simulate the process, but don\'t actually touch anything'
    )

    return parser


def get_crops_and_remove_from_listing(img, col, crop_x, crop_y, split, bottom=False) -> [(str, str)]:
    """
    Get crops in <col> (0-based) of <img> (both image and mask); remove crops
    from file listing.
    """

    # We want all crops of <img> in the n.th column. We know that these have all
    # the same x value, which is exactly n*crop_x.
    res = []
    if bottom:
        max_y = max(file_mapping[img]['y'])
        n_taken = 0
        for i in reversed(range(len(file_mapping[img]['image']))):
            if file_mapping[img]['x'][i] == col * crop_x and max_y - (n_taken * crop_y) == file_mapping[img]['y'][i]:
                res.append((file_mapping[img]['image'][i], file_mapping[img]['mask'][i]))
                locations[orig_img][split].append((file_mapping[img]['x'][i], file_mapping[img]['y'][i]))
                for k in file_mapping[img].keys():
                    file_mapping[img][k][i] = None
    else:
        for i in range(len(file_mapping[img]['image'])):
            if file_mapping[img]['x'][i] == col*crop_x and file_mapping[img]['y'][i] % crop_y == 0:
                res.append((file_mapping[img]['image'][i], file_mapping[img]['mask'][i]))
                locations[orig_img][split].append((file_mapping[img]['x'][i], file_mapping[img]['y'][i]))
                for k in file_mapping[img].keys():
                    file_mapping[img][k][i] = None
    for k in file_mapping[img].keys():
        file_mapping[img][k] = [val for val in file_mapping[img][k] if val is not None]

    return res


def remove_overlaps(img, col, crop_x):
    """
    Remove all crops that overlap given column.
    """
    # Overlapping crops have an x coordinate with <col>*(crop_x -1) < x < <col>*(crop_x +1).
    global n_removed
    to_remove = 0
    for i in range(len(file_mapping[img]['image'])):
        if col * (crop_x - 1) < file_mapping[img]['x'][i] < col*(crop_x + 1):
            for k in file_mapping[img].keys():
                file_mapping[img][k][i] = None
            to_remove += 1
    for k in file_mapping[img].keys():
        file_mapping[img][k] = [val for val in file_mapping[img][k] if val is not None]

    n_removed += to_remove


args_parser = get_args_parser()
args = args_parser.parse_args()

args.train_split = int(sys.argv[3])
args.train_split = int(sys.argv[4])
args.train_split = int(sys.argv[5])

assert args.train_split + args.train_split + args.train_split == 100,\
    f'Splits must sum up to 100: {args.train_split}, {args.val_split}, {args.test_split}'

valid_file_name = re.compile(r'.*-[0-9]+x[0-9]+\+[0-9]+\+[0-9]+\.[a-zA-Z0-9_]+$')

# First, we must collect all cropped images and masks files.
image_src = sys.argv[1]
mask_src = sys.argv[2]
try:
    file_lists = {'images': sorted(os.listdir(image_src)), 'masks': sorted(os.listdir(mask_src))}
except IOError as e:
    sys.exit(f'Could not load images: {e}')
assert len(file_lists['images']) == len(file_lists['masks'])

# We need the original image names. We can reconstruct them from
# the cropped files, e.g.:
#   EInsect_T017_CTS_09.05.18_000000_1_HMC-00510x00510+00000+00000.tif
orig_images = list(np.unique([f.split('-')[0] for f in file_lists['images']]))
n_images = len(orig_images)

locations = {orig_img: {'test': [], 'validation': []} for orig_img in orig_images}

if args.dry_run:
    print()
    print('== This is a dry run, no files will be moved or changed! ==')

print(f"""
Found {len(file_lists['images'])} files in {sys.argv[1]}
Found {len(file_lists['masks'])} files in {sys.argv[2]}

By file names, we think these are crops of {n_images} original image(s).

    Partitioning:
     - training set:    {args.train_split} %
     - validation set:  {args.train_split} %
     - test set:        {args.train_split} %

    Splits will be created in: {os.getcwd()}

Ok? yes/no: """, end='')

if input() != 'yes':
    sys.exit()

args.train_split /= 100
args.train_split /= 100
args.train_split /= 100

# With the original image names and the file lists we now can create
# a mapping between them. We can later benefit from also including x
# and y coordinates for each crop.
file_mapping = {orig_img: {'image': [], 'mask': [], 'x': [], 'y': []} for orig_img in orig_images}
for idx, img in enumerate(file_lists['images']):
    # If a file name does not match the required format, something is probably
    # wrong with our input. We better stop.
    if valid_file_name.match(img) is None:
        print(f"File name has illegal format: {img}", file=sys.stderr)
        sys.exit('Abort.')
    if valid_file_name.match(file_lists['masks'][idx]) is None:
        print(f"File name has illegal format: {file_lists['masks'][idx]}", file=sys.stderr)
        sys.exit('Abort.')

    orig_name = img.split('-')[0]
    file_mapping[orig_name]['image'].append(img)
    x, y = [int(loc) for loc in img.split('.')[-2].split('+')[-2:]]
    file_mapping[orig_name]['mask'].append(file_lists['masks'][idx])
    file_mapping[orig_name]['x'].append(x)
    file_mapping[orig_name]['y'].append(y)

# We also need to know, how many columns and rows of not-overlapping crops
# each image has. We can do this by computing how often we can, in a sorted
# list of unique x (y) coordinates, add the crop width (height) and still
# find an x (y) coordinate at this or a greater position.
rows = {}
cols = {}
n_cols_to_select_test = {}
n_cols_to_select_val = {}
unique_x = {}
unique_y = {}
crop_x = crop_y = 0
for orig_img in orig_images:
    images = file_mapping[orig_img]['image']
    u_x = np.unique(file_mapping[orig_img]['x'])
    u_y = np.unique(file_mapping[orig_img]['y'])
    unique_x[orig_img] = u_x
    unique_y[orig_img] = u_y

    n_rows = n_cols = 1
    sum_x = sum_y = 0
    # To compute the number of columns we need the size of the crop. This is also encoded
    # in the file name:
    #   EInsect_T017_CTS_09.05.18_000000_1_HMC-00510x00510+00000+00000.tif
    # where 00510x00510 stands for a 510 by 510 crop.
    crop_x, crop_y = [int(size) for size in file_mapping[orig_img]['image'][0].split('-')[-1].split('+')[0].split('x')]
    while sum_x < u_x[-1]:
        sum_x += crop_x
        n_cols += 1
    while sum_y < u_y[-1]:
        sum_y += crop_y
        n_rows += 1
    rows[orig_img] = n_rows
    cols[orig_img] = n_cols
    n_cols_to_select_test[orig_img] = round(n_cols * args.train_split)
    n_cols_to_select_val[orig_img] = round(n_cols * args.train_split)

print(f"""
    Computed an average of {np.mean(list(cols.values()))} columns per image.
    Computed an average of {np.mean(list(rows.values()))} rows per image.
    Will select {np.mean(list(n_cols_to_select_test.values()))} test columns on average.
    Will select {np.mean(list(n_cols_to_select_val.values()))} validation columns on average.

Go on? yes/no:""", end=' ')
if input() != 'yes':
    sys.exit()

# We're ready to start the partitioning: Loop over the original images and select
# the appropriate number of columns for test and validation split with an increasing
# offset. Then remove each overlapping crop from the file listing. Finally all
# remaining crops go into the training set.
print('Partitioning...')
partitions = {'training': [], 'test': [], 'validation': []}
offset = 0
for i, orig_img in enumerate(orig_images):
    # Test V
    # |X|X|O| | | | | | | |
    # |X|X| | | | | | | | |
    # ------   Training

    #         Test V
    # | | | | |X|X|O| | | |
    # Training ----- Train

    print(f"    Working with {orig_img} (image {i+1}/{len(orig_images)})...")

    # Reset offset to 0, if we've reached the right edge.
    if cols[orig_img] - offset < n_cols_to_select_test[orig_img] + n_cols_to_select_val[orig_img]:
        # 4 cols, want 2+1, offset 2
        #  0 1 2 3
        # | | |X|X|
        #      ^
        # 4 - 2 < 2+1 --> offset = 0
        offset = 0

    # Collect crops for test set
    local_offset = 0
    selected_cols = []
    bottom = False
    for col in range(n_cols_to_select_test[orig_img]):
        selected_cols.append(col + offset)
        partitions['test'].extend(get_crops_and_remove_from_listing(orig_img, col + offset + local_offset, crop_x, crop_y, 'test', bottom))
        bottom = not bottom
    # Collect crops for validation set
    for col in range(n_cols_to_select_val[orig_img]):
        selected_cols.append(col + offset + n_cols_to_select_val[orig_img])
        partitions['validation'].extend(get_crops_and_remove_from_listing(orig_img, col + offset + n_cols_to_select_test[orig_img], crop_x, crop_y, 'validation', bottom))
        local_offset += 1
        bottom = not bottom
    offset += 1

    # Now all crops overlapping col <col> have to be removed.
    for col in selected_cols:
        remove_overlaps(orig_img, col, crop_x)

# All remaining files go into training set.
partitions['training'] = [(img, mask) for val in file_mapping.values() for img, mask in zip(val['image'], val['mask'])]

print(f"""
    Partitions:
     - training set:    {len(partitions['training'])} images
     - validation set:  {len(partitions['validation'])} images
     - test set:        {len(partitions['test'])} images

    Excluded because of overlaps: {n_removed}
""")

if test_mode:
    print('Files in test set:')
    for img, mask in partitions['test']:
        print('  ', img, mask)
    print('Files in validation set:')
    for img, mask in partitions['validation']:
        print('  ', img, mask)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    for orig_img in orig_images:
        fig, ax = plt.subplots(1)
        plt.xlim(0, 5000)
        plt.ylim(0, 710)
        for loc in locations[orig_img]['test']:
            ax.add_patch(Rectangle((float(loc[0]), float(loc[1])), width=510, height=510, fill=False, color='red', label='test'))
        for loc in locations[orig_img]['validation']:
            ax.add_patch(Rectangle((float(loc[0]), float(loc[1])), width=510, height=510, fill=False, color='blue', label='validation'))
        fig.legend()
        #plt.show()
        fig.savefig(f'{orig_img}_selection.png')
    sys.exit()

# Since the partition is done, we can move all files into their respective subdirectory.
# This leaves the removed overlaps untouched. Fail if output directories already exist.
if not args.dry_run:
    for directory in ['training', 'validation', 'test']:
        try:
            os.makedirs(os.path.join(directory, 'images'))
            os.makedirs(os.path.join(directory, 'masks'))
        except IOError as e:
            sys.exit(e)

print('Moving files...')
counter = 0
for split in ['training', 'validation', 'test']:
    for image, mask in partitions[split]:
        if not args.dry_run:
            try:
                shutil.move(os.path.join(image_src, image), os.path.join(split, 'images'))
                shutil.move(os.path.join(mask_src, mask), os.path.join(split, 'masks'))
            except IOError as e:
                print(f'ERROR: {e}', file=sys.stderr)
                print('Moving files back...', file=sys.stderr)
                # Clean up: move everything back
                for split in ['training', 'validation', 'test']:
                    for image in os.listdir(os.path.join(split, 'images')):
                        shutil.move(os.path.join(split, 'images', image), image_src)
                    for mask in os.listdir(os.path.join(split, 'masks')):
                        shutil.move(os.path.join(split, 'masks', mask), mask_src)
                for directory in ['training', 'validation', 'test']:
                    shutil.rmtree(directory)
                print('Done.', file=sys.stderr)
                sys.exit(1)
        counter += 2
print(f'{counter} files moved.')

if args.dry_run:
    print()
    print('== This was a dry run, nothing was actually moved. ==')
