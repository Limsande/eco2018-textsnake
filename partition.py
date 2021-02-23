"""
Partition the Ecotron-EInsect-2018 dataset into training, validation, and
testing sets.

The set contains only a few minirhizotron images. Because they are very big and
wide (about 5000x700 pixels), they are cropped into smaller parts before they
are fed into a neural net. For each crop there are also masks for roots and
center lines, and maps for radii, sine and cosine (input for the TextSnake
neural net).

This script partitions these crops into training, validation, and testing set.
But this is tricky because usually these crops overlap in both x and y
direction. We make sure that no data goes into more than one split. Because of
the overlaps, there is no other possibility than simply ignoring some of the
crops, i.e. do not put them in any split. The splits are moved into
subdirectories "training", "test", and "validation".

We want parts of every image in all splits. The idea is to select <test-split> %
of each image for the test split and <val-split> % for the validation split. We
do this by computing how many not-overlapping crops fit into the origin image
width. Since this computation is done for each image, we can perfectly deal with
images of different widths.

With this number we can separate an image into a number of columns, each as wide
as the crop width. For the test split we select #columns/100*<test-split>
(rounded to next int) of them. All not-overlapping crops being *completely* in
*one* of the selected columns go into the test split. We alternate starting to
select from top to bottom. This is because otherwise, if there is only one row
of not-overlapping crops, we would bias the upper edge.

Finally, to guarantee that no part of the selected crops goes into another split
because of overlaps, all crops overlapping the selected ones are excluded from
any further selection.

Additional #columns/100*<val-split> columns are selected for the validation
split and processed as just described.

While running this algorithm once per image, we start selecting columns from the
left, moving further to the right with every image and go back to the left side
once the right edge is reached.

This script is designed to work with the output of dl_cropImages.sh. However, it
should work with every input files as long as the file names match this format:
    <original image file name>-<arbitrary string>+<position x>+<position y>.<arbitrary file type>
where <original image file name> is used as hint for which crops belong to the
same image. Precisely, the names must match this regular expression:
    .*-.*\+[0-9]+\+[0-9]+\.[a-zA-Z0-9_]+$
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

n_removed = 0


def get_args_parser() -> argparse.ArgumentParser:
    """
    Defines and returns a parser for given command line arguments.
    :return: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description='Partition the Ecotron-EInsect-2018 dataset into training, validation, and testing sets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
    data_group.add_argument(
        '--crop-size-x',
        type=int,
        required=True,
        metavar='INT',
        help='crop width'
    )
    data_group.add_argument(
        '--crop-size-y',
        type=int,
        required=True,
        metavar='INT',
        help='crop height'
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
        '-y', '--yes',
        action='store_true',
        help='Assume answer "yes" for all questions'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only simulate the process, don\'t actually touch anything'
    )
    parser.add_argument(
        '--vis',
        action='store_true',
        help='Create visualization of selection in ./vis'
    )

    return parser


def extract_orig_name_from_file_name(file_name: str) -> str:
    return file_name.split('-')[0]


def extract_coordinates_from_file_name(file_name: str) -> (str, str):
    return (int(loc) for loc in file_name.split('.')[-2].split('+')[-2:])


def get_original_image_names(file_names: [str]) -> [str]:
    # We need the original image names. We can reconstruct them from
    # the cropped files, e.g.:
    #   EInsect_T017_CTS_09.05.18_000000_1_HMC-00510x00510+00000+00000.tif
    # becomes
    #   EInsect_T017_CTS_09.05.18_000000_1_HMC
    return list(np.unique([f.split('-')[0] for f in file_names]))


def put_into_split(file_mapping, img, col, split, start_from_bottom=False):
    """
    Get crops in <col> (0-based) of <img> (both image and mask); remove crops
    from file listing.
    """

    # We want all crops of <img> in the n.th column. We know that these have all
    # the same x value, which is exactly n*crop_x.
    crop_x = args.crop_size_x
    crop_y = args.crop_size_y
    if start_from_bottom:
        max_y = max(file_mapping[img]['y'])
        n_taken = 0
        for i in reversed(range(len(file_mapping[img]['images']))):
            if file_mapping[img]['x'][i] == col * crop_x and max_y - (n_taken * crop_y) == file_mapping[img]['y'][i]:
                file_mapping[img]['split'][i] = split
                selected_locations[orig_img][split].append((file_mapping[img]['x'][i], file_mapping[img]['y'][i]))
    else:
        for i in range(len(file_mapping[img]['images'])):
            if file_mapping[img]['x'][i] == col * crop_x and file_mapping[img]['y'][i] % crop_y == 0:
                file_mapping[img]['split'][i] = split
                selected_locations[orig_img][split].append((file_mapping[img]['x'][i], file_mapping[img]['y'][i]))


def remove_overlaps(img, col):
    """
    Remove all crops that overlap given column.
    """
    # Overlapping crops have an x coordinate with <col>*(crop_x -1) < x < <col>*(crop_x +1).
    global n_removed
    crop_x = args.crop_size_x
    to_remove = 0
    for i in range(len(file_mapping[img]['images'])):
        if col * (crop_x - 1) < file_mapping[img]['x'][i] < col*(crop_x + 1):
            for k in file_mapping[img].keys():
                file_mapping[img][k][i] = None
            to_remove += 1
    for k in file_mapping[img].keys():
        file_mapping[img][k] = [val for val in file_mapping[img][k] if val is not None]

    n_removed += to_remove


args_parser = get_args_parser()
args = args_parser.parse_args()

assert args.train_split + args.val_split + args.test_split == 100,\
    f'Splits must sum up to 100: {args.train_split}, {args.val_split}, {args.test_split}'

VALID_FILE_NAME = re.compile(r'.*-.*\+[0-9]+\+[0-9]+\.[a-zA-Z0-9_]+$')
ANNOTATIONS = ['images', 'roots', 'centerlines', 'radii', 'sin', 'cos']

# First, we must collect all cropped images with corresponding masks and maps.
try:
    file_lists = {
        'images': sorted(os.listdir(args.images)),
        'roots': sorted(os.listdir(args.root_masks)),
        'centerlines': sorted(os.listdir(args.centerline_masks)),
        'radii': sorted(os.listdir(args.radii_maps)),
        'sin': sorted(os.listdir(args.sin_maps)),
        'cos': sorted(os.listdir(args.cos_maps)),
    }
except IOError as e:
    sys.exit(f'Could not load images: {e}')
assert all(len(file_lists['images']) == len(l) for l in file_lists.values()),\
    f'No. of files does not match:\n  ' \
    f'Images: {len(file_lists["images"])}\n  ' \
    f'Roots: {len(file_lists["images"])}\n  ' \
    f'Center lines: {len(file_lists["centerlines"])}\n  ' \
    f'Radii: {len(file_lists["radii"])}\n  ' \
    f'Sine: {len(file_lists["sin"])}\n  ' \
    f'Cosine: {len(file_lists["cos"])}'

# We need the original image names (before they were cropped).
# We then can tie all masks and maps of an image together.
orig_images = get_original_image_names(file_lists['images'])
n_images = len(orig_images)

# Keep track of selected crop locations for visualisation
selected_locations = {orig_img: {'test': [], 'validation': []} for orig_img in orig_images}

if args.dry_run:
    print()
    print('== This is a dry run, no files will be moved or changed! ==')

print(
    f'\nFound {len(file_lists["images"])} files in {args.images.resolve()}',
    f'By file names, we think these are crops of {n_images} original image(s):',
    '   ' + '\n '.join(orig_images),
    '\n  Partitioning:',
    f'  - training set:    {args.train_split} %',
    f'  - validation set:  {args.val_split} %',
    f'  - test set:        {args.test_split} %\n',
    f'  Directories for splits will be created in: {os.getcwd()}',
    f'  Assumed crop size: {args.crop_size_x}x{args.crop_size_y}\n',
    'Ok? yes/no: ',
    sep='\n',
    end=''
)

if not args.yes:
    if input() != 'yes':
        sys.exit()
else:
    print('yes')

train_split = args.train_split / 100
val_split = args.val_split / 100
test_split = args.test_split / 100

# With the original image names and the file lists we now can create
# a mapping between them, like
#   img name -> ([crop file...], [root mask file...], ...).
# We can later benefit from also including x and y coordinates for
# each crop.
file_mapping = {
    orig_img: {
        'images': [],
        'roots': [],
        'centerlines': [],
        'radii': [],
        'sin': [],
        'cos': [],
        'x': [],
        'y': [],
        'split': []
    } for orig_img in orig_images}

# Fill the mapping
for idx, img in enumerate(file_lists['images']):
    if VALID_FILE_NAME.match(img) is not None:
        orig_name = extract_orig_name_from_file_name(img)
    else:
        sys.exit(f'File name has illegal format: {img}. Abort.')

    for key in file_lists.keys():
        if VALID_FILE_NAME.match(file_lists[key][idx]) is None:
            # If a file name does not match the required format, something is probably
            # wrong with our input. We better stop.
            sys.exit(f'File name has illegal format: {file_lists[key][idx]}. Abort.')
        else:
            file_mapping[orig_name][key].append(file_lists[key][idx])

    # Extract image coordinates of this crop.
    x, y = extract_coordinates_from_file_name(img)
    file_mapping[orig_name]['x'].append(x)
    file_mapping[orig_name]['y'].append(y)

    # Default value, will be overwritten if necessary.
    file_mapping[orig_name]['split'].append('training')

# We also need to know, how many columns and rows of not-overlapping crops
# each image has. We can do this by computing how often we can, in a sorted
# list of unique x (y) coordinates, add the crop width (height) and still
# find an x (y) coordinate at this or a greater position.

# For statistics only
n_rows_per_image = {}
n_cols_per_image = {}

n_cols_to_select_test = {}
n_cols_to_select_val = {}
for orig_img in orig_images:
    n_cols = max(file_mapping[orig_img]['x']) // args.crop_size_x + 1
    n_rows = max(file_mapping[orig_img]['y']) // args.crop_size_y + 1

    n_rows_per_image[orig_img] = n_rows
    n_cols_per_image[orig_img] = n_cols
    n_cols_to_select_test[orig_img] = round(n_cols * test_split)
    n_cols_to_select_val[orig_img] = round(n_cols * val_split)

if any([n < 1 for n in n_cols_per_image.values()]):
    print(
        '\nHuh?! Not all images are wide enough!\n',
        '  ' + '\n'.join(f'{img}: {n_cols} column(s)' for img, n_cols in n_cols_per_image.items()),
        f'\nIs {args.crop_size_x}x{args.crop_size_y} the correct crop size? Are all images there?',
        sep='\n', file=sys.stderr)
    sys.exit(1)

print(
    f'\n  Computed an average of {np.mean(list(n_cols_per_image.values()))} columns per image.',
    f'  Computed an average of {np.mean(list(n_rows_per_image.values()))} rows per image.',
    f'  Will select {np.mean(list(n_cols_to_select_test.values()))} test columns on average.',
    f'  Will select {np.mean(list(n_cols_to_select_val.values()))} validation columns on average.',
    '\nGo on? yes/no:',
    sep='\n', end=' ')

if not args.yes:
    if input() != 'yes':
        sys.exit()
else:
    print('yes')

# We're ready to start the partitioning: Loop over the original images and select
# the appropriate number of columns for test and validation split with an increasing
# offset. Then remove each overlapping crop from the file listing. Finally all
# remaining crops go into the training set.
print('\nPartitioning...')
offset = 0
n_validation = n_test = 0
for i, orig_img in enumerate(orig_images):
    # Test V
    # |X|X|O| | | | | | | |
    # |X|X| | | | | | | | |
    # ------   Training

    #         Test V
    # | | | | |X|X|O| | | |
    # Training ----- Train

    print(f'  Working with {orig_img} (image {i+1}/{len(orig_images)})...')

    # Reset offset to 0, if we've reached the right edge.
    if n_cols_per_image[orig_img] - offset < n_cols_to_select_test[orig_img] + n_cols_to_select_val[orig_img]:
        # 4 cols, want 2+1, offset 2
        #  0 1 2 3
        # | | |X|X|
        #      ^
        # 4 - 2 < 2+1 --> offset = 0
        offset = 0

    # Collect crops for test set
    local_offset = 0
    selected_cols = []
    start_from_bottom = False
    for col in range(n_cols_to_select_test[orig_img]):
        put_into_split(
            file_mapping,
            orig_img,
            col=col + offset + local_offset,
            split='test',
            start_from_bottom=start_from_bottom
        )
        selected_cols.append(col + offset)
        n_test += 1
        start_from_bottom = not start_from_bottom

    # Collect crops for validation set
    for col in range(n_cols_to_select_val[orig_img]):
        put_into_split(
            file_mapping,
            orig_img,
            col=col + offset + n_cols_to_select_test[orig_img],
            split='validation',
            start_from_bottom=start_from_bottom
        )
        selected_cols.append(col + offset + n_cols_to_select_val[orig_img])
        n_validation += 1
        local_offset += 1
        start_from_bottom = not start_from_bottom
    offset += 1

    # Now all crops overlapping selected columns have to be removed.
    for col in selected_cols:
        remove_overlaps(orig_img, col)

n_training = len(file_lists["images"]) - n_validation - n_test - n_removed
print(
    '\n  Partitions:',
    f'  - training set:    {n_training} images ({n_training * len(ANNOTATIONS)} files)',
    f'  - validation set:  {n_validation} images ({n_validation * len(ANNOTATIONS)} files)',
    f'  - test set:        {n_test} images ({n_test * len(ANNOTATIONS)} files)',
    f'\n  Excluded because of overlaps: {n_removed}',
    sep='\n'
)

if args.vis:
    do_vis = True
    if not os.path.isdir('./vis'):
        try:
            os.mkdir('./vis')
        except IOError as e:
            print(f'Could not create visualization directory: {e}. Skipping.', file=sys.stderr)
            do_vis = False

    if do_vis:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        for orig_img in orig_images:
            fig, ax = plt.subplots(1)

            plt.axis('scaled')
            plt.xlim(0, 5000)
            plt.ylim(0, 710)
            plt.gca().invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            plt.xlabel('x')
            plt.ylabel('y')

            for loc in selected_locations[orig_img]['test']:
                ax.add_patch(Rectangle(
                    (float(loc[0]), float(loc[1])),
                    width=args.crop_size_x,
                    height=args.crop_size_y,
                    fill=False,
                    color='red',
                    label='test'
                ))
            for loc in selected_locations[orig_img]['validation']:
                ax.add_patch(Rectangle(
                    (float(loc[0]), float(loc[1])),
                    width=args.crop_size_x,
                    height=args.crop_size_y,
                    fill=False,
                    color='blue',
                    label='validation'
                ))
            fig.legend()
            fig.savefig(f'./vis/{orig_img}_selection.png')

# Since the partition is done, we can move all files into their respective subdirectory.
# This leaves the removed overlaps untouched. Fail if output directories already exist.
if not args.dry_run:
    for directory in ['training', 'validation', 'test']:
        try:
            os.makedirs(os.path.join(directory, 'images'))
            os.makedirs(os.path.join(directory, 'masks'))
        except IOError as e:
            sys.exit(e)

print('\nMoving files...')
counter = 0
for i, orig_img in enumerate(orig_images):
    for key in ANNOTATIONS:
        for f in file_mapping[orig_img][key]:
            if not args.dry_run:
                try:
                    shutil.move(
                        os.path.join(
                            vars(args)[key],
                            f
                        ),
                        os.path.join(
                            file_mapping[orig_img]['split'],
                            key
                        )
                    )
                except IOError as e:
                    print(f'ERROR: {e}', file=sys.stderr)
                    print('Moving files back...', file=sys.stderr)
                    # Clean up: move everything back
                    for ii in range(i):
                        for key in ANNOTATIONS:
                            for f in file_mapping[orig_images[ii]][key]:
                                try:
                                    shutil.move(
                                        os.path.join(
                                            file_mapping[orig_img]['split'],
                                            key,
                                            f
                                        ),
                                        os.path.join(
                                            vars(args)[key]
                                        )
                                    )
                                except IOError as e:
                                    sys.exit(
                                        f'ERROR while moving files back: {e}\nYou could try to recover by hand.'
                                    )
                    for directory in ['training', 'validation', 'test']:
                        shutil.rmtree(directory)
                    print('Done.', file=sys.stderr)
                    sys.exit(1)
            counter += 1
print(f'{counter} files moved.')

if args.dry_run:
    print()
    print('== This was a dry run, nothing was actually moved. ==')
