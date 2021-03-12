"""
Partition the Ecotron-EInsect-2018 dataset into training, validation, and
testing sets.

The set contains only a few minirhizotron images. This script also expects for
each image five feature maps and masks defining the input for the TextSnake
neural net. The maps and masks can be generated with
https://gitlab.informatik.uni-halle.de/moeller/minirhizotron_annotation.

Because the minirhizotron images are very big and wide (about 5000x700 pixels),
they (and of course all feature maps and masks) are cropped into smaller parts
before they are fed into the neural net.

This script partitions these crops into training, validation, and testing set.
But this is tricky because usually these crops overlap in both x and y
direction. We make sure that no data goes into more than one set. Because of
the overlaps, there is no other possibility than simply ignoring some of the
crops, i.e. do not put them in any set. The sets are moved into
subdirectories "training", "test", and "validation".

We want parts of every image in all splits. The idea is to divide an image into
three not-overlapping patches, one for each subset. The size of the validation
and test patch is configurable via the --val-split and --test-split arguments,
which makes the size of the training patch 1 minus the sum of these arguments.
Crops at the patch boundaries belonging to two patches are excluded.

This script is designed to work with the output of dl_cropImages.sh. However, it
should work with every input files as long as the file names match this format:
    <original image file name>-<arbitrary string>+<position x>+<position y>.<arbitrary file type>
where <original image file name> is used as hint for which crops belong to the
same image. <position x> and <position y> denote the crop's upper left corner in
its original image's coordinate system. Precisely, the names must match this
regular expression:
    .*-.*\+[0-9]+\+[0-9]+\.[a-zA-Z0-9_]+$
An example:
    EInsect_T017_CTS_09.05.18_000000_1_HMC-roots-00510x00510+00000+00000.tif
However, it can be easily adapted to other file names by changing the
VALID_FILE_NAME variable and get_metadata_from_filename function.
"""

import argparse
import os
import pathlib
import random
import re
import sys
from collections import namedtuple


def get_metadata_from_filename(file_name: str) -> namedtuple:
    """
    Extract metadata like original image name and crop position from the
    given file name. Change this function to use a different file name pattern.
    """
    if os.path.isabs(f):
        file_name = os.path.basename(file_name)
    original_image_name = file_name.split('-')[0]
    x_pos = int(file_name.split('.')[-2].split('+')[-2:][0])
    Metadata = namedtuple('Metadata', ['original_image_name', 'x_pos'])
    return Metadata(original_image_name, x_pos)


def visualize_selection(image_list, selected_locations):
    # First, detect if we're running in a gui-less environment. In this case,
    # we have to choose an appropriate backend and can only save the plots to
    # file.
    import matplotlib
    gui_present = True
    if not os.environ.get('DISPLAY', None):
        # Either on Windows or no GUI. Try to plot with current interpreter. If
        # this fails, we have no gui.
        probe = "import matplotlib.pyplot as plt; import sys\ntry: plt.figure()\nexcept: sys.exit(1)"
        cmd = f'{sys.executable} -c "{probe}"'
        if os.system(cmd) != 0:
            # No GUI. Use backend "agg" for PNG files
            matplotlib.use('agg')
            gui_present = False

    # Normally, in a dry run we would show the plots instead of saving them.
    # But if we have no gui, save them nonetheless.
    if args.dry_run and not gui_present:
        print(f"Running in a gui-less environment. Saving visualization to ./vis.")

    # Save the figures if a) this is no dry run, or b) we have no gui
    do_save_figs = not args.dry_run or not gui_present

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    if do_save_figs:
        if not os.path.isdir('./vis'):
            try:
                os.mkdir('./vis')
            except IOError as e:
                print(f'\nCould not create visualization directory: {e}. Skipping.', file=sys.stderr)
                return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch

    for img in image_list.keys():
        # todo get image width and height
        img_height = 710

        fig, ax = plt.subplots(1)

        # FIXME
        # plt.title(img)

        plt.axis('scaled')
        plt.xlim(0, 5000)
        plt.ylim(0, img_height)
        plt.gca().invert_yaxis()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.xlabel('x')
        plt.ylabel('y')

        for x in selected_locations[img]['test']:
            ax.add_patch(Rectangle(
                (x, 0),
                width=args.crop_width,
                height=img_height,
                fill=True,
                color='orange',
                label='test'
            ))
        for x in selected_locations[img]['validation']:
            ax.add_patch(Rectangle(
                (x, 0),
                width=args.crop_width,
                height=img_height,
                fill=True,
                color='blue',
                label='validation'
            ))
        for x in selected_locations[img]['training']:
            ax.add_patch(Rectangle(
                (x, 0),
                width=args.crop_width,
                height=img_height,
                fill=True,
                color='green',
                label='training'
            ))
        fig.legend(
            handles=[
                Patch(color='green', label='Training'),
                Patch(color='blue', label='Validation'),
                Patch(color='orange', label='Test')
            ],
            labels=['Training', 'Validation', 'Test']
        )

        if not do_save_figs:
            plt.show()
        else:
            try:
                fig.savefig(f'./vis/{img}_selection.png')
            except IOError as e:
                print(f'\nCould not save visualization: {e}. Skipping.', file=sys.stderr)
                return

    if do_save_figs:
        print('\nSaved visualization in ./vis')


class Image:
    """
    Represents a complete minirhizotron image. An instance consists of several
    columns of possibly vertically and horizontally overlapping crops. A
    column holds all crops with the same unique position on the horizontal (x)
    axis. A column contains crops of all feature maps and masks, so conceptually
    an instance of this class is a multi-channel image composed of several
    one-channel images of the same scene.
    """

    def __init__(self, name, col_width, valid_file_name_regex):
        self.name = name
        self.col_width = col_width
        self.n_cols = 0

        self._valid_file_name_regex = valid_file_name_regex

        # We map each x position to its column for easy access. We also keep a
        # sorted index for iteration in correct order.
        self._cols = {}
        self._x_positions = []

        self._moved_cols = []

    def insert(self, file_path: str, annot_type: str) -> None:
        """
        Insert the crop represented by file_name into this image.
        :raises ValueError: if file_path has an illegal format, see __doc__
        """
        if self._valid_file_name_regex.match(os.path.basename(file_path)) is None:
            raise ValueError(f'Illegal file name: {os.path.basename(file_path)}')
        x_pos = get_metadata_from_filename(file_path).x_pos
        if x_pos in self._x_positions:
            col = self._cols[x_pos]
        else:
            col = Column()
            self._x_positions.append(x_pos)
            self._x_positions.sort()
        col.insert(Crop(file_path, annot_type))
        self._cols[x_pos] = col

        self.n_cols = len(self._cols)

    # def validate_columns(self) -> bool:
    #     """
    #     Check if all columns are valid (see Column.validate).
    #     """
    #     for col in self._cols.values():
    #         if not all(col.validate()):
    #             return False
    #     return True
    #
    # def get_invalid_columns(self) -> {int: [str]}:
    #     """
    #     Return invalid columns as a dict of pairs (xpos -> [reasons]).
    #     :rtype: {int: [str]}
    #     """
    #     invalid_columns = {}
    #     for xpos, col in self._cols:
    #         all_annotations_present, file_counts_do_match = col.validate()
    #         if not all_annotations_present:
    #             invalid_columns[xpos] = ['reason']
    #         if not file_counts_do_match:
    #             invalid_columns[xpos].append('reason')
    #     return invalid_columns
    #
    # def validate_file_count(self):
    #     pass
    #
    # def get_file_counts(self):
    #     pass
    #
    # def validate_horizontal_alignment(self):
    #     pass
    #
    # def get_horizontal_alignment(self):
    #     pass
    #
    # def validate(self) -> (bool, bool, bool):
    #     """
    #     Validate this image. Image is valid if
    #         - all columns are valid
    #         - all columns have the same number of files
    #         - columns can be aligned without gaps using their knwon horizontal position and width
    #     :return: (bool, bool, bool):
    #     """
    #     columns_are_valid = True
    #     for x_pos, col in self._cols.items():
    #         if not col.validate(): pass
    #
    #     sorted_xpos = sorted(self._cols.keys())
    #     columns_do_line_up = True
    #     for i in range(len(sorted_xpos) - 1):
    #         columns_do_line_up = sorted_xpos[i] == sorted_xpos[i+1] + self.col_width
    #
    #     return False

    def select_randomly(self, val_split: float, test_split: float) -> {str: int}:
        """
        Randomly divide this image into training, validation and test split.

        The image is divided into three randomly ordered consecutive patches.
        The fractions of the image going into each patch are given by
        val_split, test_split, and 1 - (val_split + test_split), respectively.
        Columns in which the patches overlap are removed.

        :return dict with number of selected crops (keys 'training',
            'validation', 'test', 'ignore')
        """

        def _select(start, n, label) -> int:
            """
            Label all columns in [start, start+n) with label.
            """
            n_selected = 0
            interval = [i % self.n_cols for i in range(start, start + n)]
            for i in interval:
                x = self._x_positions[i]
                n_selected += self._cols[x].mark_as(label)
            return n_selected

        def _remove_overlaps(start, end) -> int:
            """
            Remove unlabelled columns in [start-col_width, end+col_width].
            """
            start = self._x_positions[start % self.n_cols]
            end = self._x_positions[end % self.n_cols]
            n_removed = 0
            for x, col in self._cols.items():
                if start - self.col_width <= x <= start or end <= x <= end + self.col_width:
                    if col.label is None:
                        n_removed += col.mark_as('ignore')
            return n_removed

        def _next_unlabelled_col(x):
            """
            Return index of first unlabelled column after x.
            """
            for i in range(self.n_cols):
                idx = (x + i) % self.n_cols
                x_current = self._x_positions[idx]
                if self._cols[x_current].label is None:
                    return idx

        # When computing number of columns per split we must take into account
        # that some columns will be removed, i.e. we want to compute the split
        # sizes as fraction of the number of actual selected columns, not of
        # the total number of columns.
        delta_x = self._x_positions[1] - self._x_positions[0]
        removed_per_split = self.col_width / delta_x
        # * 2 because 2 gaps between 3 splits
        n_val = round((self.n_cols - removed_per_split * 2) * val_split)
        n_test = round((self.n_cols - removed_per_split * 2) * test_split)
        n_train = round((self.n_cols - removed_per_split * 2) * (1 - val_split - test_split))

        # Start splitting with a random column
        #start = random.randint(0, len(self._cols) -1)
        start = 0

        stats = dict.fromkeys(['training', 'validation', 'test', 'ignore'], 0)

        # Place patches in arbitrary order
        for n, label in random.sample(list(zip([n_train, n_val, n_test], ['training', 'validation', 'test'])), k=3):
            # Mark patch
            stats[label] += _select(start, n, label)
            # Remove columns overlapping this patch
            stats['ignore'] += _remove_overlaps(start, start + n - 1)
            # Next patch starts at next unlabelled column
            start = _next_unlabelled_col(start)

        return stats

    def get_labels(self) -> {int: str}:
        """
        Return a mapping of column positions and labels.
        """
        return {x: col.label for x, col in self._cols.items()}

    def to_disk(self, dry_run: bool) -> int:
        """
        Move all files of this image to the output directories defined by each
        column's label. Returns number of files moved.
        :param dry_run: Do not actually move files.
        :raises IOError: if an error occurs while moving the files
        """
        file_counter = 0
        for k, col in self._cols.items():
            self._moved_cols.append(k)
            file_counter += col.move(dry_run=dry_run)
        return file_counter

    def rollback(self) -> None:
        """
        Undo all former file movements.
        :raises IOError: if an error occurs while moving the files
        """
        for k in self._moved_cols:
            self._cols[k].move_back()


class Crop:
    """
    A crop is a portion of an image. The crop resides as actual image file
    somewhere on disk.
    """

    def __init__(self, file_path, annot_type):
        self._file_path = file_path
        self.annot_type = annot_type
        self._file_was_moved = False
        self._new_path = None

    def move_to(self, path: str) -> None:
        """
        Move the file associated with this crop to the directory
        path/annot_type, where annot_type is this crop's annotation type.
        :raises IOError: if an error occurs while moving the file
        """
        self._new_path = os.path.join(path, self.annot_type, os.path.basename(self._file_path))
        os.rename(self._file_path, self._new_path)
        self._file_was_moved = True

    def move_back(self) -> None:
        """
        Undo a former file movement by moving the file back to its origin.
        :raises IOError: if an error occurs while moving the file
        """
        if self._file_was_moved:
            os.rename(self._new_path, self._file_path)
            pass


class Column:
    """
    A Column is a collection of vertically aligned crops, which can also overlap
    vertically. It is a collection of all crops with the same horizontal
    position from the actual minirhizotron image as well as from its feature
    maps and masks.
    """
    def __init__(self):
        self._content = []
        self._file_counts = {}
        self.label = None

    def insert(self, item: Crop) -> None:
        """
        Insert the Crop into this column.
        """
        self._content.append(item)
        self._file_counts[item.annot_type] = self._file_counts.get(item.annot_type, 0) + 1

    def mark_as(self, label: str) -> int:
        """
        Mark this column with the provided label. Returns number of labelled
        crops.
        """
        self.label = label
        return len(self._content) // len(ANNOTATIONS)

    def move(self, dry_run: bool) -> int:
        """
        Move all files of this column to the corresponding directory, if this
        column is not labeled to be ignored. Returns number of files moved.
        :param dry_run: Do not actually move files.
        :raises IOError: if an error occurs while moving the files
        """
        if self.label == 'ignore':
            return 0

        file_counter = 0
        for crop in self._content:
            if not dry_run:
                crop.move_to(self.label)
            file_counter += 1

        return file_counter

    # def validate(self) -> (bool, bool):
    #     """
    #     Check if this column is complete, i.e. all types of annotations (see ANNOTATIONS) are present, and each have the
    #     same number of files registered.
    #     :return: (bool, bool): whether all annotation types are present, and whether all hold the same number of files
    #     """
    #     all_annotations_present = all(a in list(self._file_counts.keys()) for a in ANNOTATIONS)
    #     file_counts_do_match = all(n == list(self._file_counts.values())[0] for n in self._file_counts.values())
    #     return all_annotations_present, file_counts_do_match

    def move_back(self) -> None:
        """
        Undo all former file movements.
        :raises IOError: if an error occurs while moving the files
        """
        if self.label == 'ignore':
            return

        for crop in self._content:
            crop.move_back()


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
        '--roots',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with root masks for given images'
    )
    data_group.add_argument(
        '--centerlines',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with center line masks for given images'
    )
    data_group.add_argument(
        '--radii',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with radii maps for given images'
    )
    data_group.add_argument(
        '--sin',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with sine maps for given images'
    )
    data_group.add_argument(
        '--cos',
        type=pathlib.Path,
        required=True,
        metavar='DIR',
        help='directory with cosine maps for given images'
    )
    data_group.add_argument(
        '--crop-width',
        type=int,
        required=True,
        metavar='INT',
        help='crop width'
    )

    split_group = parser.add_argument_group('Split control')
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
    parser.add_argument(
        '-r', '--random-seed',
        metavar='INT',
        type=int,
        help='Seed for the random number generator'
    )

    return parser


# This is used to verify all file names encountered in the given input
# directories. Change the pattern to match different file names.
VALID_FILE_NAME = re.compile(r'.*-.*\+[0-9]+\+[0-9]+\.[a-zA-Z0-9_]+$')

ANNOTATIONS = ['images', 'roots', 'centerlines', 'radii', 'sin', 'cos']

args_parser = get_args_parser()
args = args_parser.parse_args()

if args.val_split + args.test_split > 100:
    sys.exit(f'ERROR: Requested splits to large (expected sum<100): validation: {args.val_split}, test:{args.test_split}')

if args.random_seed:
    random.seed(args.random_seed)

if args.dry_run:
    print('\n=== This is a dry run, no files will be moved or changed! ===\n')

# Create a list of Image instances, each holding all its crops. The Images
# provide methods to split each individually into training, test, and
# validation set.
print('Listing files and assembling images...\n')
image_list = {}

# only for sanity check
file_list_lengths = []

for annot_type in ANNOTATIONS:
    try:
        # resolve() to absolut file path to easily keep track of
        # where each file came from. We need this in case of
        # rollback caused by an error while moving files at the end
        # of this script.
        with os.scandir(vars(args)[annot_type].resolve()) as file_list:
            file_list = list(file_list)
            file_list_lengths.append(len(file_list))
            print(f'  {vars(args)[annot_type]}: {len(file_list)} files')

            # Image assembly
            for f in file_list:
                f = f.path  # f is os.DirEntry
                original_image_name = get_metadata_from_filename(file_name=f).original_image_name
                img = image_list.get(
                    original_image_name,
                    Image(
                        name=original_image_name,
                        col_width=args.crop_width,
                        valid_file_name_regex=VALID_FILE_NAME
                    )
                )
                try:
                    img.insert(file_path=f, annot_type=annot_type)
                except ValueError as e:
                    # If a file name does not match the required format,
                    # something is probably wrong with our input. We better
                    # stop.
                    sys.exit(f'ERROR: {e}. Abort.')
                image_list[original_image_name] = img
    except IOError as e:
        sys.exit(f'ERROR: Could not load images: {e}')

if not all(n == file_list_lengths[0] for n in file_list_lengths):
    sys.exit('ERROR: No. of files do not match!')

# print('Validating images...')
# everything_valid = True
# for name, img in image_list.items():
#     print(f'  {name}... ', end='')
#     columns_are_valid = img.validate_columns()
#     file_count_is_valid = img.validate_file_count()
#     horizontal_alignment_is_valid = img.validate_horizontal_alignment()
#     if columns_are_valid and file_count_is_valid and horizontal_alignment_is_valid:
#         print('ok')
#     else:
#         print('NOT OK')
#         everything_valid = False
#         if not columns_are_valid:
#             for col, reason in img.get_invalid_columns():
#                 print(f'  Column {col}: {reason}')
#         if not file_count_is_valid:
#             print(f'  Number of annotation files differ:')
#             for col, count in img.get_file_counts():
#                 print(f'    Column {col}: {count} files')
#         if not horizontal_alignment_is_valid:
#             print(f'  Columns do not horizontally align (width: {img.col_width}):')
#             for col, xpos in img.get_horizontal_alignment():
#                 print(f'    Column {col} starts at {xpos}')
#
# if not everything_valid:
#     sys.exit(
#         f'ERROR: Could not assemble images properly! Are files missing?'
#         f'Is the crop width {args.crop_width} correct?'
#     )

print(
    f'\n  Found {len(image_list)} image(s):',
    '    ' + '\n '.join(list(image_list.keys())),
    '\nPartitioning:',
    f'  - training set:    {100 - args.val_split - args.test_split} %',
    f'  - validation set:  {args.val_split} %',
    f'  - test set:        {args.test_split} %\n',
    f'Directories for splits will be created in: {os.getcwd()}\n'
    'Ok? yes/no: ',
    sep='\n',
    end=''
)

if not args.yes:
    if input() != 'yes':
        sys.exit()
else:
    print('yes')

# Prepare output directories.
if not args.dry_run:
    for directory in ['training', 'validation', 'test']:
        if not os.path.isdir(directory):
            for dd in ANNOTATIONS:
                try:
                    os.makedirs(os.path.join(directory, dd))
                except IOError as e:
                    sys.exit(e)
        elif os.listdir(directory) is not []:
            sys.exit(f'Directory ./{directory} exists and is not empty! Abort.')

val_split = args.val_split / 100
test_split = args.test_split / 100

print('\nSplitting into sets...')

# Keep track of selected crop locations for visualisation
selected_locations = {
    orig_img: {
        'training': [],
        'test': [],
        'validation': [],
        'ignore': []
    } for orig_img in image_list.keys()}

stats = dict.fromkeys(['training', 'validation', 'test', 'ignore'], 0)
for img_name, img in image_list.items():
    current_stats = img.select_randomly(val_split=val_split, test_split=test_split)
    for k in stats.keys():
        stats[k] += current_stats[k]
    labels = img.get_labels()
    for x, label in labels.items():
        selected_locations[img_name][label].append(x)

n_crops_total = sum(stats.values())
n_crops_selected = n_crops_total - stats['ignore']
print(
    f'\n  Crops total:        {n_crops_total}',
    f'  Removed because of overlaps: {stats["ignore"]}',
    f'\n  Partitioning:',
    f'    - training set:   {stats["training"]}/{n_crops_selected} crops '
    f'({round(stats["training"]/n_crops_selected * 100, 1)} %)',
    f'    - validation set: {stats["validation"]}/{n_crops_selected} crops '
    f'({round(stats["validation"] / n_crops_selected * 100, 1)} %)',
    f'    - test set:       {stats["test"]}/{n_crops_selected} crops '
    f'({round(stats["test"] / n_crops_selected * 100, 1)} %)',
    '\nShould I move the files now? yes/no: ',
    sep='\n',
    end=''
)

if not args.yes:
    if input() != 'yes':
        sys.exit()
else:
    print('yes')

# Since the partition is done, we can move all files into their respective
# subdirectory. This leaves the removed overlaps untouched.
print(f'\nMoving files (each crop contains {len(ANNOTATIONS)} files)...')
file_counter = 0
for img in image_list.values():
    try:
        file_counter += img.to_disk(dry_run=args.dry_run)
    except IOError as e:
        for img in image_list.values():
            try:
                img.rollback()
            except IOError as e:
                sys.exit('ERROR: {e}')
print(f'{file_counter} files moved.')
print(
    'This can be undone with (in Bash):',
    '  declare -A dirs',
    f'  dirs=([images]="{args.images}" [roots]="{args.roots}" '
    f'[centerlines]="{args.centerlines}" [radii]="{args.radii}" '
    f'[sin]="{args.sin}" [cos]="{args.cos}")',
    '  for split in "training" "test" "validation"; do',
    '    for d in ${!dirs[*]}; do',
    '      mv $split/$d/* "${dirs[$d]}" && rmdir "$split/$d/"',
    '    done && rmdir $split',
    '  done',
    sep='\n'
)

# Visualize selection if requested.
if args.vis:
    visualize_selection(image_list, selected_locations)

if args.dry_run:
    print('\n== This was a dry run, nothing was actually moved. ==')
