# TextSnake and Ecotron2018

See also the [TextSnake](https://link.springer.com/chapter/10.1007/978-3-030-01216-8_2) publication.

## Installation and usage
The script runs out of the box, no setup required:
```bash
$ python partition.py --help
usage: partition.py [-h] --images DIR --roots DIR --centerlines DIR --radii DIR --sin DIR --cos DIR --crop-width INT --val-split
                    INT --test-split INT [-y] [--dry-run] [--vis] [-r INT]

Partition the Ecotron-EInsect-2018 dataset into training, validation, and testing sets.

optional arguments:
  -h, --help            show this help message and exit
  -y, --yes             Assume answer "yes" for all questions
  --dry-run             Only simulate the process, don't actually touch anything
  --vis                 Create visualization of selection in ./vis
  -r INT, --random-seed INT
                        Seed for the random number generator

Data input:
  --images DIR          directory with images
  --roots DIR           directory with root masks for given images
  --centerlines DIR     directory with center line masks for given images
  --radii DIR           directory with radii maps for given images
  --sin DIR             directory with sine maps for given images
  --cos DIR             directory with cosine maps for given images
  --crop-width INT      crop width

Split control:
  --val-split INT       percentage of data going into validation set
  --test-split INT      percentage of data going into test set

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
```
