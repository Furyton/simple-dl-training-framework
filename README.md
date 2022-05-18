# simple dl training framework

structure:

- asset:

    here you can put the hyper-params of the model, and there are some example files.

- config:

    example config file

- configuration:

    - `options.py`: a bunch of arguments

    - `config.py`: some macro vars, mainly used for creating log directories.

    - `utils.py`: some util functions

- dataloaders:

    - `MaskDataloader.py` and `NextItItemDataloader.py` are example dataloader creator used in recommender models.

- models:

    you can put your loss function in `loss.py`. make your own model inherent from `base.py`. `Ensembler.py` is a simple ensemble model that average the response of several models. Note the model should output a tuple like (output, loss, regularization loss, ...), so you can just sum up them in the trainer and do other thing about the output. 

- scheduler:

    - you can design your own training schedule. Currently only support linear stream. Basically you initialize the corresponding trainer and logger first, and invoke `Routine` to do the schedule.

- trainers:

    - you can put your loss function in `loss.py`. all trainers should inherent from `BaseTrainer.py`.
