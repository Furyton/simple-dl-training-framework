import argparse
import json
from os import path

from configuration.options import args, parser
from scheduler.BasicSched import BasicScheduler
from scheduler.DistillSched import DistillScheduler
from scheduler.EnsembleSched import EnsembleScheduler
from utils import *

if __name__ == '__main__':
    with open(path.normpath(args.config_file), 'r') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        if args.kwargs is not None:
            t_args.__dict__.update(args.kwargs)
        args = parser.parse_args(namespace=t_args)

    export_root = setup_train(args)

    if args.sched.lower() == "basic":
        sched = BasicScheduler(args, export_root)
    elif args.sched.lower() == "ensemble":
        sched = EnsembleScheduler(args, export_root)
    elif args.sched.lower() == "distill":    
        sched = DistillScheduler(args, export_root)
    else:
        raise ValueError

    sched.run()
