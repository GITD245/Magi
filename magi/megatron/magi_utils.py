import argparse

def add_magi_args(parser):
    group = parser.add_argument_group(title="magi")

    group.add_argument("--magi", action="store_true")
    group.add_argument("--magi-profile-flag", action="store_true")

    # group.add_argument("--top-k", type=int, default=2)
    # group.add_argument("--balance-loss-weight", type=float, default=1)
    # group.add_argument("--balance-strategy", type=str,choices = ['noisy','gshard','switch','swipe','naive'], default=None)
    # group.add_argument("--hidden-hidden-size", type=int, default=None)

    return parser