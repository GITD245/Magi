import argparse

def add_magi_args(parser):
    group = parser.add_argument_group(title="magi_config")

    group.add_argument("--model", type=str, default="", help="record model name for log")
    group.add_argument("--magi-profile-flag", action="store_true", help="whether to profile the time of expert layer")

    group.add_argument("--magi-policy", type=int, default=3, help="magi police: 0-no policy; 1-ranking broadcast; 2-ranking double; 3-popularity;")
    group.add_argument("--magi-schedule-interval", type=int, default=10, help="schedule interval, setting 0 to enable dynamic schedule interval") # MAGI_TODO: default should be 0 to use dynamic interval
    group.add_argument("--magi-proxy-expert-ratio", type=float, default=0.5, help="the ratio of proxy expert numbers")
    group.add_argument("--magi-token-redirect-flag", type=bool, default=True, help="whether to redirect the tokens")
    
    group.add_argument("--janus", action="store_true", help="janus mode")
    group.add_argument("--fastermoe", action="store_true", help="fastermoe mode")

    return parser