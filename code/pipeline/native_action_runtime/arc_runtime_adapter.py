"""Only replace formatting with frozen stage v2; leave generation/fuser intact."""
import sys
import protocol_min
import legacy_methods
_paths=sys.path[:]
import arc_protocol
sys.path[:]=_paths

def format_arc(example, use_template=True):
    return arc_protocol.receiver_prompt(example) if use_template else arc_protocol.helper_body(example)

protocol_min.format_openbook=format_arc
legacy_methods.format_openbook=format_arc
