"""Create an association from a list"""

import argparse
import sys

from .lib.rules_elpp_base import DMS_ELPP_Base
from .registry import AssociationRegistry

__all__ = ["asn_from_list"]


def asn_from_list(items, rule=DMS_ELPP_Base, **kwargs):
    """Create an association from a list

    Parameters
    ----------
    items: [object [, ...]]
        List of items to add.

    rule: `Association` rule
        The association rule to use.

    kwargs: dict
        Other named parameters required or pertinent to adding
        the items to the association.

    Returns
    -------
    association: `Association`-based instance
        The association with the items added.

    Notes
    -----
    This is a lower-level tool for artificially creating
    an association. As such, the association created may not be valid.
    It is presume the user knows what they are doing.
    """
    asn = rule()
    asn._add_items(items, **kwargs)
    if "target" in kwargs.keys():
        target = kwargs["target"]
        asn["target"] = target
    return asn


def _cli(args=None):
    """Command-line interface for list_to_asn

    Parameters
    ----------
    args: [str, ...], or None
        The command line arguments. Can be one of
            - `None`: `sys.argv` is then used.
            - `[str, ...]`: A list of strings which create the command line
              with the similar structure as `sys.argv`
    """

    if args is None:
        args = sys.argv[1:]
    if isinstance(args, str):
        args = args.split(" ")

    parser = argparse.ArgumentParser(
        description="Create an association from a list of files",
        usage="asn_from_list -o mosaic_asn.json\n--product-name my_mosaic *.asdf",
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="File to write association to",
    )

    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="json",
        help='Format of the association files. Default: "%(default)s"',
    )

    parser.add_argument(
        "--product-name",
        type=str,
        help="The product name when creating a Level 3 association",
    )

    parser.add_argument(
        "-r",
        "--rule",
        type=str,
        default="DMS_ELPP_Base",
        help=('The rule to base the association structure on. Default: "%(default)s"'),
    )
    parser.add_argument(
        "--ruledefs",
        action="append",
        help=(
            "Association rules definition file(s) If not specified, the default"
            " rules will be searched."
        ),
    )
    parser.add_argument(
        "-i",
        "--id",
        type=str,
        default="o999",
        help='The association candidate id to use. Default: "%(default)s"',
        dest="acid",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="None",
        help='The target name for the association. Default: "%(default)s"',
        dest="target",
    )

    parser.add_argument(
        "filelist",
        type=str,
        nargs="+",
        help="File list to include in the association",
    )

    parsed = parser.parse_args(args=args)
    print("Parsed args:", parsed)

    # Get the rule
    rule = AssociationRegistry(parsed.ruledefs, include_bases=True)[parsed.rule]

    with open(parsed.output_file, "w") as outfile:
        asn = asn_from_list(
            parsed.filelist,
            rule=rule,
            product_name=parsed.product_name,
            acid=parsed.acid,
            target=parsed.target,
        )
        _, serialized = asn.dump(format=parsed.format)
        outfile.write(serialized)
