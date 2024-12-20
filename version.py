import argparse
import logging
import os

from pypi_simple import PyPISimple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_version(new_version: str, rc_number=None) -> None:

    version_file = os.path.join("VERSION.txt")

    if new_version[0] == "v":
        new_version = new_version[1:]

    if rc_number:
        new_version = f"{new_version}-rc{rc_number}"

    with open(version_file, "w") as ver_file:
        ver_file.write(new_version)
        ver_file.close()


def compute_release_candidate(is_test_pypi=False) -> int:

    pypi_url = "https://test.pypi.org/simple" if is_test_pypi else "https://pypi.org/simple"
    rc_number = 1

    with PyPISimple(pypi_url) as client:

        requests_page = client.get_project_page("deepnado")

        # Get most recent last package
        pkg = requests_page.packages[-1]

        # version
        version = pkg.version

        if "rc" in version:
            logger.info("Found a prior release candidate.")
            last_candidate_ver = version.split("rc")[-1]
            rc_number = int(last_candidate_ver) + 1
            print(rc_number)

    return rc_number


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, help="New Version String")
    parser.add_argument("-c", "--compute", action="store_true", help="Compute RC")

    args = parser.parse_args()
    rc_number = None

    if args.version is None:
        logger.error("Error: Missing Argument! Try again.")

    if args.compute:
        rc_number = compute_release_candidate(is_test_pypi=True)

    logger.info(f"Version string: {args.version}")
    set_version(args.version, rc_number=rc_number)
