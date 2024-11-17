import argparse
import os


def set_version(new_vesrion: str):

    version_file = os.path.join("VERSION.txt")

    with open(version_file, "w") as ver_file:
        ver_file.write(new_vesrion)
        ver_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=str, help="New Version String")
    args = parser.parse_args()

    if args.version is None:
        print("Error: Missing Argument! Try again.")
    else:
        print(f"Version string: {args.version}")
        set_version(args.version)
