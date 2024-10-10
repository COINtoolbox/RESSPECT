import pooch
from pathlib import Path

# Create a Pooch object to handle the data fetching
POOCH = pooch.create(
    path=pooch.os_cache("resspect"),
    base_url="doi:10.5281/zenodo.13883296",
    registry={
        "SIMGEN_PUBLIC_DES.tar.gz": "md5:3ed2c475512d8c8cf8a2b8907ed21ed0",
    },
)


def fetch_example_data():
    """Use Pooch to fetch the example data, unpack it, and create a symlink to
    the data directory in the project's data folder.
    """

    unpacked_files = POOCH.fetch(
        "SIMGEN_PUBLIC_DES.tar.gz",
        processor=pooch.Untar(extract_dir="."),
    )

    # Get the parent directory of the unpacked files.
    # Note that index 0 is the tar file itself.
    original_directory = Path(unpacked_files[1]).resolve().parent
    print(f"Data unpacked into directory: {original_directory}")


if __name__ == "__main__":
    fetch_example_data()
