import os
import shutil

from astropy.utils.data import download_file

__all__ = ["download_data"]

URL = "https://github.com/COINtoolbox/RESSPECT/raw/master/data/"


def download_data(filename, sub_path='', env_var='RESSPECT_TEST'):
    """Download file from the archive and store it in the local cache.

    Parameters
    ----------
    filename : str
        The filename, e.g. SIMGEN_PUBLIC_DES.tar.gz
    sub_path : str
        By default the file is stored at the root of the cache directory, but
        using ``path`` allows to specify a sub-directory.
    env_var: str
        Environment variable containing the path to the cache directory.

    Returns
    -------
    str
        Name of the cached file with the path added to it.
    """
    # Find cache path and make sure it exists
    root_cache_path = os.getenv(env_var)
    sub_path = os.path.dirname(filename) if sub_path is None else \
        os.path.join(sub_path, os.path.dirname(filename))

    if root_cache_path is None:
        raise ValueError('Environment variable not set: {:s}'.format(env_var))

    root_cache_path = os.path.expanduser(root_cache_path)

    cache_path = os.path.join(root_cache_path, sub_path)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # Now check if the local file exists and download if not
    local_path = os.path.join(cache_path, os.path.basename(filename))
    if not os.path.exists(local_path):
        tmp_path = download_file(URL + filename, cache=False)
        shutil.move(tmp_path, local_path)

        # `download_file` ignores Access Control List - fixing it
        os.chmod(local_path, 0o664)

    #

    return local_path
