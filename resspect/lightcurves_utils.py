"""
    utils for fit_lightcurves methods
"""

import io
import tarfile
from typing import AnyStr
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from resspect.snana_fits_to_pd import read_fits

SNPCC_LC_MAPPINGS = {
    "snii": {'2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25',
             '26', '27', '30', '31', '32', '33', '34', '35', '36', '37', '38',
             '39', '40', '41', '42', '43', '44'},
    "snibc": {'1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16',
              '18', '22', '23', '29', '45', '28'}
}


def read_file(file_path: str) -> list:
    with open(file_path, "r") as file:
        lines = [line.split() for line in file.readlines()]
        return list(filter(lambda x: len(x) > 1, lines))


def get_snpcc_sntype(value: str) -> str:
    if value in SNPCC_LC_MAPPINGS["snibc"]:
        return 'Ibc'
    if value in SNPCC_LC_MAPPINGS["snii"]:
        return 'II'
    if value == '0':
        return 'Ia'
    raise ValueError('Unknown SNPCC supernova type!')


def read_tar_file(file_path: str) -> AnyStr:
    with tarfile.open(file_path, 'r:gz') as tar:
        tar_members = tar.getmembers()[0]
        return tar.extractfile(tar_members).read()


def read_resspect_full_photometry_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.tar.gz'):
        tar_content = read_tar_file(file_path)
        return pd.read_csv(io.BytesIO(tar_content))
    if file_path.endswith('.FITS'):
        _, full_photometry = read_fits(
            file_path, drop_separators=True)
        return full_photometry
    if file_path.endswith(('.csv', '.csv.gz')):
        return pd.read_csv(file_path, index_col=False)
    raise ValueError(f"Unknown RESSPECT photometry data file: {file_path}")


def get_photometry_with_id_name_and_snid(
        full_photometry: pd.DataFrame,
        id_names_list: list, snid: int) -> Tuple[
        pd.DataFrame, Union[str, None]]:
    """
    This function loads photometry data of the given SNID.
    The full_photometry DataFrame should contain one the column name passed in
    id_names_list. Otherwise the function returns empty dataframe and none snid
    name

    Parameters
    ----------
    full_photometry
        photometry DataFrame
    id_names_list
        list of available SNID column names
    snid
        SNID
    """
    for snid_column_name in id_names_list:
        if snid_column_name in full_photometry.keys():
            snid_indices = full_photometry[snid_column_name] == snid
            return full_photometry[snid_indices], snid_column_name
    return pd.DataFrame(), None


def _update_resspect_filter_values(
        filters_array: np.ndarray, filters: list) -> np.ndarray:
    updated_band = np.zeros_like(filters_array)
    for each_filter in filters:
        first_case = "b'" + each_filter + " '"
        second_case = "b'" + each_filter + "'"
        third_case = "b'" + each_filter + "' "
        updated_band[(filters_array == first_case) |
                     (filters_array == second_case) |
                     (filters_array == third_case)] = each_filter
    return updated_band


def insert_band_column_to_resspect_df(
        photometry_df: pd.DataFrame, filters: list) -> pd.DataFrame:
    if 'b' in str(photometry_df['FLT'].values[0]):
        updated_band = _update_resspect_filter_values(
            photometry_df['FLT'].values, filters)
        photometry_df.insert(1, 'band', updated_band, True)
    else:
        photometry_df.insert(1, 'band', photometry_df['FLT'].values, True)
    return photometry_df


def load_resspect_photometry_df(photometry_df: pd.DataFrame) -> pd.DataFrame:
    photometry_dict = {
        'mjd': photometry_df['MJD'].values,
        'band': photometry_df['band'].values,
        'flux': photometry_df['FLUXCAL'].values,
        'fluxerr': photometry_df['FLUXCALERR'].values
    }
    if 'SNR' in photometry_df.keys():
        photometry_dict['SNR'] = photometry_df['SNR'].values
    else:
        photometry_dict['SNR'] = (photometry_dict['flux'] /
                                  photometry_dict['fluxerr'])
    return pd.DataFrame(photometry_dict)


def read_plasticc_full_photometry_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.tar.gz'):
        tar_content = read_tar_file(file_path)
        return pd.read_csv(io.BytesIO(tar_content))
    if file_path.endswith(('.csv', '.csv.gz')):
        full_photometry = pd.read_csv(file_path, index_col=False)
        if ' ' in full_photometry.keys()[0]:
            full_photometry = pd.read_csv(file_path, sep=' ', index_col=False)
        return full_photometry
    raise ValueError(f"Unknown PLAsTiCC photometry data file: {file_path}")


def _update_plasticc_filter_values(
        filters_array: np.ndarray, mapping_dict: dict) -> np.ndarray:
    updated_filters_array = np.zeros_like(filters_array, dtype=object)
    for key, value in mapping_dict.items():
        updated_filters_array[filters_array == key] = value
    return updated_filters_array


def load_plasticc_photometry_df(
        photometry_df: pd.DataFrame, filter_mapping_dict) -> pd.DataFrame:
    photometry_dict = {
        'mjd': photometry_df['mjd'].values,
        'band': _update_plasticc_filter_values(
            photometry_df['passband'].values, filter_mapping_dict),
        'flux': photometry_df['flux'].values,
        'fluxerr': photometry_df['flux_err'].values,
        'detected_bool': photometry_df['detected_bool'].values
    }
    return pd.DataFrame(photometry_dict)


def load_snpcc_photometry_df(
        photometry_raw: np.ndarray, header: list) -> pd.DataFrame:
    photometry_dict = {
        'mjd': np.array(
            photometry_raw[:, header.index('MJD')]).astype(np.float),
        'band': np.array(
            photometry_raw[:, header.index('FLT')]),
        'flux': np.array(
            photometry_raw[:, header.index('FLUXCAL')]).astype(np.float),
        'fluxerr': np.array(
            photometry_raw[:, header.index('FLUXCALERR')]).astype(np.float),
        'SNR': np.array(
            photometry_raw[:, header.index('SNR')]).astype(np.float),
        'MAG': np.array(
            photometry_raw[:, header.index('MAG')]).astype(np.float),
        'MAGERR': np.array(
            photometry_raw[:, header.index('MAGERR')]).astype(np.float)
    }
    return pd.DataFrame(photometry_dict)
