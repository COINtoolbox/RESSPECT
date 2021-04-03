"""
    utils for fit_lightcurves methods
"""

import io
import tarfile
from typing import AnyStr
from typing import Tuple

import numpy as np
import pandas as pd

from resspect.snana_fits_to_pd import read_fits

SNPCC_LC_MAPPINGS = {
    "snii": ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25',
             '26', '27', '30', '31', '32', '33', '34', '35', '36', '37', '38',
             '39', '40', '41', '42', '43', '44'],
    "snibc": ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16',
              '18', '22', '23', '29', '45', '28']
}


def read_file(file_path: str) -> list:
    with open(file_path, "r") as file:
        lines = [line.split() for line in file.readlines()]
        return [line for line in lines if len(line) > 1]


def get_snpcc_sntype(value: str) -> str:
    if value in SNPCC_LC_MAPPINGS["snibc"]:
        return 'Ibc'
    elif value in SNPCC_LC_MAPPINGS["snii"]:
        return 'II'
    elif value == '0':
        return 'Ia'
    raise ValueError('Unknown supernova type!')


def read_tar_file(file_path: str) -> AnyStr:
    with tarfile.open(file_path, 'r:gz') as tar:
        tar_members = tar.getmembers()[0]
        return tar.extractfile(tar_members).read()


def read_resspect_full_photometry_data(file_path: str) -> pd.DataFrame:
    if '.tar.gz' in file_path:
        tar_content = read_tar_file(file_path)
        return pd.read_csv(io.BytesIO(tar_content))
    elif '.FITS' in file_path:
        _, full_photometry = read_fits(
            file_path, drop_separators=True)
        return full_photometry
    else:
        return pd.read_csv(file_path, index_col=False)


def get_photometry_with_id_name_and_snid(
        full_photometry: pd.DataFrame,
        id_names_list: list, snid: int) -> Tuple[pd.DataFrame, str]:
    snid_indices = pd.Series()
    id_name = None
    for each_id in id_names_list:
        if each_id in full_photometry.keys():
            snid_indices = full_photometry[each_id] == snid
            id_name = each_id
            break
    return full_photometry[snid_indices], id_name


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
    if '.tar.gz' in file_path:
        tar_content = read_tar_file(file_path)
        return pd.read_csv(io.BytesIO(tar_content))
    else:
        full_photometry = pd.read_csv(file_path, index_col=False)
        if ' ' in full_photometry.keys()[0]:
            full_photometry = pd.read_csv(file_path, sep=' ', index_col=False)
    return full_photometry


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
