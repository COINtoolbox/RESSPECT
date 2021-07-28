"""
    utils for fit_lightcurves methods
"""

import io
import os
import tarfile
from typing import AnyStr
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from resspect.snana_fits_to_pd import read_fits


BAZIN_HEADERS = {
    'plasticc_header': [
        'id', 'redshift', 'type', 'code', 'sample', 'queryable', 'last_rmag',
        'uA', 'uB', 'ut0', 'utfall', 'utrise', 'gA', 'gB', 'gt0', 'gtfall',
        'gtrise', 'rA', 'rB', 'rt0', 'rtfall', 'rtrise', 'iA', 'iB', 'it0',
        'itfall', 'itrise', 'zA', 'zB', 'zt0', 'ztfall', 'ztrise', 'YA', 'YB',
        'Yt0', 'Ytfall', 'Ytrise'],
    'snpcc_header': [
        'id', 'redshift', 'type', 'code', 'orig_sample', 'queryable',
        'last_rmag', 'gA', 'gB', 'gt0', 'gtfall', 'gtrise', 'rA', 'rB',
        'rt0', 'rtfall', 'rtrise', 'iA', 'iB', 'it0', 'itfall', 'itrise',
        'zA', 'zB', 'zt0', 'ztfall', 'ztrise'],
    'header_with_cost': [
        'id', 'redshift', 'type', 'code', 'orig_sample', 'queryable',
        'last_rmag', 'cost_4m', 'cost_8m', 'gA', 'gB', 'gt0', 'gtfall',
        'gtrise', 'rA', 'rB', 'rt0', 'rtfall', 'rtrise', 'iA', 'iB', 'it0',
        'itfall', 'itrise', 'zA', 'zB', 'zt0', 'ztfall', 'ztrise']
}


SNPCC_LC_MAPPINGS = {
    "snii": {2, 3, 4, 12, 15, 17, 19, 20, 21, 24, 25,
             26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38,
             39, 40, 41, 42, 43, 44},
    "snibc": {1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16,
              18, 22, 23, 29, 45, 28}
}

SNPCC_FEATURES_HEADER = [
    'id', 'redshift', 'type', 'code', 'orig_sample',
    'gA', 'gB', 'gt0', 'gtfall', 'gtrise', 'rA', 'rB',
    'rt0', 'rtfall', 'rtrise', 'iA', 'iB', 'it0', 'itfall',
    'itrise', 'zA', 'zB', 'zt0', 'ztfall', 'ztrise'
]

PLASTICC_RESSPECT_FEATURES_HEADER = [
    'id', 'redshift', 'type', 'code', 'orig_sample', 'uA', 'uB', 'ut0',
    'utfall', 'utrise', 'gA', 'gB', 'gt0', 'gtfall','gtrise', 'rA', 'rB',
    'rt0', 'rtfall', 'rtrise', 'iA', 'iB', 'it0', 'itfall', 'itrise', 'zA',
    'zB', 'zt0', 'ztfall', 'ztrise', 'YA', 'YB', 'Yt0', 'Ytfall', 'Ytrise'
]

PLASTICC_TARGET_TYPES = {
    90: 'Ia', 67: '91bg', 52: 'Iax', 42: 'II', 62: 'Ibc',
    95: 'SLSN', 15: 'TDE', 64: 'KN', 88: 'AGN', 92: 'RRL', 65: 'M-dwarf',
    16: 'EB', 53: 'Mira', 6: 'MicroL', 991: 'MicroLB', 992: 'ILOT',
    993: 'CART', 994: 'PISN', 995: 'MLString'
}


def read_file(file_path: str) -> list:
    """
     This function reads input file and filters empty entries

     Parameters
     ----------
     file_path
         input file path
     """
    with open(file_path, "r") as input_file:
        lines = [line.split() for line in input_file.readlines()]
        return list(filter(lambda x: len(x) > 1, lines))


def get_snpcc_sntype(value: int) -> str:
    """
     This function returns SNPCC class using SNPCC_LC_MAPPINGS

     Parameters
     ----------
     value
         sncode value
     """
    if value in SNPCC_LC_MAPPINGS["snibc"]:
        return 'Ibc'
    if value in SNPCC_LC_MAPPINGS["snii"]:
        return 'II'
    if value == 0:
        return 'Ia'
    raise ValueError('Unknown SNPCC supernova type!')


def read_tar_file(file_path: str) -> AnyStr:
    """
     Reads tarfile using with gzip compression and returns tarfile contents

     Parameters
     ----------
     file_path
         tarfile path
     """
    with tarfile.open(file_path, 'r:gz') as tar:
        tar_members = tar.getmembers()[0]
        return tar.extractfile(tar_members).read()


def read_resspect_full_photometry_data(file_path: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
    """
     Reads RESSPECT full photometry data to pandas dataframe

     Parameters
     ----------
     file_path
         RESSPECT data file path, one of (tar.gz, FITS, csv, csv.gz)
     """
    header = pd.DataFrame([])
    if file_path.endswith('.tar.gz'):
        tar_content = read_tar_file(file_path)
        return header, pd.read_csv(io.BytesIO(tar_content))
    if file_path.endswith('.FITS'):
        header, full_photometry = read_fits(
            file_path, drop_separators=True)
        return header, full_photometry
    if file_path.endswith(('.csv', '.csv.gz')):
        return header, pd.read_csv(file_path, index_col=False)
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
    """
     Returns updated filter values array by converting binary strings to proper
     filter values

     Parameters
     ----------
     filters_array
        array with binary string filter values
     filters
        available filter values
     """
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
    """
     Inserts band column(filter values) to RESSPECT dataframe by copying 'FLT'
     column values to a new column 'band'
     Also updates the filter values if they are binary strings.
     Parameters
     ----------
     photometry_df
         RESSPECT photometry dataframes
     filters
        Available filter values
     """
    if 'b' in str(photometry_df['FLT'].values[0]):
        updated_band = _update_resspect_filter_values(
            photometry_df['FLT'].values, filters)
        photometry_df.insert(1, 'band', updated_band, True)
    else:
        photometry_df.insert(1, 'band', photometry_df['FLT'].values, True)
    return photometry_df


def load_resspect_photometry_df(photometry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns updated RESSPECT photometry dataframe by dropping unnecessary
     columns ('SNID', 'FLT' and 'SIM_MAGOBS' columns are dropped here)

    Parameters
    ----------
    photometry_df
        RESSPECT photometry dataframe
    """
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
    """
     Reads PLAsTiCC full photometry data to pandas dataframe

     Parameters
     ----------
     file_path
         PLAsTiCC data file path, one of (tar.gz, csv, csv.gz)
     """
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
    """
    Updates PLAsTiCC filters id array by proper filter names

    Parameters
    ----------
    filters_array
        PLAsTiCC filters array
    mapping_dict
        filter id to name mapping dict
        ex: { 0: 'u', 1: 'g', ..}
    """
    updated_filters_array = np.zeros_like(filters_array, dtype=object)
    for key, value in mapping_dict.items():
        updated_filters_array[filters_array == key] = value
    return updated_filters_array


def load_plasticc_photometry_df(
        photometry_df: pd.DataFrame, filter_mapping_dict) -> pd.DataFrame:
    """
    Returns updated PLAsTiCC photometry dataframe by dropping unnecessary
     columns

    Parameters
    ----------
    photometry_df
        PLAsTiCC photometry dataframe
    filter_mapping_dict
        filter id to name mapping dict
        ex: { 0: 'u', 1: 'g', ..}
    """
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
    """
    Loads SNPCC photometry raw data to pandas dataframe
    Parameters
    ----------
    photometry_raw
        SNPCC photometry raw array
    header
        SNPCC header column names list
    """
    photometry_dict = {
        'mjd': np.array(
            photometry_raw[:, header.index('MJD')]).astype(float),
        'band': np.array(
            photometry_raw[:, header.index('FLT')]),
        'flux': np.array(
            photometry_raw[:, header.index('FLUXCAL')]).astype(float),
        'fluxerr': np.array(
            photometry_raw[:, header.index('FLUXCALERR')]).astype(float),
        'SNR': np.array(
            photometry_raw[:, header.index('SNR')]).astype(float),
        'MAG': np.array(
            photometry_raw[:, header.index('MAG')]).astype(float),
        'MAGERR': np.array(
            photometry_raw[:, header.index('MAGERR')]).astype(float)
    }
    return pd.DataFrame(photometry_dict)


def find_available_key_name_in_header(
        header_keys: list, keys_to_find: list) -> Union[str, None]:
    """
    Iteratively checks if one of the item of key_to_find is in header_keys list
    and returns the item which passes the condition

    Parameters
    ----------
    header_keys
        available header keys
    keys_to_find
        interesting keys list to search
    """
    for each_key in keys_to_find:
        if each_key in header_keys:
            return each_key
    return None


def get_resspect_header_data(
        path_header_file: str, path_photo_file: str) -> pd.DataFrame:
    """
    Reads RESSPECT meta header content: If header is a FITS file, photometry
    file will be used to extract the header content

    Parameters
    ----------
    path_header_file
        RESSPECT header file path
    path_photo_file
        RESSPECT photometry file path
    """
    if path_header_file.endswith(('tar.gz', '.csv', 'csv.gz')):
        _, meta_header = read_resspect_full_photometry_data(path_header_file)
    elif path_header_file.endswith('.FITS'):
        meta_header, _ = read_resspect_full_photometry_data(path_photo_file)
    else:
        raise ValueError(
            f"Unknown RESSPECT header data file: {path_header_file}")
    return meta_header


def maybe_create_directory(directory_path: str):
    """
    Creates diretory if it doesnt exist
    Parameters
    ----------
    directory_path
        directory path to create
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def get_query_flags(light_curve_data, telescope_names: list,
                    query_flags_threshold: int = 7200) -> list:
    """
    Checks if query is possible
    Parameters
    ----------
    light_curve_data
        An instance of LightCurve class
    telescope_names
        Names of the telescopes under consideration for spectroscopy.
        Only used if "get_cost == True".
        Default is ["4m", "8m"].
    query_flags_threshold
        Threshold for exposure time data
    """
    query_flags = []
    for each_name in telescope_names:
        if light_curve_data.exp_time[each_name] < query_flags_threshold:
            query_flags.append(True)
        else:
            query_flags.append(False)
    return query_flags
