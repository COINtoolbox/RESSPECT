# Copyright 2020 resspect software
# Author: Rupesh Durgesh, Emille Ishida, and Amanda Wasserman
#
# created on 14 April 2022
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import logging
import os
from copy import copy
from itertools import repeat
from typing import IO

import numpy as np
import pandas as pd

from resspect.feature_extractors.bazin import BazinFeatureExtractor
from resspect.feature_extractors.bump import BumpFeatureExtractor
from resspect.feature_extractors.malanchev import MalanchevFeatureExtractor
from resspect.lightcurves_utils import get_resspect_header_data
from resspect.lightcurves_utils import read_plasticc_full_photometry_data
from resspect.lightcurves_utils import SNPCC_FEATURES_HEADER
from resspect.lightcurves_utils import TOM_FEATURES_HEADER
from resspect.lightcurves_utils import TOM_MALANCHEV_FEATURES_HEADER
from resspect.lightcurves_utils import SNPCC_MALANCHEV_FEATURES_HEADER
from resspect.lightcurves_utils import find_available_key_name_in_header
from resspect.lightcurves_utils import PLASTICC_TARGET_TYPES
from resspect.lightcurves_utils import PLASTICC_RESSPECT_FEATURES_HEADER
from resspect.tom_client import TomClient

__all__ = ["fit_snpcc", "fit_plasticc", "fit_TOM", "request_TOM_data"]


FEATURE_EXTRACTOR_MAPPING = {
    "bazin": BazinFeatureExtractor,
    "bump": BumpFeatureExtractor,
    "malanchev": MalanchevFeatureExtractor
}


def _get_features_to_write(light_curve_data) -> list:
    """
    Returns features list to write

    Parameters
    ----------
    light_curve_data
        fitted light curve data
    """
    features_list = [light_curve_data.id, light_curve_data.redshift,
                     light_curve_data.sntype, light_curve_data.sncode,
                     light_curve_data.sample]
    features_list.extend(light_curve_data.features)
    return features_list


def write_features_to_output_file(
        light_curve_data, features_file: IO):
    """
    Writes fitted light curve data to output features file

    Parameters
    ----------
    light_curve_data
        fitted ligtht curve data
    features_file
        features output file
    """
    current_features = _get_features_to_write(
        light_curve_data)
    features_file.write(
        ','.join(str(each_feature) for each_feature
                 in current_features) + '\n')


def _snpcc_sample_fit(
        file_name: str, path_to_data_dir: str, feature_extractor: str):
    """
    Reads SNPCC file and performs fit.
    
    Parameters
    ----------
    file_name
        SNPCC file name
    path_to_data_dir
         Path to directory containing the set of individual files,
         one for each light curve.
    feature_extractor
        Function used for feature extraction.
        Options are 'bazin', 'bump', or 'malanchev'.
    """
    light_curve_data = FEATURE_EXTRACTOR_MAPPING[feature_extractor]()
    light_curve_data.load_snpcc_lc(
        os.path.join(path_to_data_dir, file_name))
    light_curve_data.fit_all()
    
    return light_curve_data


def fit_snpcc(
        path_to_data_dir: str, features_file: str,
        file_prefix: str = "DES_SN", number_of_processors: int = 1,
        feature_extractor: str = 'bazin'):
    """
    Perform fit to all objects in the SNPCC data.

     Parameters
     ----------
     path_to_data_dir: str
         Path to directory containing the set of individual files,
         one for each light curve.
     features_file: str
         Path to output file where results should be stored.
     file_prefix: str
        File names prefix
     number_of_processors: int, default 1
        Number of cpu processes to use.
     feature_extractor: str, default bazin
        Function used for feature extraction.
    """
    if feature_extractor == 'bazin':
        header = SNPCC_FEATURES_HEADER
    elif feature_extractor == 'malanchev':
        header = SNPCC_MALANCHEV_FEATURES_HEADER

    files_list = os.listdir(path_to_data_dir)
    files_list = [each_file for each_file in files_list
                  if each_file.startswith(file_prefix)]
    multi_process = multiprocessing.Pool(number_of_processors)
    logging.info("Starting SNPCC " + feature_extractor + " fit...")
    with open(features_file, 'w') as snpcc_features_file:
        snpcc_features_file.write(','.join(header) + '\n')
        
        for light_curve_data in multi_process.starmap(
                _snpcc_sample_fit, zip(
                    files_list, repeat(path_to_data_dir), repeat(feature_extractor))):
            if 'None' not in light_curve_data.features:
                write_features_to_output_file(
                    light_curve_data, snpcc_features_file)
    logging.info("Features have been saved to: %s", features_file)


def _plasticc_sample_fit(
        index: int, snid: int, path_photo_file: str,
        sample: str, light_curve_data,
        meta_header: pd.DataFrame):
    """
    Performs fit for PLAsTiCC dataset with snid

    Parameters
    ----------
    index
        index of snid
    snid
        Identification number for the desired light curve.
    path_photo_file: str
        Complete path to light curve file.
    sample: str
        'train' or 'test'. Default is None.
    light_curve_data
        light curve class
    meta_header
        photometry meta header data
    """
    light_curve_data.load_plasticc_lc(path_photo_file, snid)
    light_curve_data.fit_all()
    light_curve_data.redshift = meta_header['true_z'][index]
    light_curve_data.sncode = meta_header['true_target'][index]
    light_curve_data.sntype = PLASTICC_TARGET_TYPES[
        light_curve_data.sncode]
    light_curve_data.sample = sample
    light_curve_data_copy = copy(light_curve_data)
    light_curve_data.clear_data()
    return light_curve_data_copy


def fit_plasticc(path_photo_file: str, path_header_file: str,
                 output_file: str, sample='train',
                 feature_extractor: str = "bazin",
                 number_of_processors: int = 1):
    """
    Perform fit to all objects in a given PLAsTiCC data file.
    Parameters
    ----------
    path_photo_file: str
        Complete path to light curve file.
    path_header_file: str
        Complete path to header file.
    output_file: str
        Output file where the features will be stored.
    sample: str
        'train' or 'test'. Default is 'train'.
    number_of_processors: int, default 1
        Number of cpu processes to use.
    feature_extractor: str, default bazin
        feature extraction method
    """

    name_list = ['SNID', 'snid', 'objid', 'object_id']
    meta_header = read_plasticc_full_photometry_data(path_header_file)
    meta_header_keys = meta_header.keys().tolist()
    id_name = find_available_key_name_in_header(
        meta_header_keys, name_list)
    light_curve_data = FEATURE_EXTRACTOR_MAPPING[feature_extractor]()

    if sample == 'train':
        snid_values = meta_header[id_name]
    elif sample == 'test':
        light_curve_data.full_photometry = read_plasticc_full_photometry_data(
            path_photo_file)
        snid_values = pd.DataFrame(np.unique(
            light_curve_data.full_photometry[id_name].values),
            columns=[id_name])[id_name]
    snid_values = np.array(list(snid_values.items()))
    multi_process = multiprocessing.Pool(number_of_processors)
    logging.info("Starting PLAsTiCC " + feature_extractor + " fit...")
    with open(output_file, 'w') as plasticc_features_file:
        # TODO: Current implementation uses bazin features header for
        #  all feature extraction
        plasticc_features_file.write(
            ','.join(PLASTICC_RESSPECT_FEATURES_HEADER) + '\n')
        iterator_list = zip(
            snid_values[:, 0].tolist(), snid_values[:, 1].tolist(), repeat(path_photo_file),
            repeat(sample), repeat(light_curve_data), repeat(meta_header))
        for light_curve_data in multi_process.starmap(
                _plasticc_sample_fit, iterator_list):
            if 'None' not in light_curve_data.features:
                write_features_to_output_file(
                    light_curve_data, plasticc_features_file)
    logging.info("Features have been saved to: %s", output_file)

def _TOM_sample_fit(
        obj_dic: dict, feature_extractor: str):
    """
    Reads SNPCC file and performs fit.
    
    Parameters
    ----------
    id
        SNID
    feature_extractor
        Function used for feature extraction.
        Options are 'bazin', 'bump', or 'malanchev'.
    """
    light_curve_data = FEATURE_EXTRACTOR_MAPPING[feature_extractor]()
    light_curve_data.photometry = pd.DataFrame(obj_dic['photometry'])
    light_curve_data.dataset_name = 'TOM'
    light_curve_data.filters = ['u', 'g', 'r', 'i', 'z', 'Y']
    light_curve_data.id = obj_dic['objectid']
    light_curve_data.redshift = obj_dic['redshift']
    light_curve_data.sntype = 'unknown'
    light_curve_data.sncode = obj_dic['sncode']
    light_curve_data.sample = 'N/A'

    light_curve_data.fit_all()
    
    return light_curve_data

def fit_TOM(data_dic: dict, output_features_file: str,
            number_of_processors: int = 1,
            feature_extractor: str = 'bazin'):
    """
    Perform fit to all objects from the TOM data.

     Parameters
     ----------
     data_dic: str
         Dictionary containing the photometry for all light curves.
     output_features_file: str
         Path to output file where results should be stored.
     number_of_processors: int, default 1
        Number of cpu processes to use.
     feature_extractor: str, default bazin
        Function used for feature extraction.
    """
    if feature_extractor == 'bazin':
        header = TOM_FEATURES_HEADER
    elif feature_extractor == 'malanchev':
        header = TOM_MALANCHEV_FEATURES_HEADER

    multi_process = multiprocessing.Pool(number_of_processors)
    logging.info("Starting TOM " + feature_extractor + " fit...")
    with open(output_features_file, 'w') as TOM_features_file:
        TOM_features_file.write(','.join(header) + '\n')
        
        for light_curve_data in multi_process.starmap(
                _TOM_sample_fit, zip(
                    data_dic, repeat(feature_extractor))):
            if 'None' not in light_curve_data.features:
                write_features_to_output_file(
                    light_curve_data, TOM_features_file)
    logging.info("Features have been saved to: %s", output_features_file)

def request_TOM_data(url: str = "https://desc-tom-2.lbl.gov", username: str = None, 
                     passwordfile: str = None, password: str = None, detected_since_mjd: float = None, 
                     detected_in_last_days: float = None, mjdnow: float = None, cheat_gentypes: list = None):
    tom = TomClient(url = url, username = username, passwordfile = passwordfile, 
                    password = password)
    dic = {}
    if detected_since_mjd is not None:
        dic['detected_since_mjd'] = detected_since_mjd
    if detected_in_last_days is not None:
        dic['detected_in_last_days'] = detected_in_last_days
    if mjdnow is not None:
        dic['mjd_now'] = mjdnow
    if cheat_gentypes is not None:
        dic['cheat_gentypes'] = cheat_gentypes
    res = tom.post('elasticc2/gethottransients', json = dic)
    data_dic = res.json()
    return data_dic


def main():
    return None


if __name__ == '__main__':
    main()
