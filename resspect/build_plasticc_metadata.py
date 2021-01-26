# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 26 January 2021
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

from resspect import read_fits
import re
import numpy as np
import pandas as pd
import os

__all__ = ['get_SNR_headers', 'calculate_SNR', 'build_metadata']


def get_SNR_headers():
    """Construct header for redshift and SNR summary statistics. 
    
    Returns
    -------
    list: header for output from function 'calculate_SNR'    
    """
    # LSST filters
    filters = [b'u ', b'g ', b'r ', b'i ', b'z ', b'Y ']
    
    # values which will be output per filter
    summary_stats = ['SIM_PEAKMAG_', 'SNR_mean_', 'SNR_max_', 'SNR_std_']

    # store SNR statistics
    SNR_names = ['SNID', 'snana_file_index', 'code_zenodo', 'code_SNANA', 'type', 'redshift']
    for stats in summary_stats:
        for fil in filters:
            SNR_names.append(stats + str(fil)[2])
            
    return SNR_names


def calculate_SNR(snid: int, train_ids: np.array, 
                  photo_data: pd.DataFrame, head_data: pd.DataFrame,
                 snana_file_index: int):
    """Calculate mean, max and std for all LSST filters.
    
    Parameters
    ----------
    snid: int
        Object identification number.
    train_ids: np.array
        All ids contained in training data from zenodo files.
    photo_data: pd.DataFrame
        Data from SNANA PHOT files.
    snana_file_index: int
        Identifies SNANA file hosting this object, [1-10].
    head_data: pd.DataFrame
        Data from SNANA HEAD files.
        
    Returns
    -------
    features_values: list
        Values corresponding to features_names.
    """
    
    types_names = {90: 'Ia', 62: 'Ibc', 42: 'II', 67: '91bg', 52: 'Iax',
                   64:'KN', 95: 'SLSN', 994: 'PISN', 992: 'ILOT', 
                   993: 'CaRT', 15: 'TDE', 88: 'AGN', 92: 'RRL', 
                   65: 'M-dw', 16: 'EB', 53: 'Mira', 991: 'BMicroL',
                   6: 'MicroL'}
    
    # LSST filters
    filters = [b'u ', b'g ', b'r ', b'i ', b'z ', b'Y ']
    
    if snid not in train_ids:

        flag_id = photo_data['SNID'] == snid

        flux = photo_data[flag_id]['FLUXCAL']
        fluxerr = photo_data[flag_id]['FLUXCALERR']

        SNR_all = flux/fluxerr

        redshift = head_data['SIM_REDSHIFT_CMB'].values[indx]
    
        # store values
        line = [snid, snana_file_index, code_zenodo, code_snana, 
                types_names[code_zenodo], redshift]

        # calculate SNR statistics 
        for fil in filters:                
            line.append(head_data['SIM_PEAKMAG_' + str(fil)[2]].values[indx])
            
            for f in [np.mean, max, np.std]:
                flag = photo_data[flag_id]['FLT'] == fil
                SNR_fil = SNR_all.values[flag]
                line.append(f(SNR_fil))
                
        return line
    
    else:
        return []


def build_plasticc_metadata(fname_meta: str, snana_dir: str, out_fname,
                            screen=False):
    """Save canonical metadata to file.
    
    Parameters
    ----------
    fname_meta: str
        Complete path to zenodo PLAsTiCC metadata file.
    snana_dir: str
        Path to directory containing all SNANA files for PLAsTiCC sim.
    out_fname: str
        Output file name.
    screen: bool (optional)
        If True, print intermediate steps to screen. Default is False.
        
    Returns
    -------
    Writes metadata to file. Columns include redshift, SIM_PEAKMAG and SNR
    statistics in different filters.
    """

    # map between zenodo and SNANA types
    SNANA_types = {90:11, 62:{1:3, 2:13}, 42:{1:2, 2:12, 3:14},
                   67:41, 52:43, 64:51, 95:60, 994:61, 992:62,
                   993:63, 15:64, 88:70, 92:80, 65:81, 16:83,
                   53:84, 991:90, 6:{1:91, 2:93}}
    
    # read zenodo metadata
    meta = pd.read_csv(fname_meta)

    # identify only DDF objects
    ddf_flag = meta['ddf_bool'].values == 1

    # get ids
    ids = meta['object_id'].values[ddf_flag]    

    names = get_SNR_headers()

    if not os.path.isfile(out_fname):
        op = open(out_fname, 'w+')
        for item in names[:-1]:
            op.write(item + ',')
        op.write(names[-1] + '\n')

    else: 
        op = open(out_fname, 'a+')
    
    for code_zenodo in list(SNANA_types.keys())[1:]:
    
        if screen:
            print(code_zenodo)

        if code_zenodo not in [62, 42, 6]:
            code_snana = SNANA_types[code_zenodo]

            for n in range(1, 11):
            
                if screen:
                    print('n = ', n)

                fname2 = snana_dir + 'LSST_DDF_MODEL' + str(code_snana).zfill(2) + \
                     '/LSST_DDF_NONIa-00' + str(n).zfill(2) + '_PHOT.FITS.gz'

                photo = read_fits(fname2)

                for indx in range(photo[0].shape[0]):
                 
                    if screen:
                        print('indx = ', indx)

                   # read data for 1 object
                    snid_raw = photo[0]['SNID'].values[indx]
                    snid = int(re.sub("[^0-9]", "", str(snid_raw)))
                
                    if screen:
                        print('snid = ', snid)

                    line = calculate_SNR(snid=snid, train_ids=train_ids, 
                                         photo_data=photo[1], head_data=photo[0],
                                         snana_file_index=n)
                
                    if len(line) > 0:                    
                        for item in line[:-1]:
                            op.write(str(item) + ',')
                        op.write(str(line[-1]) + '\n')
                    else:
                        print('Object in training')
                
        else:
            for subtype in SNANA_types[code_zenodo].keys():
                code_snana = SNANA_types[code_zenodo][subtype]
            
                for n in range(1, 11):
                    fname2 = snana_dir + 'LSST_DDF_MODEL' + str(code_snana).zfill(2) + \
                             '/LSST_DDF_NONIa-00' + str(n).zfill(2) + '_PHOT.FITS.gz'

                    photo = read_fits(fname2)

                    for indx in range(photo[0].shape[0]):

                        # read data for 1 object
                        snid_raw = photo[0]['SNID'].values[indx]
                        snid = int(re.sub("[^0-9]", "", str(snid_raw)))

                        line = calculate_SNR(snid=snid, train_ids=train_ids, 
                                             photo_data=photo[1], head_data=photo[0],
                                             snana_file_index=n)
                    
                        if len(line) > 0:                    
                            for item in line[:-1]:
                                op.write(str(item) + ',')
                            op.write(str(line[-1]) + '\n')
                        
                        else:
                            print('Object in training!')
                
    op.close()
    

def main():
    return None


if __name__ == '__main__':
    main()