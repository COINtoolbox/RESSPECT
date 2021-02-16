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

from resspect.snana_fits_to_pd import read_fits
import re
import numpy as np
import pandas as pd
import os

__all__ = ['get_SNR_headers', 'calculate_SNR', 'build_plasticc_metadata']


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
    SNR_names = ['SNID', 'snana_file_index', 'code_zenodo', 'code_SNANA',
                 'type', 'redshift']
    for stats in summary_stats:
        for fil in filters:
            SNR_names.append(stats + str(fil)[2])
            
    return SNR_names


def calculate_SNR(snid: int, photo_data: pd.DataFrame, 
                  head_data: pd.DataFrame, code_zenodo: int, 
                  snana_file_index: int, code_snana: int):
    """Calculate mean, max and std for all LSST filters.
    
    Parameters
    ----------
    code_snana: int
        Type identification in SNANA files.
    code_zenodo: int
        Type identification in zenodo files.
    head_data: pd.DataFrame
        Data from SNANA HEAD files.
    photo_data: pd.DataFrame
        Data from SNANA PHOT files.
    snana_file_index: int
        Identifies SNANA file hosting photometry for this object, [1-10].
    snid: int
        Object identification number.
        
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

    flag_id_photo = photo_data['SNID'] == snid

    flux = photo_data[flag_id_photo]['FLUXCAL'].values
    fluxerr = photo_data[flag_id_photo]['FLUXCALERR'].values

    SNR_all = flux/fluxerr
    
    indx = np.random.choice(range(flux.shape[0]))

    flag_id_head = head_data['SNID'].values == snid
    redshift = head_data['SIM_REDSHIFT_CMB'].values[flag_id_head][0]
    
    # store values
    line = [snid, snana_file_index, code_zenodo, code_snana, 
            types_names[code_zenodo], redshift]
    
    for fil in filters: 
        line.append(head_data['SIM_PEAKMAG_' + str(fil)[2]].values[flag_id_head][0])
        
    # calculate SNR statistics 
    for f in [np.mean, max, np.std]:               
        for fil in filters: 
            
            flag_fil = photo_data[flag_id_photo]['FLT'] == fil
            neg_flag = flux > -100
            flag2 = np.logical_and(flag_fil, neg_flag)
            
            if sum(flag2) > 0:
                SNR_fil = SNR_all[flag2]    
                line.append(f(SNR_fil))
    
    if len(line) == 30:
        return line
    else:
        return []


def build_plasticc_metadata(fname_meta: str, snana_dir: str, out_fname,
                            screen=False, extragal=True, field='DDF'):
    """Save canonical metadata to file.
    
    Parameters
    ----------
    fname_meta: str
        Complete path to zenodo PLAsTiCC metadata file.
    snana_dir: str
        Path to directory containing all SNANA files for PLAsTiCC sim.
    out_fname: str
        Output file name.
    extragal: bool (optional)
        If True, search only for extragalactic objects. Default is True.
    field: str (optional)
        Fields to be consider. Options are 'DDF', 'WFD' or 'all'.
        Default is DDF.
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
    
    extragal_zenodo_types = [15, 42, 52, 62, 64, 67, 88, 90, 95]
    
    # read zenodo metadata
    meta = pd.read_csv(fname_meta)

    if field == 'DDF':
        # identify only DDF objects
        ddf_flag = meta['ddf_bool'].values == 1
    elif field == 'WFD':
        ddf_flag = meta['ddf_bool'].values == 0
    else:
        ddf_flag = np.array([True for i in range(meta.shape[0])])

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
    
    # which group to search for
    if extragal:
        search_group = extragal_zenodo_types
    else:
        search_group = list(SNANA_types.keys())
    
    for code_zenodo in search_group:
    
        if screen:
            print('code_zenodo: ', code_zenodo)

        if code_zenodo not in [62, 42, 6]:
            code_snana = SNANA_types[code_zenodo]

            for n in range(1, 11):
                fname2 = snana_dir + 'LSST_DDF_MODEL' + str(code_snana).zfill(2) + \
                     '/LSST_DDF_NONIa-00' + str(n).zfill(2) + '_PHOT.FITS.gz'

                photo = read_fits(fname2)

                for indx in range(photo[0].shape[0]):
                   # read data for 1 object
                    snid_raw = photo[0]['SNID'].values[indx]
                    snid = int(re.sub("[^0-9]", "", str(snid_raw)))

                    if snid in ids:                        
                        line = calculate_SNR(snid=snid, 
                                             code_zenodo=code_zenodo,
                                             photo_data=photo[1],
                                             head_data=photo[0],
                                             snana_file_index=n,
                                             code_snana=code_snana)
                        
                        if len(line) > 0:
                            for item in line[:-1]:
                                op.write(str(item) + ',')
                            op.write(str(line[-1]) + '\n')
                        
            del photo
                
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

                        if snid in ids:
                            line = calculate_SNR(snid=snid, 
                                                 code_snana=code_snana,
                                                 code_zenodo=code_zenodo,
                                                 photo_data=photo[1], 
                                                 head_data=photo[0],
                                                 snana_file_index=n)
                            
                            if len(line) > 0:
                                for item in line[:-1]:
                                    op.write(str(item) + ',')
                                op.write(str(line[-1]) + '\n')
                    
                    del photo
                
    op.close()
    

def main():
    return None


if __name__ == '__main__':
    main()