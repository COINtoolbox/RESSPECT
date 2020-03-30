# Copyright 2020 snactclass software
# Author: Mi Dai
#         
# created on 19 March 2019
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

import numpy as np
import pandas as pd
import os
from .snanapipe.snana_hook import SNANAHook
from io import StringIO

__all__ = ['get_distances']

def get_distances(snid_file,data_folder=None,data_prefix=None,select_modelnum=None,salt2mu_prefix='test_salt2mu',
                  maxsnnum=1000,select_orig_sample=None,**kwargs):
    """Calculates distances and erros for cosmology metric.

    Parameters
    ----------
    snid_file (str): filename of file that contains information of input snids
    data_folder (str): folder that contains the raw SNANA simulation files
    data_prefix (str): prefix/genversion of SNANA sim
    select_modelnum (list): SNANA model number (only one number is allowed for now)
    select_orig_sample (list): 'train' or 'test' (only one sample is allowed for now)
    salt2mu_prefix (str): filename prefix for SALT2mu output
    maxsnnum:

    Returns
    -------
    result_df (pd.DataFrame): columns=['id','z','mu','mu_err','fitprob']
    
    
    Example
    -------
    from actsnclass.cosmo_metric import get_distances
    get_distances('results/photo_ids/test_photoids_loop_0.dat',
                  data_prefix='RESSPECT_LSST_TRAIN_DDF_PLASTICC',
                  data_folder='/media/RESSPECT/data/RESSPECT_NONPERFECT_SIM/SNDATA_SIM_TMP/SNDATA_SIM_NEW_RATES',
                  select_modelnum=[90],
                  salt2mu_prefix='test_salt2mu_res',
                  maxsnnum=10,
                  select_orig_sample=['train'])
    """

    result_dict = parse_snid_file(snid_file,select_modelnum=select_modelnum,maxsnnum=maxsnnum,select_orig_sample=select_orig_sample)
    for f,modelnum,sntype in zip(result_dict['snid'],result_dict['modelnum'],result_dict['sntype']):
        phot_version = '{}_MODEL{}_SN{}'.format(data_prefix,modelnum,sntype)
        hook = SNANAHook(snid_file=f,data_folder=data_folder,phot_version=phot_version,salt2mu_prefix=salt2mu_prefix)
        hook.run()    
        
    result_df = parse_salt2mu_output('{}.fitres'.format(salt2mu_prefix))
    
    return result_df

def parse_snid_file(snid_file,outfolder='snid',select_modelnum=None,select_orig_sample=None,maxsnnum=1000):
    df = pd.read_csv(snid_file)
    df['modelnum'] = [int(str(x)[:2]) for x in df['code']]
    snid_file_list = []
    modelnum_list = []
    sntype_list = []
    orig_sample_list = []
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)
    if select_modelnum is None:
        numlist = df['modelnum'].unique()
    else:
        numlist = select_modelnum
    if select_orig_sample is None:
        samplelist = df['orig_sample'].unique()
    else:
        samplelist = select_orig_sample
    for samplename in samplelist:
        for num in numlist:
            f = '{}/{}_{}_{}'.format(outfolder,os.path.split(snid_file)[1],samplename,num)
            df_subsample = df.set_index(['orig_sample','modelnum']).loc[[samplename,num]]
            df_subsample = df_subsample.sample(np.min([len(df_subsample),maxsnnum]))
            df_subsample['id'].to_csv(f,index=False)        
            snid_file_list.append(f)
            modelnum_list.append(num)
            orig_sample_list.append(samplename)
            sntype = df_subsample.drop_duplicates('type')['type'].values[0]
            if sntype == 'Ia':
                sntype = 'Ia-SALT2'
            sntype_list.append(sntype)
    return {'snid':snid_file_list,'modelnum':modelnum_list,'sntype':sntype_list,'orig_sample':orig_sample_list}

def parse_salt2mu_output(fitres_file,timeout=50):
    timetot = 0
    while not os.path.isfile(fitres_file) and timetot<timeout:
        time.sleep(5)
        timetot += 5
        
    if not os.path.isfile(fitres_file):
        raise RuntimeError("salt2mu fitres file [{}] does not exist".format(fitres_file))
    
    cname_map = {'id':'CID',
                 'z':'zHD',
                 'mu':'MU',
                 'mu_err':'MUERR',
                 'fitprob':'FITPROB'}
    with open(fitres_file,'r') as f:
        lines = f.readlines()
    newlines = []
    for line in lines:
        if (not line.strip().startswith('SN:')) and (not line.strip().startswith('VARNAMES:')):
            continue
        newlines.append(line)
    string_to_read = StringIO('\n'.join(newlines))
    fitres = pd.read_csv(string_to_read,sep='\s+',comment='#')
    cols = [value for key, value in cname_map.items()]
    df = fitres[cols]
    df.columns = [key for key, value in cname_map.items()]
    return df

def main():
    return None

if __name__ == '__main__':
    main()
