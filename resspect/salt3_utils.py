# Copyright 2020 resspect software
# Author: Mi Dai
#         
# created on 30 March 2020
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

__all__ = ['get_distances', 'parse_salt2mu_output', 'parse_snid_file']


def get_distances(snid_file,data_folder: str, data_prefix: str,
                  select_modelnum: list, select_orig_sample: list,
                  salt2mu_prefix='test_salt2mu',
                  fitres_prefix='test_fitres',
                  combined_fitres_name='test_fitres_combined.fitres',
                  maxsnnum=1000, 
                  salt3_outfile='salt3pipeinput.txt',
                  salt3_tempfile='$RESSPECT_DIR/auxiliary_files/salt3pipeinput_template.txt',
                  outputdir='results/salt3',
                  data_prefix_has_sntype=True,
                  **kwargs):
    """Calculates distances and errors for cosmology metric.

    Parameters
    ----------
    snid_file: str 
        Filename of file that contains information of input snids.
    data_folder: str
        Folder that contains the raw SNANA simulation files.
    data_prefix: str
        Prefix/genversion of SNANA sim.
    select_modelnum: list
        list of SNANA model numbers to be fitted 
    select_orig_sample: list
        Original simulated sample. 
        Options are ['train'] or ['test'] (only one sample is allowed for now).
    salt2mu_prefix: str (optional)
         Filename prefix for SALT2mu output.
         Default is 'test_salt2mu'.
    fitres_prefix: str (optional)
        prefix for snana light curve fitting result files
    combined_fitres_name: str (optional)
        filename for the combined (different types) fitres file
    maxsnnum: int (optional)
         max number of objects to fit. Default is 1000.
    salt3_outfile: str (optional)
         generated salt3 input file
    salt3_tempfile: str (optional)
         template file for SALT3 input    
    outputdir: str (optional)
         dir of all outputs
    data_prefix_has_sntype: bool (optional)
         True if _sntype is part of the data_prefix

    Returns
    -------
    result_df: dict
        'distances': pd.DataFrame
            Keywords: ['id','z','mu','mu_err','fitprob']
        'cosmopars': pd.DataFrame
            Keywords: ['w','wsig_marg','OM','OM_sig','chi2','Ndof','sigint','wran','OMran','label']
    
    Example
    -------
    >>> from resspect.cosmo_metric import get_distances
    
    >>> get_distances('results/photo_ids/test_photoids_loop_0.dat',
                  data_prefix='RESSPECT_LSST_TRAIN_DDF_PLASTICC',
                  data_folder='~/SNDATA_SIM_NEW_RATES/',
                  select_modelnum=[90],
                  salt2mu_prefix='test_salt2mu_res',
                  maxsnnum=10,
                  select_orig_sample=['train'],
                  data_prefix_has_sntype=True)
    """

    #check if outputdir exists. create one if not.
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)    
    salt3_outfile = os.path.join(outputdir,salt3_outfile)
    
    result_dict = parse_snid_file(snid_file, select_modelnum=select_modelnum,
                                  select_orig_sample=select_orig_sample,
                                  maxsnnum=maxsnnum,outfolder=os.path.join(outputdir,'snid'),
                                  **kwargs)

    fitres_list = []
    for i,(f,modelnum,sntype) in enumerate(zip(result_dict['snid'],
                                               result_dict['modelnum'],
                                               result_dict['sntype'])):
        print(i,"modelnum =",modelnum,"sntype=",sntype)

        prefix = os.path.join(outputdir,fitres_prefix.strip()+'_'+str(i))
        fitres_file = prefix+'.FITRES.TEXT'
        if data_prefix_has_sntype:
            if int(modelnum) == 99:
                phot_version = '{}_MODEL{}_{}'.format(data_prefix,modelnum,sntype)
            else:
                phot_version = '{}_MODEL{}_SN{}'.format(data_prefix,modelnum,sntype)
        else:
            phot_version = '{}_MODEL{:02d}'.format(data_prefix,modelnum)
        hook = SNANAHook(snid_file=f, data_folder=data_folder, 
                         phot_version=phot_version, fitres_prefix=prefix,
                         salt2mu_prefix = salt2mu_prefix,
                         stages=['lcfit'], glue=False,tempfile=salt3_tempfile,
                         outfile=salt3_outfile,**kwargs)
        hook.run() 
        fitres_list.append(fitres_file)
    
    if len(fitres_list) == 0:
        print("No objects were fitted - check input modelnums")
        return pd.DataFrame(columns=['id','z','mu','mu_err','fitprob'])

    combined_fitres_name_str = os.path.join(outputdir,combined_fitres_name)
    combine_fitres(fitres_list,output=combined_fitres_name_str)
    salt2mu_prefix_str = os.path.join(outputdir,salt2mu_prefix.strip()+'_combined')
    hook = SNANAHook(salt2mu_prefix=salt2mu_prefix_str,
                     combined_fitres=combined_fitres_name_str,
                     stages=['getmu','cosmofit'], glue=False,
                     outfile=salt3_outfile,
                     tempfile=salt3_tempfile,**kwargs)
    hook.run()
      
    result_df = {}    
    result_df['distances'] = parse_salt2mu_output('{}.fitres'.format(salt2mu_prefix_str))
    result_df['cosmopars'] = parse_wfit_output('{}.M0DIF.cospar'.format(salt2mu_prefix_str))
    
    return result_df


def parse_snid_file(snid_file: str,
                    select_modelnum: list, select_orig_sample: list,
                    outfolder='snid',
                    maxsnnum=1000,
                    random_state=None,
                    **kwargs):
    """Parse the snid file that is output from the pipeline.

    Parameters
    ----------
    select_modelnum: list
        list of SNANA model numbers
    select_orig_sample: list
        Original simulated sample. 
        Options are ['train'] or ['test'] (only one sample is allowed for now).
    snid_file: str
        snid file that is output from the pipeline
    maxsnnum: int (optional)
        maximum of number of sn to return. Default is 1000. If smaller than the total number in snid_file, a random sample is drawn.
    outfolder: str (optional)
        output folder that stores the parsed snid files for each type and sample. Default is 'snid'.
    random_state: int or numpy.random.RandomState, optional
        random_state in pd.DataFrame.sample() for sampling maxsnnum of objects    
    
    Returns
    -------
    dict
        Keywords: ['snid', 'modelnum', 'sntype', 'orig_sample']
    """

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

    # get object identifier
    if 'id' in df.keys():
        id_name = 'id'
    elif 'object_id' in df.keys():
        id_name = 'object_id'
    elif 'objid' in df.keys():
        id_name = 'objid'
    
    for samplename in samplelist:
        for num in numlist:
            f = '{}/{}_{}_{}'.format(outfolder, os.path.split(snid_file)[1],
                                     samplename,num)
            if num not in df.modelnum.unique():
                continue
            df_subsample = df.set_index(['orig_sample','modelnum']).loc[(samplename,num)]
            df_subsample = df_subsample.sample(np.min([len(df_subsample),maxsnnum]),random_state=random_state)
            df_subsample[id_name].to_csv(f,index=False)        
            snid_file_list.append(f)
            modelnum_list.append(num)
            orig_sample_list.append(samplename)
            sntype = df_subsample.drop_duplicates('type')['type'].values[0]

            if sntype == 'Ia':
                sntype = 'Ia-SALT2'
            sntype_list.append(sntype)

    return {'snid':snid_file_list, 'modelnum':modelnum_list,
            'sntype':sntype_list, 'orig_sample':orig_sample_list}


def parse_salt2mu_output(fitres_file: str, timeout=50):
    """Parse the salt2mu output and return useful columns as a pd.DataFrame.

    Parameters
    ----------
    fitres_file: str
        The salt2mu output
    timeout: int (optional)
        Max waiting time for determining if the file exists. Default is 50 (seconds). 
        This is used to avoid error caused by a delay in the salt2mu output being written out.

    Returns
    -------
    df: pd.DataFrame
        Keywords: ['id','z','mu','mu_err','fitprob']
    """

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
        if (not line.strip().startswith('SN:')) and \
             (not line.strip().startswith('VARNAMES:')):
            continue
        newlines.append(line)

    string_to_read = StringIO('\n'.join(newlines))
    fitres = pd.read_csv(string_to_read,sep='\s+',comment='#')
    cols = [value for key, value in cname_map.items()]
    df = fitres[cols]

    df.columns = [key for key, value in cname_map.items()]

    return df

                
def combine_fitres(fitres_list,output='fitres_combined.fitres'):
    dflist = []
    for fitres in fitres_list:
        try:
            df = pd.read_csv(fitres,comment='#',sep='\s+')
        except pd.io.common.EmptyDataError:
            df = pd.DataFrame()

        dflist.append(df)

    res = pd.concat(dflist,ignore_index=True,sort=False).dropna(axis=1)
    res.to_csv(output,index=False,sep=' ',float_format='%.5e')
#     cmd = 'cat {} > {}'.format(' '.join(fitres_list),output)
#     os.system(cmd)

def parse_wfit_output(cospar_file):
    """Parse the wfit output and return as a pd.DataFrame.

    Parameters
    ----------
    cospar_file: str
        The wfit output

    Returns
    -------
    df: pd.DataFrame
        Keywords: ['w','wsig_marg','OM','OM_sig','chi2','Ndof','sigint','wran','OMran','label']
    """
    
    df = pd.read_csv(cospar_file,comment='#',sep='\s+',header=None)
    df.columns = ['w','wsig_marg','OM','OM_sig','chi2','Ndof','sigint','wran','OMran','label']

    return df


def main():
    return None

if __name__ == '__main__':
    main()
