# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 14 April 2020
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

import pandas as pd


__all__ = ['efficiency', 'purity', 'fom', 'accuracy', 'get_snpcc_metric',
           'cosmo_metric', 'get_cosmo_metric']


def efficiency(label_pred: list, label_true: list, ia_flag=1):
    """Calculate efficiency.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: int (optional)
        Flag used to identify Ia objects. Default is 1.

    Returns
    -------
    efficiency: float
       Fraction of correctly classified SN Ia.

    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    tot_ia = sum([label_true[i] == ia_flag for i in range(len(label_true))])

    return float(cc_ia) / tot_ia


def purity(label_pred: list, label_true: list, ia_flag=1):
    """ Calculate purity.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: int (optional)
        Flag used to identify Ia objects. Default is 1.

    Returns
    -------
    Purity: float
        Fraction of true SN Ia in the final classified Ia sample.

    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    wr_nia = sum([label_pred[i] != label_true[i] and label_true[i] != ia_flag for i in range(len(label_pred))])

    if cc_ia + wr_nia > 0:
        return float(cc_ia) / (cc_ia + wr_nia)
    else:
        return 0


def fom(label_pred: list, label_true: list, ia_flag=1, penalty=3.0):
    """
    Calculate figure of merit.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: bool (optional)
        Flag used to identify Ia objects. Default is 1.
    penalty: float
        Weight given for non-Ias wrongly classified.

    Returns
    -------
    figure of merit: float
        Efficiency x pseudo-purity (purity with a penalty for false positives)

    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    wr_nia = sum([label_pred[i] != label_true[i] and label_true[i] != ia_flag for i in range(len(label_pred))])
    tot_ia = sum([label_true[i] == ia_flag for i in range(len(label_true))])

    if (cc_ia + penalty * wr_nia) > 0:
        return (float(cc_ia) / (cc_ia + penalty * wr_nia)) * float(cc_ia) / tot_ia
    else:
        return 0


def accuracy(label_pred: list, label_true: list):
    """Calculate accuracy.

    Parameters
    ----------
    label_pred: list
        predicted labels
    label_true: list
        true labels

    Returns
    -------
    Accuracy: float
        Global fraction of correct classifications.

    """

    cc = sum([label_pred[i] == label_true[i] for i in range(len(label_pred))])

    return cc / len(label_pred)


def get_snpcc_metric(label_pred: list, label_true: list, ia_flag=1,
                     wpenalty=3):
    """
    Calculate the metric parameters used in the SNPCC.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: bool (optional)
        Flag used to identify Ia objects. Default is 1.
    wpenalty: float
        Weight given for non-Ias wrongly classified.


    Returns
    -------
    metric_names: list
        Name of elements in metrics: [accuracy, eff, purity, fom]
    metric_values: list
        list of calculated metrics values for each element

    """

    calc_eff = efficiency(label_pred=label_pred,
                          label_true=label_true, ia_flag=ia_flag)
    calc_purity = purity(label_pred=label_pred,
                         label_true=label_true, ia_flag=ia_flag)
    calc_fom = fom(label_pred=label_pred,
                   label_true=label_true, ia_flag=ia_flag, penalty=wpenalty)
    calc_accuracy = accuracy(label_pred=label_pred, label_true=label_true)

    metric_values = [calc_accuracy, calc_eff, calc_purity, calc_fom]
    metric_names = ['accuracy', 'efficiency', 'purity', 'fom']

    return metric_names, metric_values

def cosmo_metric(data: str, comp_data: str):
    """Calculate the Fisher-matrix based difference between two sets.

    Parameters
    ----------
    data: str or pd.DataFrame
        Path to original data set or data frame from which 
        Fisher matrix will be calculated.
        The data should be formated: ['id', 'z', 'mu', 'muerr'].
    comp_data: str or pd.DataFrame
        Path to second data set or data frame to be compared 
        to the original data.
        The data should be formated: ['id', 'z', 'mu', 'muerr'].   
 
    Returns
    -------
    metrics_names: list
        Name of elements in metrics: [fisher_diff]
    metric_values: list
        list of calculated metrics values for each element
    """
    from resspect.cosmo_metric_utils import compare_two_fishers
    
    # read distances
    if isinstance(data, str):
        data = pd.read_csv(data)
    
    if isinstance(comp_data, str):
        comp_data = pd.read_csv(comp_data)

    # get useful columns only
    data1 = data[['z','mu','mu_err']].values
    data2 = comp_data[['z','mu','mu_err']].values
    
    # compare results from 2 fisher matrices
    fisher_diff = compare_two_fishers(data1, data2)
    
    return ['fisher_diff'], [fisher_diff]

def get_cosmo_metric(input_fname_root: str, loop: int, 
        calc_dist=False, 
        data_folder='/media/RESSPECT/data/RESSPECT_NONPERFECT_SIM/SNDATA_SIM_TMP/SNDATA_SIM_TMP/',
        data_prefix='RESSPECT_LSST_TEST_DDF_PLASTICC', maxsnnum=1000, 
        salt2mu_prefix='test_salt2mu_res', salt3_outfile='salt3pipeinput.txt',
        save_dist=True, select_modelnum=[90], select_orig_sample=['train'],
        dist_root_fname='dist_loop_'):
    """Calculate distances and cosmology metrics for a list of ids.

    Parameters
    ----------
    input_fname_root: list str 
        Root file name for input data for cosmology metric.
        Files should contain distances or object ids.
    loop: int 
        Current learning loop. 
        The cosmology metric can only be calculated in comparison
        to a previous state. "loop = 0" does not generate output.
    calc_dist: bool (optional)
        If True, calculate distances for objects given ids.
        Otherwise, read distances from file. Default is False.
    data_folder: str (optional)
        Folder containing all SNANA files. Only used if "calc_dist == True".
        Default is set to RESSPECT data in COIN server.
    data_prefix: str (optional)
        Root of SNANA file name for this sim. Only used if "calc_dist == True".
        Default is "RESSPECT_LSST_TEST_DDF_PLASTICC".
    dist_root_fname: str (optional)
        Root for output file name where calculated distances will be stored. 
        Only used if "calc_dist == True". Default is 'dist_loop_'.
    maxsnnum: int (optional)
        Number of objects to be fitted. 
        If this number is lower than the total number of objects, 
        a random sample will be chosen. Otherwise all objects will
        be fitted. Only used if "calc_dist == True". Default is 1000.
    salt2mu_prefix: str (optional)
        File name for output SALT2 fitres file. Only used if "calc_dist == True".
        Default is "test_salt2mu_res".
    salt3_outfile: str (optional)
        Output file for SALT3 results. Only used if "calc_dist == True".
        Default is 'salt3pipeinput.txt'.
    save_dist: bool (optional)
        If True, save calculated distances to file. 
        Only used if "calc_dist == True". Default is True.
    select_model_num: list int (optional)
        List of codes for models to be fitted. Only used if "calc_dist == True".
        At this point only Ias, [90], are implemented.    
    select_orig_sample: list str (optional)
        Original sample. It can only handle one element at this point.
        Only used if "calc_dist == True".
        ['train'] or ['test'] is accepted. Default is ['train'].
     
    Returns
    -------
    metrics_names: list
        Name of elements in metrics: [fisher_diff]
    metric_values: list
        list of calculated metrics values for each element
    """
    from resspect.salt3_utils import get_distances

    if loop == 0:
        # if first loop, return nothing
        return  ['fisher_diff'], []
    else:
        if calc_dist:
            dist_original = get_distances(input_fname_root + str(loop - 1) + '.csv',
                            data_folder=data_folder, data_prefix=data_prefix, 
                            select_modelnum=select_modelnum, salt2mu_prefix=salt2mu_prefix,
                            maxsnnum=maxsnnum, select_orig_sample=select_orig_sample,
                            salt3_outfile=salt3_outfile)

            dist_comp = get_distances(input_fname_root + str(loop) + '.csv',
                            data_folder=data_folder, data_prefix=data_prefix, 
                            select_modelnum=select_modelnum, salt2mu_prefix=salt2mu_prefix,
                            maxsnnum=maxsnnum, select_orig_sample=select_orig_sample,
                            salt3_outfile=salt3_outfile)

            if save_dist:
                if loop == 1:
                    dist_original.to_csv(dist_root_fname + str(loop - 1) + '.csv', index=False)
                dist_comp.to_csv(dist_root_fname + str(loop) + '.csv', index=False)
                
        else:
           dist_original = input_fname_root + str(loop - 1) + '.csv'
           dist_comp = input_fname_root + str(loop) + '.csv'

        names, values = cosmo_metric(data=dist_original, comp_data=dist_comp)            
        
        return names, values

      
def main():
    return None


if __name__ == '__main__':
    main()
