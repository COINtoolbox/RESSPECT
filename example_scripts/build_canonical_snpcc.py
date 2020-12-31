# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 31 December 2020
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

#######################  README  ####################################
#                                                                   #
# This script shows examples on how to build canonical data for the #
# SNPCC data sample.                                                #
#                                                                   #    
#####################################################################

time_domain = True

if time_domain:

    import pandas as pd
    import numpy as np
    
    # file with full light curve canonical sample
    canonical_fname = '~/data/Bazin_SNPCC_canonical.dat'
    
    # directory with full original sample time domain
    td_dir = '~/data/time_domain/'
    
    # output directory for canonical time domain sample
    outdir_td_canonical = '~/data/time_domain_canonical/'
    
    # read canonical data
    canonical = pd.read_csv(canonical_fname, sep=' ', index_col=False)
    canonical_ids = canonical[canonical['queryable'].values]['id'].values
    
    print(' Total number of canonical objects: ', len(canonical_ids))
    
    # read already processed time domain sample and separate canonical
    for day in range(20, 181):
    
        data = pd.read_csv(td_dir + 'day_' + str(day) + '.dat', sep=' ',
                           index_col=False)
        data2 = data.copy(deep=True)
        
        print(' Total number of objects in day ', day, ' = ', data.shape[0])
        
        flag_canonical = np.array([item in canonical_ids 
                                   for item in data['id'].values])
        flag_final = np.logical_and(flag_canonical, data['queryable'].values)
        
        data2['queryable'] = flag_final
        
        print('Difference in day ', day, ' = ', 
              sum(flag_final) - sum(data['queryable'].values))
        print('   Size canonical = ', sum(flag_canonical))
        print('   Queryable in ', day, ' = ', sum(data['queryable'].values))
        
        data2.to_csv(outdir_td_canonical + 'day_' + str(day) + '.dat',
                     index=False)
                     
else:

    from resspect import build_snpcc_canonical
    from resspect import plot_snpcc_train_canonical

    # define variables
    data_dir = '~/data/SIMGEN_PUBLIC_DES/'         # raw data directory
    features_file = '~/data/Bazin.dat'             # features file for full LC
    
    output_sample_file = 'Bazin_SNPCC_canonical.dat'         
    output_metadata_file = 'Bazin_metadata.dat'
    output_plot_file = 'compare_canonical_train.png'    

    sample = build_snpcc_canonical(path_to_raw_data=data_dir, path_to_features=features_file,
                                   output_canonical_file=output_sample_file,
                                   output_info_file=output_metadata_file,
                                   compute=True, save=True)
                               
    plot_snpcc_train_canonical(sample, output_plot_file=output_plot_file)
