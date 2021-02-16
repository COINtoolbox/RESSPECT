# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 29 January 2021
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
# PLAsTiCC data sample, in the full light curve scenario. It        #
# assumes you already performed feature                             #
# extraction for the original train and test samples.               #
#                                                                   #
#  It requires information from the original SNANA sim files.       #
#                                                                   #
#####################################################################

from resspect import build_plasticc_metadata, build_plasticc_canonical

## Build metadata file
for sample in ['train', 'test']:
    build_plasticc_metadata(fname_meta='~/plasticc_' + sample + '_metadata.csv',
                            snana_dir='~/PLAsTiCC/SNANA/', 
                            out_fname='plasstic_' + sample + '_metadata_extragl_DDF.csv',
                            screen=True, field='DDF', models='extragal')

## path to metadata 
path_to_metadata = {}
path_to_metadata['train'] = out_dir + 'plasstic_train_metadata_extragl_DDF.csv'
path_to_metadata['test'] = out_dir + 'plasstic_test_metadata_extragl_DDF.csv'

# path to input features files
input_features_files = {}
input_features_files['test'] = '~/plasticc_test_bazin_extragal_DDF.csv.gz'
input_features_files['validation'] = '~/plasticc_validation_bazin_extragal_DDF.csv.gz'

# path to output canonical file names
output_features_files = {}
output_features_files['pool'] = 'data/plasticc_pool_bazin_extragal_DDF.csv'
output_features_files['test'] = 'data/plasticc_test_bazin_extragal_DDF.csv'
output_features_files['validation'] = 'data/plasticc_validation_bazin_extragal_DDF.csv'

output_meta_file = 'data/plasticc_pool_metadata_extragal_DDF.csv'

# name for output plot
plto_fname = 'plots/compare_train_canonical.png'

# build canonical sample
build_plasticc_canonical(n_neighbors=2, path_to_metadata=path_to_metadata,
                         output_features_files=output_features_files, 
                         output_meta_file=output_meta_file,
                         input_features_files=input_features_files,
                         screen=True, plot=True, 
                         plot_fname=plot_fname)
    
    
    
    