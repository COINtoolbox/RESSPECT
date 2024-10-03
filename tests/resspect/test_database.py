# Copyright 2020 resspect software
# Author: Emille E. O. Ishida
#
# created on 30 March 2021
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
import pytest

from resspect import DataBase


def test_load_bazin_features(test_data_path):
    """Test loading Bazin features."""
    
    # test full light curve case
    fname1 = str(test_data_path / "Bazin.csv") 
    
    data1 = DataBase()
    data1.load_features(path_to_file=fname1,
                        screen=True, survey='DES', sample=None, feature_extractor='bazin')
    
    # read data independently
    data_temp1 = pd.read_csv(fname1)        
    
    sizes1 = len(data_temp1.keys()) == \
            len(data1.features_names) - 1 + len(data1.metadata_names)
    
    queryable1 = 'queryable' in data1.metadata_names
    
    # test time domain case
    fname2 = str(test_data_path / 'day_20.dat')
    
    data2 = DataBase()
    data2.load_features(path_to_file=fname2, screen=True,
                        survey='DES', sample=None, feature_extractor='bazin')
                              
    data_temp2 = pd.read_csv(fname2)
    
    sizes2 = len(data_temp2.keys()) == len(data2.features_names) + \
                                len(data2.metadata_names)
                                
    queryable2 = 'queryable' in data2.metadata_names
    
    assert (sizes1 and queryable1)
    assert (sizes2 and queryable2)
    
    
if __name__ == '__main__':
    pytest.main()
