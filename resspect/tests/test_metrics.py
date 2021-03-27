# Copyright 2021 resspect software
# Author: Emille Ishida
#
# created on 12 March 2021
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

import pytest
import numpy as np


@pytest.fixture(scope='module')
def labels():
    """Define the data dict with mock classes.
    """
    
    labels = {}
    labels['real'] = [0,0,0,0,0,1,1,1,1,1]
    labels['pred'] = [0,1,0,0,0,0,0,1,1,1]
    
    return labels


def test_efficiency(labels):
    """Test efficiency consistency.
    
    Parameters
    ----------
    labels: fixture
        Mock real and predicted labels.
    """

    from resspect import efficiency
    
    eff = efficiency(labels['pred'], labels['real'])
    
    assert eff == 0.6
    

def test_purity(labels):
    """Test purity consistency.
    
    Parameters
    ----------
    labels: fixture
        Mock real and predicted labels.
    """
    
    from resspect import purity
    
    pur = purity(labels['pred'], labels['real'])
    
    assert pur == 0.75
    

def test_fom(labels):
    """Test accuracy consistency.
    
    Parameters
    ----------
    labels: fixture
        Mock real and predicted labels.
    """
    
    from resspect import fom, efficiency
    
    res = fom(labels['pred'], labels['real'])
    
    assert res == 0.3


def test_accuracy(labels):
    """Test accuracy consistency.
    
    Parameters
    ----------
    labels: fixture
        Mock real and predicted labels.
    """
    
    from resspect import accuracy
    
    acc = accuracy(labels['pred'], labels['real'])
    
    assert acc == 0.7

def test_get_snpcc_metric(labels):
    """Test all snpcc metrics consistency.
    
    Parameters
    ----------
    labels: fixture
        Mock real and predicted labels.
    """
    
    from resspect import get_snpcc_metric
    
    res = get_snpcc_metric(labels['pred'], labels['real'])
    val = np.array([0.7, 0.6, 0.75, 0.3])
    
    assert np.all(val == np.array(res[1]))
    

if __name__ == '__main__':  
    pytest.main()
