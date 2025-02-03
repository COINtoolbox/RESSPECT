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

import numpy as np
import pytest

from resspect.feature_extractors.light_curve import LightCurve
from resspect.feature_extractors.bazin import BazinFeatureExtractor


@pytest.fixture(scope='function')
def input_lc(test_data_path):
    """ Read an SNPCC light curve. """

    path_to_lc = test_data_path / "DES_SN848233.DAT"
    lc = LightCurve()
    lc.load_snpcc_lc(str(path_to_lc))
    
    return lc


@pytest.fixture(scope='function')
def input_bazin_lc(test_data_path):
    """ Read an SNPCC light curve. """

    path_to_lc = test_data_path / "DES_SN848233.DAT"
    lc = BazinFeatureExtractor()
    lc.load_snpcc_lc(str(path_to_lc))

    return lc


def test_load_snpcc_lc(test_data_path):
    """ Test loading a light curve from SNPCC data. """

    path_to_lc = test_data_path / "DES_SN848233.DAT"

    lc = LightCurve()
    lc.load_snpcc_lc(path_to_lc)

    # expected header for lc.photometry
    header = np.array(['mjd', 'band', 'flux', 'fluxerr', 'SNR', 
                       'MAG', 'MAGERR'])
    
    assert np.all(header == lc.photometry.keys())
       
def test_load_plasticc_lc(test_data_path):
    """ Test loading a light curve from PLAsTiCC. """
    
    path_to_lc = test_data_path / "plasticc_lightcurves.csv.gz"
    lc = LightCurve()
    lc.load_plasticc_lc(str(path_to_lc), snid=229855)
    
    header = np.array(['mjd', 'band', 'flux', 'fluxerr', 'detected_bool'])
    
    assert np.all(header == lc.photometry.keys())
    

def test_conv_flux_mag(input_lc):
    """ Test flux to magnitude conversion. """
    
    flux = input_lc.photometry['flux'].values
    mag = input_lc.conv_flux_mag(flux)
    
    assert np.all(mag > 20)
    

def test_check_queryable(input_bazin_lc):
    """ Test consistency of queryable tests. """
    
    input_bazin_lc.fit_all()
    
    min_mjd = min(input_bazin_lc.photometry['mjd'].values) 
    epochs = [15, 80]
    
    res = {}
    for c in range(1, 3):
        res[c] = {}
        
        for d in epochs:
            res[c][d] = {}  
        
            for f in input_bazin_lc.filters:
                res[c][d][f] = input_bazin_lc.check_queryable(mjd=min_mjd + d,
                                                        filter_lim=24, 
                                                        criteria=c, 
                                                        days_since_last_obs=2,
                                                        filter_cut=f)
            
    fid = {}  
    for c in range(1, 3):
        fid[c] = {}        
        for d in epochs:
            fid[c][d] = {}
    
    for c in range(1, 3):
        for f in input_bazin_lc.filters:
            fid[c][epochs[0]][f] = False
            fid[c][epochs[1]][f] = True
    
    # check results
    check = []
    
    for c in range(1, 3):
        for d in epochs:
            for f in input_bazin_lc.filters:
                check.append(fid[c][d][f] == res[c][d][f])
                
    assert np.all(np.array(check))
    
    
def test_calc_exp_time(input_lc):
    """ Test exposure time calculator. """
    
    min_mjd = min(input_lc.photometry['mjd'].values)
    input_lc.check_queryable(mjd=min_mjd + 80, filter_lim=24, 
                             criteria=1, days_since_last_obs=2,
                             filter_cut='r')
    
    res = input_lc.calc_exp_time(telescope_diam=4, SNR=5,
                                 telescope_name='4m')
    
    assert res > 0
    
    
def test_fit_bazin(input_bazin_lc):
    """ Test bazin fit inside light curve. """
    
    params = input_bazin_lc.fit(band='r')
    
    assert len(params) == 5
    
    
def test_fit_bazin_all(input_bazin_lc):
    """ Test Bazin fit in all filters. """
    
    input_bazin_lc.fit_all()
    l1 = len(input_bazin_lc.features)
    l2 = len(input_bazin_lc.filters) * 5
    
    assert l1 == l2
    
    
def test_evaluate_bazin(input_bazin_lc):
    """ Test if bazin values are non-negative. """
    
    min_mjd = min(input_bazin_lc.photometry['mjd'].values)
    max_mjd = max(input_bazin_lc.photometry['mjd'].values)
    
    # generate a grid of times
    t = np.arange(min_mjd, max_mjd, 0.5)
    
    input_bazin_lc.fit_all()
    flux = input_bazin_lc.evaluate(time=np.random.choice(t, size=5))
    
    res = []
    for f in input_bazin_lc.filters:
        res = res + flux[f]
    
    res = np.array(res)
    
    assert not np.isnan(res).any()
        

if __name__ == '__main__':
    pytest.main()
