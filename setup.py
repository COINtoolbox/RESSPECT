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

import setuptools


setuptools.setup(
    name='resspect',
    version='1.2',
    packages=setuptools.find_packages(),
    py_modules=['resspect/bazin',
                'resspect/build_snpcc_canonical',
                'resspect/classifiers',
                'resspect/database',
                'resspect/fit_lightcurves',
                'resspect/learn_loop',
                'resspect/metrics',
                'resspect/plot_results',
                'resspect/query_strategies',
                'resspect/snana_fits_to_pd',
                'resspect/time_domain',
                'resspect/time_domain_PLAsTiCC'],
    scripts=['resspect/scripts/build_canonical.py',
             'resspect/scripts/build_time_domain.py',
             'resspect/scripts/fit_dataset.py',
             'resspect/scripts/make_metrics_plots.py',
             'resspect/scripts/run_loop.py',
             'resspect/scripts/run_time_domain.py'],
    url='https://github.com/COINtoolbox/resspect/tree/RESSPECT',
    license='GNU3',
    author='The RESSPECT team',
    author_email='contact@cosmostatistics-initiative.org',
    description='resspect - Recommendation System for Spectroscopic Follow-up'
)
